from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class _LdpcSpec:
    code_rate: float
    codeword_length: int
    info_length: int
    column_weight: int
    seed: int


SPECS = {
    "1/2": _LdpcSpec(0.5, 192, 96, 3, 11),
    "2/3": _LdpcSpec(2 / 3, 192, 128, 2, 22),
    "3/4": _LdpcSpec(0.75, 192, 144, 2, 33),
}


def _rate_key(code_rate: float) -> str:
    if abs(code_rate - 0.5) < 1e-6:
        return "1/2"
    if abs(code_rate - 2 / 3) < 1e-6:
        return "2/3"
    if abs(code_rate - 0.75) < 1e-6:
        return "3/4"
    raise ValueError(f"Unsupported LDPC rate {code_rate}")


def _build_sparse_a(rows: int, cols: int, column_weight: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((rows, cols), dtype=np.uint8)
    row_weight = np.zeros(rows, dtype=int)
    for col in range(cols):
        # Favor lightly loaded rows while keeping placement deterministic.
        order = np.argsort(row_weight + 0.25 * rng.random(rows))
        chosen = order[:column_weight]
        a[chosen, col] = 1
        row_weight[chosen] += 1
    # Avoid zero rows.
    zero_rows = np.flatnonzero(a.sum(axis=1) == 0)
    for row in zero_rows:
        col = int(rng.integers(0, cols))
        a[row, col] = 1
    return a


def _build_dual_diagonal(size: int) -> np.ndarray:
    t = np.eye(size, dtype=np.uint8)
    t[1:, :-1] ^= np.eye(size - 1, dtype=np.uint8)
    return t


@lru_cache(maxsize=None)
def _family(rate_key: str) -> dict[str, object]:
    spec = SPECS[rate_key]
    m = spec.codeword_length - spec.info_length
    a = _build_sparse_a(m, spec.info_length, spec.column_weight, spec.seed)
    t = _build_dual_diagonal(m)
    h = np.concatenate([a, t], axis=1)
    row_neighbors = [np.flatnonzero(h[row]).astype(np.int32) for row in range(h.shape[0])]
    col_neighbors = [np.flatnonzero(h[:, col]).astype(np.int32) for col in range(h.shape[1])]
    return {
        "spec": spec,
        "a": a,
        "t": t,
        "h": h,
        "row_neighbors": row_neighbors,
        "col_neighbors": col_neighbors,
    }


@dataclass(frozen=True)
class FecCodec:
    """Actual LDPC encoder/decoder used by the sweep engine.

    This is a small structured LDPC family with dual-diagonal parity part for
    cheap systematic encoding and normalized min-sum iterative decoding.
    """

    code_rate: float

    @property
    def _impl(self) -> dict[str, object]:
        return _family(_rate_key(self.code_rate))

    @property
    def codeword_length(self) -> int:
        return int(self._impl["spec"].codeword_length)  # type: ignore[union-attr]

    @property
    def info_length(self) -> int:
        return int(self._impl["spec"].info_length)  # type: ignore[union-attr]

    @property
    def parity_checks(self) -> np.ndarray:
        return self._impl["h"]  # type: ignore[return-value]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if bits.size != self.info_length:
            raise ValueError(f"LDPC encoder expects {self.info_length} info bits, got {bits.size}")
        a = self._impl["a"]  # type: ignore[assignment]
        rhs = (a @ bits) & 1
        parity = np.zeros(self.codeword_length - self.info_length, dtype=np.uint8)
        parity[0] = rhs[0]
        for idx in range(1, parity.size):
            parity[idx] = rhs[idx] ^ parity[idx - 1]
        codeword = np.concatenate([bits, parity]).astype(np.uint8)
        return codeword

    def syndrome(self, codeword: np.ndarray) -> np.ndarray:
        c = np.asarray(codeword, dtype=np.uint8).ravel()
        return (self.parity_checks @ c) & 1

    def decode_llr(self, llr: np.ndarray, max_iters: int = 24, alpha: float = 0.8) -> np.ndarray:
        llr = np.asarray(llr, dtype=float).ravel()
        if llr.size != self.codeword_length:
            raise ValueError(f"LDPC decoder expects {self.codeword_length} LLRs, got {llr.size}")
        h = self.parity_checks
        row_neighbors = self._impl["row_neighbors"]  # type: ignore[assignment]
        col_neighbors = self._impl["col_neighbors"]  # type: ignore[assignment]
        m, n = h.shape
        q = np.zeros((m, n), dtype=float)
        r = np.zeros((m, n), dtype=float)
        for row, cols in enumerate(row_neighbors):
            q[row, cols] = llr[cols]

        posterior = llr.copy()
        for _ in range(max_iters):
            for row, cols in enumerate(row_neighbors):
                vals = q[row, cols]
                signs = np.where(vals >= 0.0, 1.0, -1.0)
                absvals = np.abs(vals)
                if absvals.size == 1:
                    r[row, cols[0]] = alpha * signs[0] * absvals[0]
                    continue
                min_idx = int(np.argmin(absvals))
                min1 = absvals[min_idx]
                temp = absvals.copy()
                temp[min_idx] = np.inf
                min2 = float(np.min(temp))
                sign_prod = float(np.prod(signs))
                for local_idx, col in enumerate(cols):
                    mag = min2 if local_idx == min_idx else min1
                    r[row, col] = alpha * sign_prod * signs[local_idx] * mag

            posterior = llr + r.sum(axis=0)
            hard = (posterior < 0.0).astype(np.uint8)
            if np.all(((h @ hard) & 1) == 0):
                return hard[: self.info_length]

            for col, rows in enumerate(col_neighbors):
                total = posterior[col]
                for row in rows:
                    q[row, col] = total - r[row, col]

        return (posterior < 0.0).astype(np.uint8)[: self.info_length]

