from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FecCodec:
    """Reference FEC interface.

    V1 uses this interface for throughput and curve shaping. The encode/decode
    path is intentionally simple so a standards-grade LDPC backend can replace
    it without touching sweeps or UI.
    """

    code_rate: float

    @property
    def coding_gain_db(self) -> float:
        if self.code_rate <= 0.51:
            return 4.5
        if self.code_rate <= 0.68:
            return 3.0
        return 2.0

    def encode(self, bits: np.ndarray) -> np.ndarray:
        parity_count = max(0, int(round(len(bits) * (1 / self.code_rate - 1))))
        if parity_count == 0:
            return bits.copy()
        parity = np.resize(bits, parity_count)
        return np.concatenate([bits, parity])

    def decode_hard(self, bits: np.ndarray, payload_len: int) -> np.ndarray:
        return bits[:payload_len].copy()

