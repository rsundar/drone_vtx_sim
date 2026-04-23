from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

C_LIGHT = 299_792_458.0


TDL_PRESETS: Dict[str, List[Tuple[float, float]]] = {
    "TDL-A": [(0.0, 0.0), (30.0, -4.0), (70.0, -8.0), (90.0, -10.0), (110.0, -15.0)],
    "TDL-B": [(0.0, -2.0), (20.0, 0.0), (80.0, -3.0), (160.0, -7.0), (230.0, -12.0)],
    "TDL-C": [(0.0, -4.0), (60.0, -1.0), (140.0, 0.0), (260.0, -5.0), (420.0, -9.0), (700.0, -14.0)],
}


def max_doppler_hz(speed_kmph: float, carrier_hz: float) -> float:
    return speed_kmph * 1000 / 3600 * carrier_hz / C_LIGHT


def normalize_taps(taps: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    powers = np.array([10 ** (p_db / 10) for _, p_db in taps], dtype=float)
    powers = powers / powers.sum()
    return [(delay, 10 * math.log10(power)) for (delay, _), power in zip(taps, powers)]


def rms_delay_spread_ns(taps: Sequence[Tuple[float, float]]) -> float:
    powers = np.array([10 ** (p_db / 10) for _, p_db in taps], dtype=float)
    powers = powers / powers.sum()
    delays = np.array([delay for delay, _ in taps], dtype=float)
    mean_delay = float(np.sum(delays * powers))
    return float(np.sqrt(np.sum(((delays - mean_delay) ** 2) * powers)))


@dataclass(frozen=True)
class ChannelState:
    fading_loss_db: float
    delay_penalty_db: float
    doppler_penalty_db: float
    effective_snr_db: float


def channel_effective_snr(
    snr_db: float,
    channel_model: str,
    speed_kmph: float,
    carrier_hz: float,
    tdl_profile: str = "TDL-A",
    rms_delay_ns: float = 100.0,
    rician_k_db: float = 6.0,
    custom_taps: Sequence[Tuple[float, float]] = (),
    rng: np.random.Generator | None = None,
) -> ChannelState:
    rng = rng or np.random.default_rng()
    doppler = max_doppler_hz(speed_kmph, carrier_hz)
    norm_doppler = min(1.0, doppler / 537.0)
    doppler_penalty = 2.4 * norm_doppler if "Doppler" in channel_model or "TDL" in channel_model else 0.0
    fading_loss = 0.0
    delay_penalty = 0.0

    if channel_model.startswith("Flat Rayleigh"):
        amp = max(1e-4, rng.rayleigh(scale=1 / math.sqrt(2)))
        fading_loss = -20 * math.log10(amp)
    elif channel_model.startswith("Rician"):
        k_linear = 10 ** (rician_k_db / 10)
        los = math.sqrt(k_linear / (k_linear + 1))
        scatter_sigma = math.sqrt(1 / (2 * (k_linear + 1)))
        h = los + scatter_sigma * (rng.normal() + 1j * rng.normal())
        fading_loss = -20 * math.log10(max(1e-4, abs(h)))
    elif channel_model.startswith("TDL"):
        taps = list(custom_taps) if tdl_profile == "Custom TDL" and custom_taps else TDL_PRESETS.get(tdl_profile, TDL_PRESETS["TDL-A"])
        spread = rms_delay_ns if rms_delay_ns > 0 else rms_delay_spread_ns(taps)
        fading_loss = max(0.0, rng.normal(3.0, 2.0))
        delay_penalty = min(6.0, spread / 120.0)

    effective = snr_db - fading_loss - delay_penalty - doppler_penalty
    return ChannelState(fading_loss, delay_penalty, doppler_penalty, effective)


def add_awgn(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = np.mean(np.abs(samples) ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = (rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape)) * math.sqrt(noise_power / 2)
    return samples + noise
