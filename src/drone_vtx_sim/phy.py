from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import PhyConfig


def active_subcarrier_indices(config: PhyConfig) -> np.ndarray:
    """Return FFT-bin indices for active subcarriers, excluding DC."""
    count = config.active_subcarriers
    half = count // 2
    negative = np.arange(config.fft_size - half, config.fft_size)
    positive = np.arange(1, half + 1)
    return np.concatenate([negative, positive])


def guard_subcarrier_count(config: PhyConfig) -> int:
    return config.fft_size - config.active_subcarriers - 1


def qam_constellation(modulation: str) -> np.ndarray:
    if modulation == "QPSK":
        pts = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j])
    elif modulation == "16-QAM":
        levels = np.array([-3, -1, 3, 1])
        pts = np.array([i + 1j * q for i in levels for q in levels])
    elif modulation == "64-QAM":
        levels = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
        pts = np.array([i + 1j * q for i in levels for q in levels])
    else:
        raise ValueError(f"Unsupported modulation {modulation}")
    return pts / math.sqrt(np.mean(np.abs(pts) ** 2))


def modulate_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    bits_per_symbol = int(math.log2(len(qam_constellation(modulation))))
    if len(bits) % bits_per_symbol:
        pad = bits_per_symbol - (len(bits) % bits_per_symbol)
        bits = np.pad(bits, (0, pad))
    grouped = bits.reshape(-1, bits_per_symbol)
    indices = grouped.dot(1 << np.arange(bits_per_symbol - 1, -1, -1))
    return qam_constellation(modulation)[indices]


def demodulate_hard(symbols: np.ndarray, modulation: str) -> np.ndarray:
    const = qam_constellation(modulation)
    distances = np.abs(symbols[:, None] - const[None, :])
    indices = np.argmin(distances, axis=1)
    bits_per_symbol = int(math.log2(len(const)))
    return ((indices[:, None] & (1 << np.arange(bits_per_symbol - 1, -1, -1))) > 0).astype(np.uint8).ravel()


def demodulate_llr(symbols: np.ndarray, modulation: str, noise_var: float) -> np.ndarray:
    const = qam_constellation(modulation)
    bits_per_symbol = int(math.log2(len(const)))
    labels = ((np.arange(len(const))[:, None] & (1 << np.arange(bits_per_symbol - 1, -1, -1))) > 0).astype(np.uint8)
    noise_var = max(float(noise_var), 1e-9)
    llrs = np.empty((len(symbols), bits_per_symbol), dtype=float)
    for bit_idx in range(bits_per_symbol):
        s0 = const[labels[:, bit_idx] == 0]
        s1 = const[labels[:, bit_idx] == 1]
        d0 = np.min(np.abs(symbols[:, None] - s0[None, :]) ** 2, axis=1)
        d1 = np.min(np.abs(symbols[:, None] - s1[None, :]) ** 2, axis=1)
        llrs[:, bit_idx] = (d1 - d0) / noise_var
    return llrs.ravel()


def ofdm_modulate(active_symbols: np.ndarray, config: PhyConfig) -> np.ndarray:
    bins = np.zeros(config.fft_size, dtype=np.complex128)
    active = active_subcarrier_indices(config)
    if len(active_symbols) != len(active):
        raise ValueError(f"Expected {len(active)} active symbols, got {len(active_symbols)}")
    bins[active] = active_symbols
    time = np.fft.ifft(bins) * math.sqrt(config.fft_size)
    cp_len = int(round(config.fft_size * config.cp_fraction))
    return np.concatenate([time[-cp_len:], time])


def ofdm_demodulate(samples: np.ndarray, config: PhyConfig) -> np.ndarray:
    cp_len = int(round(config.fft_size * config.cp_fraction))
    useful = samples[cp_len : cp_len + config.fft_size]
    bins = np.fft.fft(useful) / math.sqrt(config.fft_size)
    return bins[active_subcarrier_indices(config)]


def zadoff_chu(root: int, length: int) -> np.ndarray:
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)


def estimate_frame_start(samples: np.ndarray, preamble: np.ndarray) -> int:
    corr = np.abs(np.correlate(samples, preamble.conjugate(), mode="valid"))
    return int(np.argmax(corr))


def estimate_cfo_from_repetition(samples: np.ndarray, half_len: int, sample_rate_hz: float) -> float:
    first = samples[:half_len]
    second = samples[half_len : 2 * half_len]
    phase = np.angle(np.vdot(first, second))
    return float(phase * sample_rate_hz / (2 * np.pi * half_len))


def dmrs_positions(config: PhyConfig) -> np.ndarray:
    return np.arange(0, config.active_subcarriers, config.dmrs_frequency_spacing)


def interpolate_channel(pilot_positions: Iterable[int], pilot_estimates: np.ndarray, config: PhyConfig) -> np.ndarray:
    positions = np.asarray(list(pilot_positions), dtype=float)
    x = np.arange(config.active_subcarriers, dtype=float)
    real = np.interp(x, positions, pilot_estimates.real)
    imag = np.interp(x, positions, pilot_estimates.imag)
    return real + 1j * imag
