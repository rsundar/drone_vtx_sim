from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.special import erfc

from .channels import channel_effective_snr, max_doppler_hz, rms_delay_spread_ns, TDL_PRESETS
from .fec import FecCodec
from .link_budget import free_space_path_loss_db, noise_floor_dbm, received_power_dbm, snr_at_distance_db
from .models import MCS, SimulationConfig
from .throughput import usable_throughput_mbps

ProgressCallback = Callable[[int, int, str], None]


def snr_points(config: SimulationConfig) -> np.ndarray:
    sweep = config.sweep
    return np.arange(sweep.snr_min_db, sweep.snr_max_db + 0.001, sweep.snr_step_db)


def distance_points(config: SimulationConfig) -> np.ndarray:
    sweep = config.sweep
    return np.arange(sweep.distance_min_m, sweep.distance_max_m + 0.001, sweep.distance_step_m)


def raw_ber_awgn(snr_db: float, mcs: MCS) -> float:
    gamma = 10 ** (snr_db / 10)
    if mcs.modulation == "QPSK":
        return 0.5 * erfc(math.sqrt(gamma))
    m = 2 ** mcs.bits_per_symbol
    return min(0.5, (4 / mcs.bits_per_symbol) * (1 - 1 / math.sqrt(m)) * 0.5 * erfc(math.sqrt(3 * mcs.bits_per_symbol * gamma / (2 * (m - 1)))))


def post_fec_ber(pre_fec_ber: float, effective_snr_db: float, mcs: MCS) -> float:
    codec = FecCodec(mcs.code_rate)
    margin = effective_snr_db + codec.coding_gain_db - mcs.snr_threshold_db
    waterfall = 1 / (1 + math.exp(1.8 * margin))
    floor = 1e-9 if margin > 3 else 0.0
    return min(pre_fec_ber, pre_fec_ber * waterfall + floor)


def packet_error_rate(bit_error_rate: float, payload_bytes: int) -> float:
    bits = payload_bytes * 8
    return 1 - (1 - min(max(bit_error_rate, 0.0), 0.5)) ** bits


def nominal_effective_snr_db(config: SimulationConfig, snr_db: float) -> float:
    sweep = config.sweep
    state = channel_effective_snr(
        snr_db,
        sweep.channel_model,
        sweep.speed_kmph,
        config.radio.carrier_hz,
        sweep.tdl_profile,
        sweep.rms_delay_ns,
        sweep.rician_k_db,
        sweep.custom_taps,
        rng=None,
    )
    # Use an average-fade planning margin for adaptive MCS selection. The
    # packet-level simulation below still samples fading and averages PER.
    fading_margin = 0.0
    if sweep.channel_model.startswith("Flat Rayleigh"):
        fading_margin = 2.0
    elif sweep.channel_model.startswith("Rician"):
        fading_margin = max(0.2, 1.5 - 0.12 * sweep.rician_k_db)
    elif sweep.channel_model.startswith("TDL"):
        fading_margin = 3.0
    return snr_db - state.delay_penalty_db - state.doppler_penalty_db - fading_margin


def estimated_per_for_mcs(config: SimulationConfig, snr_db: float, mcs: MCS) -> float:
    pre = raw_ber_awgn(snr_db, mcs)
    post = post_fec_ber(pre, snr_db, mcs)
    return packet_error_rate(post, config.sweep.payload_bytes)


def select_mcs(config: SimulationConfig, snr_db: float) -> MCS:
    target = config.sweep.target_rate_mbps
    if config.sweep.mcs_mode != "Adaptive":
        return next((m for m in config.mcs_table if m.name == config.sweep.fixed_mcs_name), config.mcs_table[0])
    feasible: List[MCS] = []
    for mcs in config.mcs_table:
        estimated_per = estimated_per_for_mcs(config, snr_db, mcs)
        if (
            snr_db >= mcs.snr_threshold_db
            and estimated_per <= 0.05
            and usable_throughput_mbps(config.phy, mcs, estimated_per) >= target
        ):
            feasible.append(mcs)
    if feasible:
        if config.sweep.mcs_policy == "Max throughput":
            return feasible[-1]
        return feasible[0]
    return config.mcs_table[0]


def simulate_point(config: SimulationConfig, snr_db: float, rng: np.random.Generator) -> dict:
    sweep = config.sweep
    mcs = select_mcs(config, nominal_effective_snr_db(config, snr_db))
    packets = max(1, int(sweep.packets_per_point))
    effective_values = []
    pre_values = []
    post_values = []
    per_values = []
    fading_values = []
    delay_values = []
    doppler_values = []
    for _ in range(packets):
        state = channel_effective_snr(
            snr_db,
            sweep.channel_model,
            sweep.speed_kmph,
            config.radio.carrier_hz,
            sweep.tdl_profile,
            sweep.rms_delay_ns,
            sweep.rician_k_db,
            sweep.custom_taps,
            rng,
        )
        pre_i = raw_ber_awgn(state.effective_snr_db, mcs)
        post_i = post_fec_ber(pre_i, state.effective_snr_db, mcs)
        per_i = packet_error_rate(post_i, sweep.payload_bytes)
        effective_values.append(state.effective_snr_db)
        pre_values.append(pre_i)
        post_values.append(post_i)
        per_values.append(per_i)
        fading_values.append(state.fading_loss_db)
        delay_values.append(state.delay_penalty_db)
        doppler_values.append(state.doppler_penalty_db)

    effective_snr = float(np.mean(effective_values))
    pre = float(np.mean(pre_values))
    post = float(np.mean(post_values))
    per = float(np.mean(per_values))
    throughput = usable_throughput_mbps(config.phy, mcs, per)
    margin = throughput - sweep.target_rate_mbps
    return {
        "snr_db": snr_db,
        "effective_snr_db": effective_snr,
        "selected_mcs": mcs.name,
        "modulation": mcs.modulation,
        "code_rate": mcs.code_rate,
        "pre_fec_ber": pre,
        "post_fec_ber": post,
        "per": per,
        "usable_throughput_mbps": throughput,
        "target_rate_mbps": sweep.target_rate_mbps,
        "throughput_margin_mbps": margin,
        "meets_rate": throughput >= sweep.target_rate_mbps,
        "fading_loss_db": float(np.mean(fading_values)),
        "delay_penalty_db": float(np.mean(delay_values)),
        "doppler_penalty_db": float(np.mean(doppler_values)),
        "packets_averaged": packets,
        "max_doppler_hz": max_doppler_hz(sweep.speed_kmph, config.radio.carrier_hz),
    }


def run_ber_sweep(config: SimulationConfig, progress: ProgressCallback | None = None, cancel: Callable[[], bool] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(config.sweep.seed)
    points = list(snr_points(config))
    rows = []
    for idx, snr_db in enumerate(points, start=1):
        if cancel and cancel():
            break
        rows.append(simulate_point(config, float(snr_db), rng))
        if progress:
            progress(idx, len(points), f"SNR {snr_db:.1f} dB")
    return pd.DataFrame(rows)


def run_range_sweep(config: SimulationConfig, progress: ProgressCallback | None = None, cancel: Callable[[], bool] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(config.sweep.seed)
    points = list(distance_points(config))
    rows = []
    for idx, distance_m in enumerate(points, start=1):
        if cancel and cancel():
            break
        snr_db = snr_at_distance_db(float(distance_m), config.radio)
        row = simulate_point(config, snr_db, rng)
        row.update(
            {
                "distance_m": float(distance_m),
                "path_loss_db": free_space_path_loss_db(float(distance_m), config.radio.carrier_hz),
                "received_power_dbm": received_power_dbm(float(distance_m), config.radio),
                "noise_floor_dbm": noise_floor_dbm(config.radio.bandwidth_hz, config.radio.noise_figure_db),
                "outage": (not row["meets_rate"]) or row["per"] > 0.1,
            }
        )
        rows.append(row)
        if progress:
            progress(idx, len(points), f"Distance {distance_m:.0f} m")
    return pd.DataFrame(rows)


def tdl_profile_delay_spread_ns(profile: str) -> float:
    return rms_delay_spread_ns(TDL_PRESETS[profile])
