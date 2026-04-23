from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
import pandas as pd

from .channels import channel_effective_snr, max_doppler_hz, rms_delay_spread_ns, TDL_PRESETS
from .fec import FecCodec
from .link_budget import free_space_path_loss_db, noise_floor_dbm, received_power_dbm, snr_at_distance_db
from .models import MCS, SimulationConfig
from .phy import demodulate_hard, demodulate_llr, modulate_bits
from .throughput import usable_throughput_mbps

ProgressCallback = Callable[[int, int, str], None]


def snr_points(config: SimulationConfig) -> np.ndarray:
    sweep = config.sweep
    return np.arange(sweep.snr_min_db, sweep.snr_max_db + 0.001, sweep.snr_step_db)


def distance_points(config: SimulationConfig) -> np.ndarray:
    sweep = config.sweep
    return np.arange(sweep.distance_min_m, sweep.distance_max_m + 0.001, sweep.distance_step_m)

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
    seed = int(round(snr_db * 10)) + 1000 * (hash(mcs.name) & 0xFFFF)
    metrics = simulate_ldpc_blocks(FecCodec(mcs.code_rate), mcs, snr_db, 10, config.sweep.payload_bytes, np.random.default_rng(seed))
    return float(metrics["per"])


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


def simulate_ldpc_blocks(
    codec: FecCodec,
    mcs: MCS,
    effective_snr_db: float,
    samples: int,
    payload_bytes: int,
    rng: np.random.Generator,
) -> dict:
    snr_linear = 10 ** (effective_snr_db / 10)
    noise_var = 1.0 / max(snr_linear, 1e-9)
    total_coded_errors = 0
    total_coded_bits = 0
    total_info_errors = 0
    total_info_bits = 0
    block_errors = 0

    for _ in range(max(1, samples)):
        info_bits = rng.integers(0, 2, size=codec.info_length, dtype=np.uint8)
        codeword = codec.encode(info_bits)
        tx = modulate_bits(codeword, mcs.modulation)
        noise = (rng.normal(size=tx.shape) + 1j * rng.normal(size=tx.shape)) * math.sqrt(noise_var / 2)
        rx = tx + noise
        hard = demodulate_hard(rx, mcs.modulation)[: codec.codeword_length]
        llr = demodulate_llr(rx, mcs.modulation, noise_var)[: codec.codeword_length]
        decoded = codec.decode_llr(llr)

        total_coded_errors += int(np.count_nonzero(hard != codeword))
        total_coded_bits += int(codeword.size)
        total_info_errors += int(np.count_nonzero(decoded != info_bits))
        total_info_bits += int(info_bits.size)
        if np.any(decoded != info_bits):
            block_errors += 1

    pre = total_coded_errors / max(1, total_coded_bits)
    post = total_info_errors / max(1, total_info_bits)
    blocks_per_packet = int(math.ceil((payload_bytes * 8) / codec.info_length))
    bler = block_errors / max(1, samples)
    per = 1 - (1 - bler) ** blocks_per_packet
    return {"pre_fec_ber": pre, "post_fec_ber": post, "bler": bler, "per": per}


def simulate_point(config: SimulationConfig, snr_db: float, rng: np.random.Generator) -> dict:
    sweep = config.sweep
    mcs = select_mcs(config, nominal_effective_snr_db(config, snr_db))
    codec = FecCodec(mcs.code_rate)
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
        metrics = simulate_ldpc_blocks(codec, mcs, state.effective_snr_db, 1, sweep.payload_bytes, rng)
        effective_values.append(state.effective_snr_db)
        pre_values.append(metrics["pre_fec_ber"])
        post_values.append(metrics["post_fec_ber"])
        per_values.append(metrics["per"])
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
