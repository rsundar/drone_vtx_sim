from __future__ import annotations

from .models import CAMERA_PRESETS_MBPS, MCS, PhyConfig


def gross_phy_rate_mbps(phy: PhyConfig, mcs: MCS) -> float:
    symbol_rate = phy.subcarrier_spacing_hz / (1 + phy.cp_fraction)
    bits_per_ofdm_symbol = phy.active_subcarriers * mcs.bits_per_symbol * mcs.code_rate
    return bits_per_ofdm_symbol * symbol_rate / 1e6


def usable_throughput_mbps(phy: PhyConfig, mcs: MCS, per: float = 0.0) -> float:
    sync_factor = phy.usable_data_symbols_per_frame / phy.symbols_per_frame
    pilot_factor = 1.0 - phy.pilot_fraction
    packet_factor = 1.0 - phy.packet_overhead_fraction
    goodput_factor = max(0.0, 1.0 - per)
    return gross_phy_rate_mbps(phy, mcs) * sync_factor * pilot_factor * packet_factor * goodput_factor


def meets_target_rate(phy: PhyConfig, mcs: MCS, target_rate_mbps: float, per: float = 0.0) -> bool:
    return usable_throughput_mbps(phy, mcs, per) >= target_rate_mbps


def throughput_margin_mbps(phy: PhyConfig, mcs: MCS, target_rate_mbps: float, per: float = 0.0) -> float:
    return usable_throughput_mbps(phy, mcs, per) - target_rate_mbps


def camera_presets() -> dict[str, float]:
    return dict(CAMERA_PRESETS_MBPS)

