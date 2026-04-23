from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RadioConfig:
    carrier_hz: float = 5.8e9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 27.0
    tx_antenna_gain_dbi: float = 2.0
    rx_antenna_gain_dbi: float = 5.0
    noise_figure_db: float = 6.0


@dataclass(frozen=True)
class PhyConfig:
    subcarrier_spacing_hz: float = 30e3
    sample_rate_hz: float = 30.72e6
    fft_size: int = 1024
    active_prbs: int = 51
    subcarriers_per_prb: int = 12
    cp_fraction: float = 72 / 1024
    symbols_per_frame: int = 14
    sync_symbols_per_frame: int = 2
    dmrs_symbols_per_frame: int = 2
    dmrs_frequency_spacing: int = 6
    packet_overhead_fraction: float = 0.05

    @property
    def active_subcarriers(self) -> int:
        return self.active_prbs * self.subcarriers_per_prb

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.active_subcarriers * self.subcarrier_spacing_hz

    @property
    def guard_band_hz(self) -> float:
        return max(0.0, 20e6 - self.occupied_bandwidth_hz)

    @property
    def usable_data_symbols_per_frame(self) -> int:
        return self.symbols_per_frame - self.sync_symbols_per_frame

    @property
    def pilot_fraction(self) -> float:
        pilot_symbols = self.dmrs_symbols_per_frame / max(1, self.usable_data_symbols_per_frame)
        pilot_subcarriers = 1 / max(1, self.dmrs_frequency_spacing)
        return min(0.5, pilot_symbols * pilot_subcarriers)


@dataclass(frozen=True)
class MCS:
    name: str
    modulation: str
    code_rate: float
    snr_threshold_db: float

    @property
    def bits_per_symbol(self) -> int:
        return {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}[self.modulation]


DEFAULT_MCS_TABLE: Tuple[MCS, ...] = (
    MCS("QPSK r1/2", "QPSK", 0.50, 3.0),
    MCS("QPSK r2/3", "QPSK", 2 / 3, 5.0),
    MCS("16-QAM r1/2", "16-QAM", 0.50, 8.0),
    MCS("16-QAM r2/3", "16-QAM", 2 / 3, 11.0),
    MCS("64-QAM r2/3", "64-QAM", 2 / 3, 16.0),
    MCS("64-QAM r3/4", "64-QAM", 0.75, 19.0),
)


@dataclass
class SweepConfig:
    channel_model: str = "AWGN"
    speed_kmph: float = 0.0
    tdl_profile: str = "TDL-A"
    rms_delay_ns: float = 100.0
    rician_k_db: float = 6.0
    target_rate_mbps: float = 25.0
    mcs_mode: str = "Adaptive"
    mcs_policy: str = "Meet target rate"
    fixed_mcs_name: str = "QPSK r1/2"
    packets_per_point: int = 100
    payload_bytes: int = 1200
    seed: int = 1
    snr_min_db: float = 0.0
    snr_max_db: float = 30.0
    snr_step_db: float = 2.0
    distance_min_m: float = 100.0
    distance_max_m: float = 5000.0
    distance_step_m: float = 250.0
    custom_taps: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SimulationConfig:
    radio: RadioConfig = field(default_factory=RadioConfig)
    phy: PhyConfig = field(default_factory=PhyConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    mcs_table: Tuple[MCS, ...] = DEFAULT_MCS_TABLE


CAMERA_PRESETS_MBPS: Dict[str, float] = {
    "720p30 low latency": 6.0,
    "1080p30": 12.0,
    "1080p60": 25.0,
    "4K30": 55.0,
    "Custom": 25.0,
}
