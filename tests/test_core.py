import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from drone_vtx_sim.channels import TDL_PRESETS, max_doppler_hz, normalize_taps
from drone_vtx_sim.fec import FecCodec
from drone_vtx_sim.link_budget import eirp_dbm, noise_floor_dbm
from drone_vtx_sim.models import CAMERA_PRESETS_MBPS, DEFAULT_MCS_TABLE, PhyConfig, RadioConfig, SimulationConfig, SweepConfig
from drone_vtx_sim.phy import active_subcarrier_indices, demodulate_hard, demodulate_llr, dmrs_positions, modulate_bits, ofdm_demodulate, ofdm_modulate
from drone_vtx_sim.sweeps import run_ber_sweep, run_range_sweep
from drone_vtx_sim.throughput import usable_throughput_mbps


class CoreTests(unittest.TestCase):
    def test_numerology_and_guard(self):
        phy = PhyConfig()
        self.assertEqual(phy.fft_size, 1024)
        self.assertAlmostEqual(phy.sample_rate_hz / phy.fft_size, 30_000.0)
        self.assertEqual(phy.active_subcarriers, 612)
        self.assertAlmostEqual(phy.occupied_bandwidth_hz, 18.36e6)
        self.assertAlmostEqual(phy.guard_band_hz, 1.64e6)
        self.assertEqual(len(active_subcarrier_indices(phy)), 612)
        self.assertNotIn(0, set(active_subcarrier_indices(phy)))

    def test_qam_round_trip(self):
        rng = np.random.default_rng(1)
        for modulation in ["QPSK", "16-QAM", "64-QAM"]:
            bits_per_symbol = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}[modulation]
            bits = rng.integers(0, 2, size=bits_per_symbol * 100, dtype=np.uint8)
            recovered = demodulate_hard(modulate_bits(bits, modulation), modulation)
            np.testing.assert_array_equal(bits, recovered[: len(bits)])

    def test_ldpc_parity_and_decode(self):
        rng = np.random.default_rng(7)
        for rate in [0.5, 2 / 3, 0.75]:
            codec = FecCodec(rate)
            bits = rng.integers(0, 2, size=codec.info_length, dtype=np.uint8)
            cw = codec.encode(bits)
            np.testing.assert_array_equal(codec.syndrome(cw), np.zeros(codec.parity_checks.shape[0], dtype=np.uint8))
            symbols = modulate_bits(cw, "QPSK")
            noise_var = 1e-4
            noise = (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape)) * np.sqrt(noise_var / 2)
            llr = demodulate_llr(symbols + noise, "QPSK", noise_var)[: codec.codeword_length]
            decoded = codec.decode_llr(llr, max_iters=20)
            np.testing.assert_array_equal(decoded, bits)

    def test_ofdm_round_trip(self):
        phy = PhyConfig()
        rng = np.random.default_rng(2)
        symbols = rng.normal(size=phy.active_subcarriers) + 1j * rng.normal(size=phy.active_subcarriers)
        recovered = ofdm_demodulate(ofdm_modulate(symbols, phy), phy)
        np.testing.assert_allclose(symbols, recovered, atol=1e-10)

    def test_dmrs_positions(self):
        phy = PhyConfig()
        pos = dmrs_positions(phy)
        self.assertEqual(pos[0], 0)
        self.assertLess(pos[-1], phy.active_subcarriers)
        self.assertEqual(pos[1] - pos[0], phy.dmrs_frequency_spacing)

    def test_tdl_and_doppler(self):
        self.assertAlmostEqual(max_doppler_hz(100, 5.8e9), 537.4, delta=1.0)
        normalized = normalize_taps(TDL_PRESETS["TDL-A"])
        total_power = sum(10 ** (power_db / 10) for _, power_db in normalized)
        self.assertAlmostEqual(total_power, 1.0)

    def test_link_budget(self):
        radio = RadioConfig()
        self.assertAlmostEqual(eirp_dbm(radio), 29.0)
        self.assertAlmostEqual(noise_floor_dbm(20e6, 6), -95.0, delta=0.2)

    def test_throughput_and_presets(self):
        phy = PhyConfig()
        low = usable_throughput_mbps(phy, DEFAULT_MCS_TABLE[0], per=0.0)
        high = usable_throughput_mbps(phy, DEFAULT_MCS_TABLE[-1], per=0.0)
        impaired = usable_throughput_mbps(phy, DEFAULT_MCS_TABLE[-1], per=0.5)
        self.assertLess(low, high)
        self.assertLess(impaired, high)
        self.assertEqual(CAMERA_PRESETS_MBPS["1080p60"], 25.0)

    def test_sweeps_behave(self):
        cfg = SimulationConfig(sweep=SweepConfig(snr_min_db=0, snr_max_db=12, snr_step_db=6, target_rate_mbps=6))
        awgn = run_ber_sweep(cfg)
        self.assertGreater(awgn["per"].iloc[0], awgn["per"].iloc[-1])
        fixed_cfg = SimulationConfig(
            sweep=SweepConfig(
                mcs_mode="Fixed",
                fixed_mcs_name="64-QAM r3/4",
                snr_min_db=30,
                snr_max_db=30,
                snr_step_db=1,
            )
        )
        self.assertEqual(run_ber_sweep(fixed_cfg)["selected_mcs"].iloc[0], "64-QAM r3/4")
        max_cfg = SimulationConfig(
            sweep=SweepConfig(
                mcs_mode="Adaptive",
                mcs_policy="Max throughput",
                snr_min_db=30,
                snr_max_db=30,
                snr_step_db=1,
                target_rate_mbps=6,
            )
        )
        self.assertEqual(run_ber_sweep(max_cfg)["selected_mcs"].iloc[0], "64-QAM r3/4")
        rician_cfg = SimulationConfig(sweep=SweepConfig(channel_model="Rician + Doppler", snr_min_db=12, snr_max_db=12, snr_step_db=1))
        self.assertEqual(len(run_ber_sweep(rician_cfg)), 1)
        range_cfg = SimulationConfig(sweep=SweepConfig(distance_min_m=100, distance_max_m=1000, distance_step_m=450, target_rate_mbps=25))
        range_df = run_range_sweep(range_cfg)
        self.assertGreater(range_df["snr_db"].iloc[0], range_df["snr_db"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
