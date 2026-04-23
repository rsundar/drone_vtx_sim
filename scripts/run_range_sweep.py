#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_vtx_sim.models import SimulationConfig, SweepConfig
from drone_vtx_sim.sweeps import run_range_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Drone VTX range sweep.")
    parser.add_argument("--channel", choices=["awgn", "rayleigh", "rician", "tdl"], default="tdl")
    parser.add_argument("--tdl-profile", default="TDL-A")
    parser.add_argument("--speed-kmph", type=float, default=0.0)
    parser.add_argument("--target-rate-mbps", type=float, default=25.0)
    parser.add_argument("--min-distance-m", type=float, default=100.0)
    parser.add_argument("--max-distance-m", type=float, default=5000.0)
    parser.add_argument("--step-m", type=float, default=250.0)
    parser.add_argument("--csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel_map = {
        "awgn": "AWGN",
        "rayleigh": "Flat Rayleigh + Doppler",
        "rician": "Rician + Doppler",
        "tdl": "TDL Multipath + Doppler",
    }
    cfg = SimulationConfig(
        sweep=SweepConfig(
            channel_model=channel_map[args.channel],
            tdl_profile=args.tdl_profile,
            speed_kmph=args.speed_kmph,
            target_rate_mbps=args.target_rate_mbps,
            distance_min_m=args.min_distance_m,
            distance_max_m=args.max_distance_m,
            distance_step_m=args.step_m,
        )
    )
    df = run_range_sweep(cfg)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
