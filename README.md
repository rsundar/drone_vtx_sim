# Drone VTX PHY Simulator

Python-backed drone VTX PHY simulator with a local browser UI, 5G-like 30 kHz OFDM numerology, guard bands, pilot/sync modeling, AWGN/Rayleigh/Rician/TDL Doppler channels, range budget, and target camera data-rate checks.

## Quick Start

Run from the repository root:

```bash
python3 scripts/run_ui.py
```

The command starts a local Python HTTP server and opens the UI in your browser. If the browser does not open automatically, use the printed URL, usually `http://127.0.0.1:8765/`.

Command-line sweeps:

```bash
PYTHONPATH=src python3 scripts/run_ber_sweep.py --channel tdl --tdl-profile TDL-C --speed-kmph 100 --target-rate-mbps 25
PYTHONPATH=src python3 scripts/run_range_sweep.py --channel tdl --tdl-profile TDL-A --max-distance-m 5000 --target-rate-mbps 25
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Current Scope

This is a floating-point reference simulator. The FEC module exposes a replaceable LDPC-style interface and uses rate-dependent coding behavior for BER/PER sweeps; it is not yet a standards-grade LDPC implementation.

The UI has two tabs:

- `BER/PER Curves`: SNR sweeps with BER, PER, usable throughput, and target-rate checks.
- `Range`: distance sweeps with SNR, received power, path loss, MCS, throughput margin, and outage status.
