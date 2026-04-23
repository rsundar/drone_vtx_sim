# LA9310 / VSPA Mapping Notes

The included NXP document is a VSPA-16SP instruction-set manual for LA9310, so this simulator treats it as an implementation reference rather than a waveform specification.

Candidate DSP kernels for later LA9310 work:

- Coarse synchronization: Zadoff-Chu/PSS-style correlation, peak search, coarse CFO estimate.
- OFDM front end: CP removal, 1024-point FFT, active-bin extraction, guard/DC null handling.
- Fine synchronization: secondary training symbol, residual CFO/common phase correction.
- Pilot processing: DMRS-like pilot extraction, channel estimate interpolation, time tracking.
- Equalization: one-tap per-subcarrier equalizer for flat and TDL channels.
- Demapping: QPSK, 16-QAM, 64-QAM hard/soft demap and LLR generation.
- FEC: replace the current reference FEC abstraction with production LDPC encode/decode.
- Metrics: BER/PER counters, MCS decision, throughput/range margin.

Fixed-point hook points should be added at synchronizer output, FFT output, channel estimates, equalized symbols, demapper LLRs, and LDPC decoder input.

