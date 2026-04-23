from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from .channels import TDL_PRESETS, max_doppler_hz, rms_delay_spread_ns
from .link_budget import eirp_dbm
from .models import CAMERA_PRESETS_MBPS, DEFAULT_MCS_TABLE, PhyConfig, RadioConfig, SimulationConfig, SweepConfig
from .sweeps import run_ber_sweep, run_range_sweep

BG = "#111827"
PANEL = "#1f2937"
SURFACE = "#f8fafc"
TEXT = "#f8fafc"
INK = "#111827"
MUTED = "#cbd5e1"
ACCENT = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#f59e0b"


Hit = Tuple[float, float, float, float, str]


class DroneVtxApp(tk.Tk):
    """Canvas-rendered UI to avoid macOS/Tk theme invisibility issues."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Drone VTX PHY Simulator")
        self.geometry("1260x820")
        self.minsize(1050, 700)
        self.configure(bg=BG)

        self.channel_models = ["AWGN", "Flat Rayleigh + Doppler", "Rician + Doppler", "TDL Multipath + Doppler"]
        self.camera_presets = list(CAMERA_PRESETS_MBPS)
        self.tdl_profiles = ["TDL-A", "TDL-B", "TDL-C", "Custom TDL"]
        self.mcs_modes = ["Adaptive", "Fixed"]

        self.channel_model = "AWGN"
        self.speed_kmph = 0.0
        self.target_rate_mbps = 25.0
        self.camera_preset = "1080p60"
        self.tdl_profile = "TDL-A"
        self.rms_delay_ns = rms_delay_spread_ns(TDL_PRESETS["TDL-A"])
        self.rician_k_db = 6.0
        self.tx_power_dbm = 27.0
        self.tx_gain_dbi = 2.0
        self.rx_gain_dbi = 5.0
        self.noise_figure_db = 6.0
        self.mcs_mode = "Adaptive"
        self.fixed_mcs_name = DEFAULT_MCS_TABLE[0].name
        self.snr_min_db = 0.0
        self.snr_max_db = 30.0
        self.snr_step_db = 2.0
        self.distance_min_m = 100.0
        self.distance_max_m = 5000.0
        self.distance_step_m = 250.0
        self.payload_bytes = 1200
        self.seed = 1

        self.active_tab = "ber"
        self.status = "Ready"
        self.progress = 0.0
        self.hits: List[Hit] = []
        self.last_results: Dict[str, pd.DataFrame] = {}
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.bind("<Configure>", lambda _: self._draw())
        self.after(100, self._poll_results)
        self._draw()

    def _config(self) -> SimulationConfig:
        return SimulationConfig(
            radio=RadioConfig(
                tx_power_dbm=self.tx_power_dbm,
                tx_antenna_gain_dbi=self.tx_gain_dbi,
                rx_antenna_gain_dbi=self.rx_gain_dbi,
                noise_figure_db=self.noise_figure_db,
            ),
            phy=PhyConfig(),
            sweep=SweepConfig(
                channel_model=self.channel_model,
                speed_kmph=self.speed_kmph,
                tdl_profile=self.tdl_profile,
                rms_delay_ns=self.rms_delay_ns,
                rician_k_db=self.rician_k_db,
                target_rate_mbps=self.target_rate_mbps,
                mcs_mode=self.mcs_mode,
                fixed_mcs_name=self.fixed_mcs_name,
                payload_bytes=self.payload_bytes,
                seed=self.seed,
                snr_min_db=self.snr_min_db,
                snr_max_db=self.snr_max_db,
                snr_step_db=self.snr_step_db,
                distance_min_m=self.distance_min_m,
                distance_max_m=self.distance_max_m,
                distance_step_m=self.distance_step_m,
            ),
        )

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        self.hits.clear()
        w = max(1000, c.winfo_width())
        h = max(700, c.winfo_height())
        panel_w = 340
        c.create_rectangle(0, 0, w, h, fill=BG, outline="")
        c.create_rectangle(0, 0, panel_w, h, fill=PANEL, outline="")

        self._text(24, 24, "Drone VTX PHY", size=20, weight="bold")
        self._text(24, 52, "5.8 GHz / 20 MHz / 30 kHz SCS", color=MUTED)
        y = 92
        y = self._choice(y, "Channel", self.channel_model, "channel")
        y = self._slider(y, "Drone speed", self.speed_kmph, 0, 100, "speed", "kmph")
        y = self._choice(y, "Camera preset", self.camera_preset, "camera")
        y = self._slider(y, "Target data rate", self.target_rate_mbps, 1, 80, "rate", "Mbps")
        y = self._choice(y, "TDL profile", self.tdl_profile, "tdl")
        y = self._slider(y, "RMS delay spread", self.rms_delay_ns, 10, 700, "delay", "ns")
        y = self._slider(y, "Rician K-factor", self.rician_k_db, 0, 20, "kfactor", "dB")
        y = self._choice(y, "MCS mode", self.mcs_mode, "mcs_mode")

        doppler = max_doppler_hz(self.speed_kmph, 5.8e9)
        eirp = eirp_dbm(RadioConfig(tx_power_dbm=self.tx_power_dbm, tx_antenna_gain_dbi=self.tx_gain_dbi))
        self._text(24, y + 6, f"Max Doppler: {doppler:.1f} Hz", color=MUTED)
        self._text(24, y + 30, f"EIRP: {eirp:.1f} dBm   RX gain: {self.rx_gain_dbi:.1f} dBi", color=MUTED)
        y += 70
        self._button(24, y, 130, 38, "Run Sweep", "run", ACCENT)
        self._button(166, y, 130, 38, "Cancel", "cancel", "#475569")
        y += 52
        self._button(24, y, 130, 34, "Export CSV", "export", "#334155")
        self._button(166, y, 130, 34, "Reset", "reset", "#334155")

        main_x = panel_w + 24
        self._tab(main_x, 24, 132, "BER/PER Curves", "ber")
        self._tab(main_x + 144, 24, 76, "Range", "range")

        self._draw_workspace(main_x, 76, w - main_x - 24, h - 132)
        self._draw_status(main_x, h - 42, w - main_x - 24)

    def _draw_workspace(self, x: int, y: int, w: int, h: int) -> None:
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, fill="#0b1220", outline="#334155")
        key = self.active_tab
        df = self.last_results.get(key)
        if df is None or df.empty:
            title = "BER/PER curves" if key == "ber" else "Range"
            self._text(x + 24, y + 24, title, size=18, weight="bold")
            self._text(x + 24, y + 54, "Click Run Sweep to generate the plot.", color=MUTED)
            self._draw_empty_axes(x + 58, y + 102, w - 92, h - 190)
            self._draw_placeholder_table(x + 24, y + h - 92, w - 48)
            return
        if key == "ber":
            xs = df["snr_db"].tolist()
            series = [
                ("BER", xs, df["post_fec_ber"].tolist(), "#60a5fa"),
                ("PER", xs, df["per"].tolist(), "#f87171"),
                ("Throughput", xs, df["usable_throughput_mbps"].tolist(), "#4ade80"),
            ]
            self._plot(x + 58, y + 56, w - 92, h - 190, "SNR (dB)", series, self.target_rate_mbps)
            cols = ["snr_db", "selected_mcs", "post_fec_ber", "per", "usable_throughput_mbps", "meets_rate"]
        else:
            xs = df["distance_m"].tolist()
            series = [
                ("SNR", xs, df["snr_db"].tolist(), "#60a5fa"),
                ("PER", xs, df["per"].tolist(), "#f87171"),
                ("Throughput", xs, df["usable_throughput_mbps"].tolist(), "#4ade80"),
            ]
            self._plot(x + 58, y + 56, w - 92, h - 190, "Distance (m)", series, self.target_rate_mbps)
            cols = ["distance_m", "snr_db", "selected_mcs", "per", "usable_throughput_mbps", "outage"]
        self._draw_table(x + 24, y + h - 112, w - 48, df, cols)

    def _draw_empty_axes(self, x: int, y: int, w: int, h: int) -> None:
        self.canvas.create_line(x, y, x, y + h, fill="#64748b")
        self.canvas.create_line(x, y + h, x + w, y + h, fill="#64748b")
        for i in range(5):
            yy = y + i * h / 4
            xx = x + i * w / 4
            self.canvas.create_line(x, yy, x + w, yy, fill="#1e293b")
            self.canvas.create_line(xx, y, xx, y + h, fill="#1e293b")

    def _plot(self, x: int, y: int, w: int, h: int, xlabel: str, series: list, reference_y: float) -> None:
        self._draw_empty_axes(x, y, w, h)
        all_x = [v for _, xs, _, _ in series for v in xs]
        all_y = [v for _, _, ys, _ in series for v in ys if math.isfinite(v)]
        all_y.append(reference_y)
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if min_y == max_y:
            max_y += 1
        if min_x == max_x:
            max_x += 1

        def sx(v: float) -> float:
            return x + (v - min_x) / (max_x - min_x) * w

        def sy(v: float) -> float:
            return y + (max_y - v) / (max_y - min_y) * h

        ref = sy(reference_y)
        self.canvas.create_line(x, ref, x + w, ref, fill=YELLOW, dash=(5, 4))
        self._text(x + w - 80, ref - 18, "target rate", color=YELLOW)
        for i, (name, xs, ys, color) in enumerate(series):
            pts: list[float] = []
            for xv, yv in zip(xs, ys):
                pts.extend([sx(xv), sy(yv)])
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=color, width=3, smooth=True)
            self._text(x + 10 + i * 140, y - 28, name, color=color, weight="bold")
        self._text(x + w / 2 - 40, y + h + 28, xlabel, color=MUTED)

    def _draw_table(self, x: int, y: int, w: int, df: pd.DataFrame, cols: list[str]) -> None:
        self.canvas.create_rectangle(x, y, x + w, y + 92, fill="#172033", outline="#334155")
        header = " | ".join(cols)
        self._text(x + 10, y + 10, header, color="#e2e8f0", size=10, weight="bold")
        for idx, (_, row) in enumerate(df.head(3).iterrows()):
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
            self._text(x + 10, y + 34 + idx * 18, " | ".join(vals), color=MUTED, size=10)

    def _draw_placeholder_table(self, x: int, y: int, w: int) -> None:
        self.canvas.create_rectangle(x, y, x + w, y + 72, fill="#172033", outline="#334155")
        self._text(x + 10, y + 22, "Summary table will appear here after a sweep.", color=MUTED)

    def _draw_status(self, x: int, y: int, w: int) -> None:
        self._text(x, y - 2, self.status, color=MUTED)
        bar_x = x + 260
        self.canvas.create_rectangle(bar_x, y, x + w, y + 16, fill="#334155", outline="")
        self.canvas.create_rectangle(bar_x, y, bar_x + (w - 260) * self.progress / 100, y + 16, fill=ACCENT, outline="")

    def _text(self, x: float, y: float, text: str, color: str = TEXT, size: int = 11, weight: str = "normal") -> None:
        self.canvas.create_text(x, y, text=text, anchor="nw", fill=color, font=("Helvetica", size, weight))

    def _button(self, x: int, y: int, w: int, h: int, label: str, action: str, color: str) -> None:
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=color, outline="")
        self.canvas.create_text(x + w / 2, y + h / 2, text=label, fill=TEXT, font=("Helvetica", 11, "bold"))
        self.hits.append((x, y, x + w, y + h, action))

    def _tab(self, x: int, y: int, w: int, label: str, tab: str) -> None:
        active = self.active_tab == tab
        self._button(x, y, w, 34, label, f"tab:{tab}", ACCENT if active else "#475569")

    def _choice(self, y: int, label: str, value: str, action: str) -> int:
        self._text(24, y, label, color=MUTED)
        self.canvas.create_rectangle(24, y + 22, 300, y + 52, fill=SURFACE, outline="")
        self.canvas.create_text(36, y + 37, text=value, anchor="w", fill=INK, font=("Helvetica", 11, "bold"))
        self.canvas.create_text(282, y + 37, text="cycle", anchor="e", fill="#475569", font=("Helvetica", 9))
        self.hits.append((24, y + 22, 300, y + 52, action))
        return y + 70

    def _slider(self, y: int, label: str, value: float, min_v: float, max_v: float, action: str, unit: str) -> int:
        self._text(24, y, f"{label}: {value:.1f} {unit}", color=MUTED)
        x0, x1, yy = 24, 300, y + 36
        self.canvas.create_line(x0, yy, x1, yy, fill="#94a3b8", width=4)
        pos = x0 + (value - min_v) / (max_v - min_v) * (x1 - x0)
        self.canvas.create_oval(pos - 8, yy - 8, pos + 8, yy + 8, fill=ACCENT, outline="")
        self.hits.append((x0, yy - 18, x1, yy + 18, f"slider:{action}:{min_v}:{max_v}"))
        return y + 64

    def _click(self, event: tk.Event) -> None:
        for x0, y0, x1, y1, action in reversed(self.hits):
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._handle(action, event.x, x0, x1)
                return

    def _drag(self, event: tk.Event) -> None:
        self._click(event)

    def _handle(self, action: str, x: float = 0, x0: float = 0, x1: float = 1) -> None:
        if action.startswith("tab:"):
            self.active_tab = action.split(":", 1)[1]
        elif action.startswith("slider:"):
            _, name, min_s, max_s = action.split(":")
            val = float(min_s) + max(0, min(1, (x - x0) / (x1 - x0))) * (float(max_s) - float(min_s))
            if name == "speed":
                self.speed_kmph = round(val)
            elif name == "rate":
                self.target_rate_mbps = round(val)
                self.camera_preset = "Custom"
            elif name == "delay":
                self.rms_delay_ns = round(val)
            elif name == "kfactor":
                self.rician_k_db = round(val, 1)
        elif action == "channel":
            self.channel_model = self._next(self.channel_models, self.channel_model)
        elif action == "camera":
            self.camera_preset = self._next(self.camera_presets, self.camera_preset)
            if self.camera_preset != "Custom":
                self.target_rate_mbps = CAMERA_PRESETS_MBPS[self.camera_preset]
        elif action == "tdl":
            self.tdl_profile = self._next(self.tdl_profiles, self.tdl_profile)
            if self.tdl_profile in TDL_PRESETS:
                self.rms_delay_ns = rms_delay_spread_ns(TDL_PRESETS[self.tdl_profile])
        elif action == "mcs_mode":
            self.mcs_mode = self._next(self.mcs_modes, self.mcs_mode)
        elif action == "run":
            self._run_active_sweep()
        elif action == "cancel":
            self.cancel_event.set()
            self.status = "Cancel requested"
        elif action == "export":
            self._export_csv()
        elif action == "reset":
            self._reset_defaults()
        self._draw()

    def _next(self, values: list[str], current: str) -> str:
        return values[(values.index(current) + 1) % len(values)]

    def _run_active_sweep(self) -> None:
        if self.worker and self.worker.is_alive():
            self.status = "A sweep is already running"
            return
        self.cancel_event.clear()
        cfg = self._config()
        key = self.active_tab
        runner: Callable[..., pd.DataFrame] = run_ber_sweep if key == "ber" else run_range_sweep
        started = time.time()

        def progress(done: int, total: int, label: str) -> None:
            self.result_queue.put(("progress", (done, total, label, started)))

        def worker() -> None:
            try:
                df = runner(cfg, progress=progress, cancel=self.cancel_event.is_set)
                self.result_queue.put(("done", (key, df)))
            except Exception as exc:
                self.result_queue.put(("error", str(exc)))

        self.progress = 0
        self.status = "Running sweep"
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _poll_results(self) -> None:
        changed = False
        while True:
            try:
                kind, payload = self.result_queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "progress":
                done, total, label, started = payload
                self.progress = done / max(1, total) * 100
                self.status = f"{label} ({done}/{total}) elapsed {time.time() - started:.1f}s"
            elif kind == "done":
                key, df = payload
                self.last_results[key] = df
                self.progress = 100
                self.status = f"Completed {key} sweep with {len(df)} rows"
            elif kind == "error":
                self.status = f"Error: {payload}"
        if changed:
            self._draw()
        self.after(100, self._poll_results)

    def _export_csv(self) -> None:
        df = self.last_results.get(self.active_tab)
        if df is None or df.empty:
            self.status = "No results to export"
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            self.status = f"Exported {path}"

    def _reset_defaults(self) -> None:
        self.channel_model = "AWGN"
        self.speed_kmph = 0.0
        self.target_rate_mbps = 25.0
        self.camera_preset = "1080p60"
        self.tdl_profile = "TDL-A"
        self.rms_delay_ns = rms_delay_spread_ns(TDL_PRESETS["TDL-A"])
        self.rician_k_db = 6.0
        self.mcs_mode = "Adaptive"
        self.status = "Defaults restored"
        self.progress = 0


def main() -> None:
    app = DroneVtxApp()
    app.mainloop()

