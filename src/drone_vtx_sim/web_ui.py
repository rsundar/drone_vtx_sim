from __future__ import annotations

import argparse
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .channels import TDL_PRESETS, max_doppler_hz, rms_delay_spread_ns
from .link_budget import eirp_dbm
from .models import CAMERA_PRESETS_MBPS, DEFAULT_MCS_TABLE, PhyConfig, RadioConfig, SimulationConfig, SweepConfig
from .sweeps import run_ber_sweep, run_range_sweep


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drone VTX PHY Simulator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0f172a;
      --panel: #172033;
      --panel2: #1f2a44;
      --text: #f8fafc;
      --muted: #cbd5e1;
      --line: #334155;
      --accent: #3b82f6;
      --green: #22c55e;
      --red: #ef4444;
      --yellow: #f59e0b;
      --white: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app { display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 100vh; }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
      max-height: 100vh;
    }
    main { padding: 20px; min-width: 0; }
    h1 { margin: 0 0 2px; font-size: 21px; }
    .sub { color: var(--muted); margin-bottom: 18px; }
    label { display: block; color: var(--muted); margin: 12px 0 5px; }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: #0f172a;
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
    }
    input[type="range"] { padding: 0; accent-color: var(--accent); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .metric {
      margin-top: 12px;
      padding: 10px;
      background: var(--panel2);
      border-radius: 6px;
      color: var(--muted);
    }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border: 0;
      font-weight: 700;
    }
    button.secondary { background: #475569; }
    .tabs { display: flex; gap: 10px; margin-bottom: 14px; }
    .tab {
      width: auto;
      padding: 9px 14px;
      background: #475569;
    }
    .tab.active { background: var(--accent); }
    .card {
      background: #101827;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
    }
    .inputs { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .inputs label { margin: 0; width: 130px; }
    .plot {
      width: 100%;
      height: 520px;
      display: block;
      background: #0b1220;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 14px;
      color: var(--muted);
      margin: 10px 0 16px;
    }
    .bar { flex: 1; height: 12px; background: #334155; border-radius: 99px; overflow: hidden; }
    .fill { width: 0%; height: 100%; background: var(--accent); transition: width 120ms linear; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #101827;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 12px;
    }
    th, td { padding: 8px; border-bottom: 1px solid #223047; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); background: #172033; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Drone VTX PHY</h1>
      <div class="sub">5.8 GHz / 20 MHz / 30 kHz SCS</div>

      <label>Channel model</label>
      <select id="channel_model">
        <option>AWGN</option>
        <option>Flat Rayleigh + Doppler</option>
        <option>Rician + Doppler</option>
        <option>TDL Multipath + Doppler</option>
      </select>

      <label>Drone speed: <span id="speed_label">0</span> kmph</label>
      <input id="speed_kmph" type="range" min="0" max="100" value="0" />

      <label>Camera preset</label>
      <select id="camera_preset"></select>

      <label>Target data rate: <span id="rate_label">25</span> Mbps</label>
      <input id="target_rate_mbps" type="range" min="1" max="80" value="25" />

      <label>TDL profile</label>
      <select id="tdl_profile">
        <option>TDL-A</option><option>TDL-B</option><option>TDL-C</option><option>Custom TDL</option>
      </select>

      <div class="row">
        <div>
          <label>RMS delay ns</label>
          <input id="rms_delay_ns" type="number" value="100" />
        </div>
        <div>
          <label>Rician K dB</label>
          <input id="rician_k_db" type="number" value="6" />
        </div>
      </div>

      <div class="row">
        <div><label>TX dBm</label><input id="tx_power_dbm" type="number" value="27" /></div>
        <div><label>TX dBi</label><input id="tx_gain_dbi" type="number" value="2" /></div>
        <div><label>RX dBi</label><input id="rx_gain_dbi" type="number" value="5" /></div>
        <div><label>NF dB</label><input id="noise_figure_db" type="number" value="6" /></div>
      </div>

      <label>MCS mode</label>
      <select id="mcs_mode"><option>Adaptive</option><option>Fixed</option></select>
      <label>Adaptive MCS policy</label>
      <select id="mcs_policy"><option>Meet target rate</option><option>Max throughput</option></select>
      <label>Fixed MCS</label>
      <select id="fixed_mcs"></select>
      <div class="metric" id="mcs_help">
        Adaptive mode chooses the lowest reliable MCS that meets the target rate. Change Fixed MCS to force a modulation/code-rate.
      </div>

      <div class="row">
        <div><label>Payload bytes</label><input id="payload_bytes" type="number" value="1200" /></div>
        <div><label>Random seed</label><input id="seed" type="number" value="1" /></div>
      </div>

      <div class="metric">
        <div id="doppler">Max Doppler: 0 Hz</div>
        <div id="eirp">EIRP: 29 dBm</div>
      </div>

      <div class="actions">
        <button id="run">Run Sweep</button>
        <button id="export" class="secondary">Export CSV</button>
      </div>
    </aside>

    <main>
      <div class="tabs">
        <button id="tab_ber" class="tab active">BER/PER Curves</button>
        <button id="tab_range" class="tab">Range</button>
      </div>
      <div class="status"><span id="status">Ready</span><div class="bar"><div id="fill" class="fill"></div></div></div>

      <section id="ber_panel">
        <div class="card">
          <div class="inputs">
            <label>SNR min<input id="snr_min_db" type="number" value="0" /></label>
            <label>SNR max<input id="snr_max_db" type="number" value="30" /></label>
            <label>SNR step<input id="snr_step_db" type="number" value="2" /></label>
          </div>
          <svg id="ber_plot" class="plot"></svg>
        </div>
        <div id="ber_table"></div>
      </section>

      <section id="range_panel" class="hidden">
        <div class="card">
          <div class="inputs">
            <label>Min distance m<input id="distance_min_m" type="number" value="100" /></label>
            <label>Max distance m<input id="distance_max_m" type="number" value="5000" /></label>
            <label>Step m<input id="distance_step_m" type="number" value="250" /></label>
          </div>
          <svg id="range_plot" class="plot"></svg>
        </div>
        <div id="range_table"></div>
      </section>
    </main>
  </div>
  <script>
    const presets = __CAMERA_PRESETS__;
    const mcsNames = __MCS_NAMES__;
    const tdlDelay = __TDL_DELAY__;
    let activeTab = "ber";
    let lastRows = {ber: [], range: []};

    function $(id) { return document.getElementById(id); }
    function num(id) { return Number($(id).value); }

    function init() {
      for (const [name, rate] of Object.entries(presets)) {
        const o = document.createElement("option"); o.textContent = name; $("camera_preset").appendChild(o);
      }
      $("camera_preset").value = "1080p60";
      for (const name of mcsNames) {
        const o = document.createElement("option"); o.textContent = name; $("fixed_mcs").appendChild(o);
      }
      $("speed_kmph").oninput = updateDerived;
      $("target_rate_mbps").oninput = () => { $("camera_preset").value = "Custom"; updateDerived(); };
      $("camera_preset").onchange = () => {
        const p = $("camera_preset").value;
        if (p !== "Custom") $("target_rate_mbps").value = presets[p];
        updateDerived();
      };
      $("tdl_profile").onchange = () => {
        const p = $("tdl_profile").value;
        if (tdlDelay[p]) $("rms_delay_ns").value = tdlDelay[p].toFixed(1);
      };
      $("fixed_mcs").onchange = () => {
        $("mcs_mode").value = "Fixed";
        updateMcsHelp();
      };
      $("mcs_mode").onchange = updateMcsHelp;
      $("mcs_policy").onchange = updateMcsHelp;
      $("tab_ber").onclick = () => setTab("ber");
      $("tab_range").onclick = () => setTab("range");
      $("run").onclick = runSweep;
      $("export").onclick = exportCsv;
      updateDerived();
      updateMcsHelp();
      drawEmpty("ber_plot", "Run a sweep to display BER/PER curves");
      drawEmpty("range_plot", "Run a sweep to display range results");
    }

    function setTab(tab) {
      activeTab = tab;
      $("ber_panel").classList.toggle("hidden", tab !== "ber");
      $("range_panel").classList.toggle("hidden", tab !== "range");
      $("tab_ber").classList.toggle("active", tab === "ber");
      $("tab_range").classList.toggle("active", tab === "range");
    }

    function updateDerived() {
      const speed = num("speed_kmph");
      const rate = num("target_rate_mbps");
      $("speed_label").textContent = speed.toFixed(0);
      $("rate_label").textContent = rate.toFixed(0);
      $("doppler").textContent = `Max Doppler: ${(speed * 1000 / 3600 * 5.8e9 / 299792458).toFixed(1)} Hz`;
      $("eirp").textContent = `EIRP: ${(num("tx_power_dbm") + num("tx_gain_dbi")).toFixed(1)} dBm`;
    }

    function updateMcsHelp() {
      if ($("mcs_mode").value === "Fixed") {
        $("mcs_help").textContent = `Manual MCS is active: ${$("fixed_mcs").value}. The sweep will use this modulation/code-rate at every point.`;
      } else if ($("mcs_policy").value === "Max throughput") {
        $("mcs_help").textContent = "Adaptive mode is active: the simulator will use the highest reliable MCS at each point.";
      } else {
        $("mcs_help").textContent = "Adaptive mode is active: the simulator will use the lowest reliable MCS that meets the target data rate.";
      }
    }

    function payload() {
      return {
        tab: activeTab,
        channel_model: $("channel_model").value,
        speed_kmph: num("speed_kmph"),
        target_rate_mbps: num("target_rate_mbps"),
        tdl_profile: $("tdl_profile").value,
        rms_delay_ns: num("rms_delay_ns"),
        rician_k_db: num("rician_k_db"),
        tx_power_dbm: num("tx_power_dbm"),
        tx_gain_dbi: num("tx_gain_dbi"),
        rx_gain_dbi: num("rx_gain_dbi"),
        noise_figure_db: num("noise_figure_db"),
        mcs_mode: $("mcs_mode").value,
        mcs_policy: $("mcs_policy").value,
        fixed_mcs_name: $("fixed_mcs").value,
        payload_bytes: num("payload_bytes"),
        seed: num("seed"),
        snr_min_db: num("snr_min_db"),
        snr_max_db: num("snr_max_db"),
        snr_step_db: num("snr_step_db"),
        distance_min_m: num("distance_min_m"),
        distance_max_m: num("distance_max_m"),
        distance_step_m: num("distance_step_m")
      };
    }

    async function runSweep() {
      const requestedTab = activeTab;
      $("status").textContent = "Running sweep...";
      $("fill").style.width = "30%";
      const res = await fetch("/api/sweep", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload())});
      const body = await res.json();
      if (!res.ok) {
        $("status").textContent = body.error || "Sweep failed";
        $("fill").style.width = "0%";
        return;
      }
      lastRows[requestedTab] = body.rows;
      const selected = [...new Set(body.rows.map(r => r.selected_mcs))].join(", ");
      $("status").textContent = `Completed ${requestedTab} sweep with ${body.rows.length} rows. MCS: ${selected}`;
      $("fill").style.width = "100%";
      if (requestedTab === "ber") {
        drawBerPlot("ber_plot", body.rows, num("target_rate_mbps"));
        drawTable("ber_table", body.rows, ["snr_db","selected_mcs","post_fec_ber","per","usable_throughput_mbps","throughput_margin_mbps","meets_rate"]);
      } else {
        drawRangePlot("range_plot", body.rows, num("target_rate_mbps"));
        drawTable("range_table", body.rows, ["distance_m","snr_db","selected_mcs","per","usable_throughput_mbps","throughput_margin_mbps","outage"]);
      }
    }

    function drawEmpty(id, text) {
      const svg = $(id);
      svg.innerHTML = `<text x="30" y="40" fill="#f8fafc" font-size="18" font-weight="700">${text}</text>`;
    }

    function drawBerPlot(id, rows, target) {
      drawStackedPlot(id, rows, "snr_db", [
        {
          title: "Error rate vs SNR (log scale)",
          ylabel: "BER / PER",
          log: true,
          specs: [
            ["BER", "post_fec_ber", "#60a5fa"],
            ["PER", "per", "#f87171"]
          ]
        },
        {
          title: "Usable throughput vs SNR",
          ylabel: "Mbps",
          reference: target,
          specs: [["Throughput", "usable_throughput_mbps", "#4ade80"]]
        }
      ], "SNR (dB)");
    }

    function drawRangePlot(id, rows, target) {
      drawStackedPlot(id, rows, "distance_m", [
        {
          title: "Link SNR vs distance",
          ylabel: "dB",
          specs: [["SNR", "snr_db", "#60a5fa"]]
        },
        {
          title: "Usable throughput vs distance",
          ylabel: "Mbps",
          reference: target,
          specs: [["Throughput", "usable_throughput_mbps", "#4ade80"]]
        },
        {
          title: "PER vs distance (log scale)",
          ylabel: "PER",
          log: true,
          specs: [["PER", "per", "#f87171"]]
        }
      ], "Distance (m)");
    }

    function drawStackedPlot(id, rows, xKey, panels, xlabel) {
      const svg = $(id);
      const box = svg.getBoundingClientRect();
      const w = Math.max(800, box.width), h = Math.max(420, box.height);
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      const m = {l: 70, r: 28, t: 34, b: 36};
      const xs = rows.map(r => Number(r[xKey]));
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      let html = `<rect width="${w}" height="${h}" fill="#0b1220"/>`;
      const panelGap = 28;
      const panelH = (h - m.t - m.b - panelGap * (panels.length - 1)) / panels.length;
      const plotW = w - m.l - m.r;
      const sx = x => m.l + (x - minX) / (maxX - minX || 1) * plotW;
      const fmt = v => {
        const av = Math.abs(v);
        if ((av > 0 && av < 0.01) || av >= 10000) return Number(v).toExponential(1);
        if (av >= 100) return Number(v).toFixed(0);
        if (av >= 10) return Number(v).toFixed(1);
        return Number(v).toFixed(2);
      };
      const fmtLog = v => `1e${Math.round(v)}`;

      panels.forEach((panel, panelIdx) => {
        const yTop = m.t + panelIdx * (panelH + panelGap);
        const yBottom = yTop + panelH;
        const values = [];
        for (const [, key] of panel.specs) values.push(...rows.map(r => Number(r[key])).filter(Number.isFinite));
        if (panel.reference !== undefined) values.push(panel.reference);
        const transform = v => panel.log ? Math.log10(Math.max(Number(v), 1e-9)) : Number(v);
        let minY = Math.min(...values.map(transform));
        let maxY = Math.max(...values.map(transform));
        if (minY === maxY) maxY += 1;
        if (panel.log) {
          minY = Math.min(minY, -6);
          maxY = Math.max(maxY, 0);
        }
        const sy = y => yTop + (maxY - transform(y)) / (maxY - minY || 1) * panelH;

        html += `<text x="${m.l}" y="${yTop-13}" fill="#e2e8f0" font-weight="700">${panel.title}</text>`;
        html += `<text x="16" y="${yTop + panelH / 2}" fill="#94a3b8" transform="rotate(-90 16 ${yTop + panelH / 2})">${panel.ylabel}</text>`;
        for (let i=0; i<5; i++) {
          const gx = m.l + i * plotW / 4;
          const gy = yTop + i * panelH / 4;
          const xVal = minX + i * (maxX - minX) / 4;
          const yVal = maxY - i * (maxY - minY) / 4;
          html += `<line x1="${m.l}" y1="${gy}" x2="${w-m.r}" y2="${gy}" stroke="#1e293b"/>`;
          html += `<line x1="${gx}" y1="${yTop}" x2="${gx}" y2="${yBottom}" stroke="#1e293b"/>`;
          html += `<text x="${m.l-8}" y="${gy+4}" fill="#94a3b8" font-size="11" text-anchor="end">${panel.log ? fmtLog(yVal) : fmt(yVal)}</text>`;
          if (panelIdx === panels.length - 1) {
            html += `<text x="${gx}" y="${yBottom+18}" fill="#94a3b8" font-size="11" text-anchor="middle">${fmt(xVal)}</text>`;
          }
        }
        html += `<line x1="${m.l}" y1="${yTop}" x2="${m.l}" y2="${yBottom}" stroke="#64748b"/>`;
        html += `<line x1="${m.l}" y1="${yBottom}" x2="${w-m.r}" y2="${yBottom}" stroke="#64748b"/>`;
        if (panel.reference !== undefined) {
          const ry = sy(panel.reference);
          html += `<line x1="${m.l}" y1="${ry}" x2="${w-m.r}" y2="${ry}" stroke="#f59e0b" stroke-dasharray="6 5"/>`;
          html += `<text x="${w-m.r-92}" y="${ry-8}" fill="#f59e0b">target</text>`;
        }
        panel.specs.forEach(([name,key,color], idx) => {
          const points = rows.map(r => `${sx(Number(r[xKey]))},${sy(Number(r[key]))}`).join(" ");
          html += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
          html += `<text x="${m.l + 250 + idx*110}" y="${yTop-13}" fill="${color}" font-weight="700">${name}</text>`;
        });
      });
      html += `<text x="${w/2-40}" y="${h-10}" fill="#94a3b8">${xlabel}</text>`;
      svg.innerHTML = html;
    }

    function drawTable(id, rows, cols) {
      let html = "<table><thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
      for (const r of rows.slice(0, 80)) {
        html += "<tr>" + cols.map(c => {
          const v = r[c];
          const out = typeof v === "number" ? Number(v).toPrecision(4) : String(v);
          return `<td>${out}</td>`;
        }).join("") + "</tr>";
      }
      $(id).innerHTML = html + "</tbody></table>";
    }

    function exportCsv() {
      const rows = lastRows[activeTab];
      if (!rows.length) { $("status").textContent = "No results to export"; return; }
      const cols = Object.keys(rows[0]);
      const csv = [cols.join(","), ...rows.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(","))].join("\n");
      const blob = new Blob([csv], {type: "text/csv"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `drone_vtx_${activeTab}_sweep.csv`;
      a.click();
    }

    init();
  </script>
</body>
</html>"""


def _jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            clean[key] = value
        rows.append(clean)
    return rows


def _build_config(payload: dict[str, Any]) -> SimulationConfig:
    radio = RadioConfig(
        tx_power_dbm=float(payload.get("tx_power_dbm", 27.0)),
        tx_antenna_gain_dbi=float(payload.get("tx_gain_dbi", 2.0)),
        rx_antenna_gain_dbi=float(payload.get("rx_gain_dbi", 5.0)),
        noise_figure_db=float(payload.get("noise_figure_db", 6.0)),
    )
    sweep = SweepConfig(
        channel_model=str(payload.get("channel_model", "AWGN")),
        speed_kmph=float(payload.get("speed_kmph", 0.0)),
        target_rate_mbps=float(payload.get("target_rate_mbps", 25.0)),
        tdl_profile=str(payload.get("tdl_profile", "TDL-A")),
        rms_delay_ns=float(payload.get("rms_delay_ns", 100.0)),
        rician_k_db=float(payload.get("rician_k_db", 6.0)),
        mcs_mode=str(payload.get("mcs_mode", "Adaptive")),
        mcs_policy=str(payload.get("mcs_policy", "Meet target rate")),
        fixed_mcs_name=str(payload.get("fixed_mcs_name", DEFAULT_MCS_TABLE[0].name)),
        payload_bytes=int(float(payload.get("payload_bytes", 1200))),
        seed=int(float(payload.get("seed", 1))),
        snr_min_db=float(payload.get("snr_min_db", 0.0)),
        snr_max_db=float(payload.get("snr_max_db", 30.0)),
        snr_step_db=float(payload.get("snr_step_db", 2.0)),
        distance_min_m=float(payload.get("distance_min_m", 100.0)),
        distance_max_m=float(payload.get("distance_max_m", 5000.0)),
        distance_step_m=float(payload.get("distance_step_m", 250.0)),
    )
    return SimulationConfig(radio=radio, phy=PhyConfig(), sweep=sweep)


class Handler(BaseHTTPRequestHandler):
    server_version = "DroneVtxSim/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        delays = {name: rms_delay_spread_ns(taps) for name, taps in TDL_PRESETS.items()}
        html = (
            HTML.replace("__CAMERA_PRESETS__", json.dumps(CAMERA_PRESETS_MBPS))
            .replace("__MCS_NAMES__", json.dumps([m.name for m in DEFAULT_MCS_TABLE]))
            .replace("__TDL_DELAY__", json.dumps(delays))
        )
        self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/sweep":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            cfg = _build_config(payload)
            tab = payload.get("tab", "ber")
            df = run_ber_sweep(cfg) if tab == "ber" else run_range_sweep(cfg)
            body = json.dumps({"rows": _jsonable(df)}).encode("utf-8")
            self._send(200, "application/json", body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(500, "application/json", body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web-ui] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Drone VTX local web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    port = _free_port(args.port)
    server = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}/"
    print(f"Drone VTX PHY Simulator UI: {url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server")
    finally:
        server.server_close()
