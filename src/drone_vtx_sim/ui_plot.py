from __future__ import annotations

import math
import tkinter as tk
from typing import Iterable, Sequence


class PlotCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, background="#101418", highlightthickness=0, **kwargs)

    def plot_lines(
        self,
        series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
        xlabel: str,
        ylabel: str,
        log_y: bool = False,
        reference_y: float | None = None,
    ) -> None:
        self.delete("all")
        width = max(400, self.winfo_width())
        height = max(260, self.winfo_height())
        left, right, top, bottom = 64, 22, 24, 48
        plot_w = width - left - right
        plot_h = height - top - bottom
        self.create_rectangle(0, 0, width, height, fill="#101418", outline="")
        self.create_text(18, 18, text="Simulation plot", fill="#d9e2ec", anchor="w", font=("TkDefaultFont", 12, "bold"))
        all_x = [x for _, xs, _, _ in series for x in xs]
        all_y = [y for _, _, ys, _ in series for y in ys if math.isfinite(y)]
        if reference_y is not None:
            all_y.append(reference_y)
        if not all_x or not all_y:
            self.create_text(width / 2, height / 2, text="Run a sweep to plot results", fill="#b8c1cc", font=("TkDefaultFont", 14))
            return
        if log_y:
            all_y = [math.log10(max(y, 1e-9)) for y in all_y]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if min_x == max_x:
            max_x += 1
        if min_y == max_y:
            max_y += 1

        def sx(x: float) -> float:
            return left + (x - min_x) / (max_x - min_x) * plot_w

        def sy(y: float) -> float:
            val = math.log10(max(y, 1e-9)) if log_y else y
            return top + (max_y - val) / (max_y - min_y) * plot_h

        self.create_line(left, top, left, top + plot_h, fill="#53606c")
        self.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#53606c")
        for i in range(5):
            x = left + i * plot_w / 4
            y = top + i * plot_h / 4
            self.create_line(x, top, x, top + plot_h, fill="#1e2730")
            self.create_line(left, y, left + plot_w, y, fill="#1e2730")
        self.create_text(width / 2, height - 15, text=xlabel, fill="#b8c1cc")
        self.create_text(18, height / 2, text=ylabel, fill="#b8c1cc", angle=90)

        if reference_y is not None:
            y = sy(reference_y)
            self.create_line(left, y, left + plot_w, y, fill="#f4c542", dash=(4, 4))
            self.create_text(left + plot_w - 6, y - 10, text="target", fill="#f4c542", anchor="e")

        legend_x = left + 8
        for idx, (name, xs, ys, color) in enumerate(series):
            points = []
            for x, y in zip(xs, ys):
                if math.isfinite(y):
                    points.extend([sx(x), sy(y)])
            if len(points) >= 4:
                self.create_line(*points, fill=color, width=2, smooth=True)
            self.create_text(legend_x, top + 12 + 16 * idx, text=name, fill=color, anchor="w")
