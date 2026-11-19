import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("run_directory", type=Path)
directory = parser.parse_args().run_directory.resolve()
with (directory / "column-sweep.csv").open() as source:
    results = list(csv.DictReader(source))

colors = {"exact": "#0072B2", "padded": "#D55E00"}
fig, grid = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, sharey=True, layout="constrained")
for i, rows in enumerate(("1024", "4096")):
    for j, (load, title) in enumerate((("row", "Zeilenloads"), ("gather", "LSC-Gathers"))):
        ax = grid[i, j]
        ax.set_title(f"{int(rows):,} Zeilen · {title}".replace(",", "."))
        ax.axhline(0, color="#444444", linewidth=1)
        for boundary in (16, 32, 48):
            ax.axvline(boundary, color="#dbdbdb", linewidth=0.8)
        for record in results:
            candidate = record["distinct_alternative"]
            if record["load_path"] == load and record["weight_rows"] == rows and candidate in colors:
                y = float(record["alternative_paired_median_latency_change_percent"])
                wins = int(record["alternative_faster_rounds"])
                strong = (wins >= 6 and y <= -5) or (wins <= 1 and y >= 5)
                ax.scatter(int(record["columns"]), y, color=colors[candidate],
                           alpha=0.95 if strong else 0.40, s=24 if strong else 13, linewidth=0)
        ax.set_xlim(0, 65)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        ax.grid(axis="y", alpha=0.18)
fig.supxlabel("Angeforderte Spalten")
fig.supylabel("Latenzänderung gegenüber aktueller Regel (%)")
fig.suptitle("Q8_0 auf Arrow Lake-H · K = 4.096")
grid[0, 1].legend(handles=[
    Line2D([], [], linestyle="none", marker="o", color=colors["exact"], label="Exakte Restkacheln"),
    Line2D([], [], linestyle="none", marker="o", color=colors["padded"], label="16er-Rundung"),
], loc="upper left", frameon=False, fontsize=9)
fig.savefig(directory / "column-sweep.svg")
