import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run_directory", type=Path)
root = parser.parse_args().run_directory.resolve()
with (root / "small-glu-tiles.csv").open() as source:
    rows = list(csv.DictReader(source))
formats = ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0")
data = {(r["format"], r["glu"], r["load_path"], int(r["requested_columns"])): r for r in rows
        if r["tile_width"] == "8" and int(r["requested_columns"]) < 8}
figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True, sharey=True)
for y, (op, name) in enumerate((("swiglu", "SwiGLU"), ("geglu", "GeGLU"))):
    for x, (load, name_load) in enumerate((("row", "Zeilenloads"), ("gather", "Gathers"))):
        axis = axes[y, x]
        values = np.array([[float(data[quant, op, load, col]["paired_median_delta_percent"])
                            for col in range(1, 8)] for quant in formats])
        image = axis.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40), aspect="auto")
        axis.set_title(f"{name} · {name_load}")
        axis.set_xticks(range(7), range(1, 8))
        axis.set_yticks(range(6), [name.upper() for name in formats])
        for q, quant in enumerate(formats):
            for i, col in enumerate(range(1, 8)):
                sample = data[quant, op, load, col]
                difference = values[q, i]
                strong = abs(difference) >= 5 and max(int(sample["faster_than_exact_rounds"]),
                                                     int(sample["slower_than_exact_rounds"])) >= 6
                axis.text(i, q, f"{difference:+.0f}%", ha="center", va="center",
                          color="white" if abs(difference) >= 24 else "#242424",
                          fontweight="bold" if strong else "normal", fontsize=9)
figure.supxlabel("Angeforderte Spalten")
figure.suptitle("GLU auf Arrow Lake-H · 8er-Kachel gegenüber exakter Kachel\n4.096 Gewichtszeilen · K = 4.096")
figure.colorbar(image, ax=axes, shrink=0.9, extend="both", label="Latenzänderung in % (negativ ist schneller)")
figure.savefig(root / "small-glu-tiles.svg")
