import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("run_directory", type=Path)
root = parser.parse_args().run_directory.resolve()
formats = ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0")
loads = ("row", "gather")
operations = ("swiglu", "geglu")
groups = defaultdict(dict)
samples = []
for load in loads:
    for quant in formats:
        path = root / f"{load}-{quant}-4096-k4096-1-8.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = groups[load, quant, row["glu"], row["columns"], row["tile_width"]]
            if row["round"] in group or not row["passed"] or row["iterations"] != 32 or row["rows"] != 4096:
                raise ValueError(f"Unexpected sample in {path}")
            group[row["round"]] = row["microseconds"]
            samples.append({"load_path": load, **row})

table = []
for load in loads:
    for quant in formats:
        for operation in operations:
            for columns in range(1, 9):
                widths = range(columns, 9) if quant == "q8_0" else sorted({columns, 8})
                exact = groups[load, quant, operation, columns, columns]
                for tile in widths:
                    times = groups[load, quant, operation, columns, tile]
                    if set(exact) != set(range(8)) or set(times) != set(range(8)):
                        raise ValueError(f"Incomplete cell {load} {quant} {operation} {columns} {tile}")
                    relatives = [100 * (times[i] / exact[i] - 1) for i in range(8)]
                    table.append({
                        "load_path": load, "format": quant, "glu": operation, "weight_rows": 4096, "k": 4096,
                        "requested_columns": columns, "tile_width": tile,
                        "median_us": round(statistics.median(times.values()), 3),
                        "exact_median_us": round(statistics.median(exact.values()), 3),
                        "paired_median_delta_percent": round(statistics.median(relatives), 2),
                        "faster_than_exact_rounds": sum(d < 0 for d in relatives),
                        "slower_than_exact_rounds": sum(d > 0 for d in relatives),
                    })

for name, content in (("small-glu-tiles.csv", table), ("small-glu-tiles-samples.csv", samples)):
    with (root / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(content[0]))
        writer.writeheader()
        writer.writerows(content)

print(f"{len(table)} variant cells, {len(samples)} CPU-reference checks")
for quant in formats:
    for load in loads:
        selected = [r for r in table if r["format"] == quant and r["load_path"] == load
                    and r["tile_width"] == 8 and r["requested_columns"] < 8]
        clear_rounded = [r for r in selected if r["faster_than_exact_rounds"] >= 6
                         and r["paired_median_delta_percent"] <= -5]
        clear_exact = [r for r in selected if r["slower_than_exact_rounds"] >= 6
                       and r["paired_median_delta_percent"] >= 5]
        print(quant, load, f"pad8 wins {len(clear_rounded)}/14, exact wins {len(clear_exact)}/14",
              "median delta", round(statistics.median(r["paired_median_delta_percent"] for r in selected), 1))

for load in loads:
    for op in operations:
        details = []
        for columns in range(1, 9):
            options = [r for r in table if r["load_path"] == load and r["format"] == "q8_0"
                       and r["glu"] == op and r["requested_columns"] == columns]
            best = min(options, key=lambda r: r["median_us"])
            details.append((columns, best["tile_width"], best["median_us"], best["faster_than_exact_rounds"]))
        print("q8_0", load, op, "best median", details)
