import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("run_directory", type=Path)
root = parser.parse_args().run_directory.resolve()
loads = ("row", "gather")
row_counts = (1024, 4096)
ranges = ((1, 16), (17, 32), (33, 48), (49, 64))
matrix = []
samples = []
for load in loads:
    for rows in row_counts:
        groups = defaultdict(lambda: defaultdict(dict))
        for first, last in ranges:
            path = root / f"{load}-{rows}-k4096-{first}-{last}.jsonl"
            for line in path.read_text().splitlines():
                value = json.loads(line)
                columns, schedule, round_id = value["columns"], value["schedule"], value["round"]
                slots = groups[columns][schedule]
                if round_id in slots or not value["passed"] or value["iterations"] != 32:
                    raise ValueError(f"Invalid sample in {path}")
                slots[round_id] = value["microseconds"]
                samples.append({"load_path": load, **value})
        if set(groups) != set(range(1, 65)):
            raise ValueError(f"Missing widths for {load}, rows={rows}")
        for columns, schedules in sorted(groups.items()):
            full, remainder = divmod(columns, 16)
            alt = None
            if 8 < columns and remainder:
                alt = "exact" if remainder > 8 else "padded"
            if set(schedules) != ({"current", alt} if alt else {"current"}):
                raise ValueError(f"Wrong schedules for {load}, {rows}, {columns}")
            if any(set(t) != set(range(7)) for t in schedules.values()):
                raise ValueError(f"Wrong round count for {load}, {rows}, {columns}")
            exact_tiles = full + (remainder + 7) // 8 if columns > 8 else 1
            padded_tiles = full + bool(remainder)
            current_computed = full * 16 + (16 if remainder > 8 else remainder)
            padded_computed = padded_tiles * 16 if columns > 8 else columns
            current = schedules["current"]
            exact = schedules.get("exact", current)
            padded = schedules.get("padded", current)
            row = {
                "load_path": load, "weight_rows": rows, "k": 4096, "columns": columns,
                "current_groups_per_row_tile": padded_tiles,
                "exact_groups_per_row_tile": exact_tiles,
                "padded_groups_per_row_tile": padded_tiles,
                "current_computed_columns": current_computed,
                "exact_computed_columns": columns,
                "padded_computed_columns": padded_computed,
                "current_median_us": round(statistics.median(current.values()), 3),
                "exact_median_us": round(statistics.median(exact.values()), 3),
                "padded_median_us": round(statistics.median(padded.values()), 3),
                "distinct_alternative": alt or "none",
                "alternative_faster_rounds": "",
                "current_faster_rounds": "",
                "alternative_paired_median_latency_change_percent": "",
            }
            if alt:
                ratios = [schedules[alt][i] / current[i] for i in range(7)]
                row["alternative_faster_rounds"] = sum(ratio < 1 for ratio in ratios)
                row["current_faster_rounds"] = sum(ratio > 1 for ratio in ratios)
                row["alternative_paired_median_latency_change_percent"] = round(
                    100 * (statistics.median(ratios) - 1), 2)
            matrix.append(row)

for name, data in (("column-sweep.csv", matrix), ("column-sweep-samples.csv", samples)):
    with (root / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

print(f"{len(matrix)} configurations and widths, {len(samples)} checked batches")
for load in loads:
    for rows in row_counts:
        for alt in ("exact", "padded"):
            chosen = [r for r in matrix if r["load_path"] == load and r["weight_rows"] == rows
                      and r["distinct_alternative"] == alt]
            key = "alternative_paired_median_latency_change_percent"
            clear_alt = [r["columns"] for r in chosen if r["alternative_faster_rounds"] >= 6 and r[key] <= -5]
            clear_current = [r["columns"] for r in chosen if r["current_faster_rounds"] >= 6 and r[key] >= 5]
            overall_delta = statistics.median(r[key] for r in chosen)
            print(load, rows, alt, f"median delta {overall_delta:+.1f}%, clear alternative {clear_alt}",
                  f"clear current {clear_current}")
