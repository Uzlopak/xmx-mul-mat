#!/usr/bin/env python3
"""Run one bounded GPU block from the historical benchmark sweeps."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("columns", "glu"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quant", choices=("q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"), default="q8_0")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()

    fused = args.study == "glu"
    last = args.last if args.last is not None else (8 if fused else min(args.first + 15, 64))
    rounds = args.rounds if args.rounds is not None else (8 if fused else 7)
    if not 1 <= args.first <= last <= (8 if fused else 64) or last - args.first >= 16:
        parser.error("expected a valid block of at most 16 columns, or one through eight for GLU")
    if not fused and args.quant != "q8_0":
        parser.error("the unfused column sweep uses Q8_0")

    build = args.build_dir.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    match = re.search(r"^XMX_ENABLE_GATHER_TRANSPOSE:BOOL=(ON|OFF)$", cache, re.MULTILINE)
    if not match:
        parser.error("configure the benchmark build with XMX_ENABLE_GATHER_TRANSPOSE=ON or OFF")
    load = "gather" if match[1] == "ON" else "row"
    binary = build / ("xmx_small_glu_sweep" if fused else "xmx_column_sweep")
    shape = (args.rows, args.k, args.first, last, rounds, args.iterations)
    problem = (args.quant, *shape) if fused else shape
    command = ("timeout", "90s", binary, "--run-gpu", *problem)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{load}-{args.quant}-{args.rows}-k{args.k}-{args.first}-{last}"
    if not fused:
        prefix = args.output_dir / f"{load}-{args.rows}-k{args.k}-{args.first}-{last}"
    failures = re.compile(r"device[ _-]lost|engine reset|GPU (hang|reset|recovery)|timed.out job|"
                          r"exec queue reset|device coredump|xe.*wedged", re.IGNORECASE)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    def check_power():
        if Path("/sys/class/power_supply/AC/online").read_text().strip() != "1":
            raise SystemExit("AC power disconnected")

    def check_journal():
        result = subprocess.run(("journalctl", "-k", "--since", started_at, "--no-pager"),
                                capture_output=True, text=True, timeout=8)
        log = prefix.with_suffix(".kernel.log")
        log.write_text(result.stdout + result.stderr)
        if result.returncode or failures.search(result.stdout + result.stderr):
            raise SystemExit(f"Kernel journal check failed. See {log}")

    expected = set()
    for columns in range(args.first, last + 1):
        if fused:
            widths = range(columns, 9) if args.quant == "q8_0" else sorted({columns, 8})
            expected.update((columns, r, (glu, width)) for r in range(rounds)
                            for glu in ("swiglu", "geglu") for width in widths)
        else:
            schedules = ["current"]
            if columns > 8 and columns % 16:
                schedules.append("exact" if columns % 16 > 8 else "padded")
            expected.update((columns, r, schedule) for r in range(rounds) for schedule in schedules)

    check_power()
    check_journal()
    log = prefix.with_suffix(".log")
    with log.open("w") as errors, prefix.with_suffix(".jsonl").open("w") as raw:
        environment = {**os.environ, "ONEAPI_DEVICE_SELECTOR": "level_zero:0"}
        process = subprocess.Popen(tuple(map(str, command)), stdout=subprocess.PIPE, stderr=errors,
                                   env=environment, text=True)
        try:
            seen = set()
            previous = None
            for line in process.stdout:
                check_power()
                value = json.loads(line)
                columns = value["columns"]
                variant = (value["glu"], value["tile_width"]) if fused else value["schedule"]
                label = (columns, value["round"], variant)
                if label not in expected or label in seen or not value["passed"]:
                    raise SystemExit(f"Unexpected benchmark sample. See {log}")
                if previous is not None and columns != previous:
                    check_journal()
                previous = columns
                seen.add(label)
                raw.write(line)
                raw.flush()
            check_journal()
            if process.wait(timeout=10):
                raise SystemExit(f"Benchmark stopped with code {process.returncode}. See {log}")
            if seen != expected:
                raise SystemExit(f"Incomplete benchmark. See {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print(f"{args.study}, {args.quant}, {load}, columns {args.first}..{last}: {len(seen)} passed")


if __name__ == "__main__":
    main()
