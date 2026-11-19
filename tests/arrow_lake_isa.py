#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check Arrow Lake AOT kernel instruction and scratch budgets."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


AOT_SECTION = "__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen"
TRAITS = {"q8_0": "q8_0ILb0", "q2_k": "q2_kILb0", "q3_k": "q3_kILb0", "q4_k": "q45_kILb0ELb0",
          "q5_k": "q45_kILb1ELb0", "q6_k": "q6_kILb0"}
K_BACK_EDGE = re.compile(r"\bgoto\.b\s+\([^)]*\)\s+(L\d+)\s+(L\d+)")
ACTIVATION_LOAD = re.compile(r"\bload\.ugm\.d(?:32x(?:8|16)|64x(?:4|8))t\.a64\b")
HALF_GATHER = re.compile(r"\b(?:a64 byte gathering read 16b|load\.ugm\.d16u32\.a64)\b")
PACKED_SCALE_GATHER = re.compile(r"\bload\.ugm\.d32x4\.a64\b")
DPAS = re.compile(r"^\s*dpas\.\S+\s+\((\d+)\|", re.MULTILINE)
ONES_MOV = re.compile(r"\bmov\s+\((8|16)\|[^)]*\)\s+r(\d+)\.0<1>:\w+\s+0x0*1010101:")
DPAS_B_REGISTER = re.compile(r"^\s*dpas\.8x1\s+\(8\|[^)]*\)\s+\S+\s+\S+\s+r(\d+):b\s", re.MULTILINE)
SCRATCH = re.compile(r"^ +(?:- )?type:\s+scratch\n(?: +usage:[^\n]*\n)? +size:\s+(\d+)", re.MULTILINE)
DPAS_PER_COLUMN = {"q8_0": 16, "q2_k": 24, "q3_k": 32, "q4_k": 24, "q5_k": 24, "q6_k": 32}


def run(*command):
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise ValueError(f"{command[0]} failed: {process.stderr.strip() or process.stdout.strip()}")


def k_loops(asm):
    for edge in K_BACK_EDGE.finditer(asm):
        start = asm.index(f"\n{edge[2]}:\n") + 1
        yield start, asm[start:edge.start()]


def scratch_bytes(path, metadata):
    name = path.name.removeprefix(".text.").removesuffix(".asm")
    if name not in metadata:
        raise ValueError(f"Missing .ze_info entry for {name}")
    return sum(map(int, SCRATCH.findall(metadata[name])))


def kernel_metrics(directory, metadata, quant, columns, tiled):
    # Select non-fused, non-short-row kernels with SLM and native 2D disabled.
    suffix = f"EEELi{columns}ELi1ELb0ELb0ELb{int(tiled)}EEE"
    paths = list(directory.glob(f".text.*xmx_mul_mat_submit*{TRAITS[quant]}{suffix}*.asm"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {quant} kernel for {columns} columns, tiled={tiled}, got {len(paths)}")
    path = paths[0]
    asm = path.read_text()
    loops = list(k_loops(asm))
    if len(loops) != 1:
        raise ValueError(f"Expected one K loop for {quant}, got {len(loops)}")
    start, loop = loops[0]
    widths = DPAS.findall(loop)
    if not widths or set(widths) != {"8"} or "load.ugm.d32x8.a64" not in loop:
        raise ValueError(f"Expected execution-size-8 DPAS and LSC weight gathers in the {quant} K loop")
    metrics = {
        "activation_block_loads": len(ACTIVATION_LOAD.findall(loop)),
        "dpas": len(widths),
        "fp16_scale_gathers": len(HALF_GATHER.findall(loop)),
        "scratch_bytes_per_thread": scratch_bytes(path, metadata),
    }
    if quant == "q8_0":
        metrics["packed_scale_gathers"] = len(PACKED_SCALE_GATHER.findall(loop))
    if quant in ("q4_k", "q5_k"):
        # The current compiler fills the B operand for the minimum correction
        # before the loop. An 8-wide int8 operand spans eight 32-byte GRFs.
        ones_registers = set()
        for width, register in ONES_MOV.findall(asm[:start]):
            base = int(register)
            ones_registers.update(range(base, base + int(width) // 8))
        metrics["dpas_with_preloaded_ones"] = sum(
            all(register in ones_registers for register in range(base, base + 8))
            for base in map(int, DPAS_B_REGISTER.findall(loop))
        )
    return metrics


def partial_column_metrics(directory, metadata, quant, single_tail):
    pattern = f".text.*xmx_mul_mat_submit_partial_columnsINS_*{TRAITS[quant]}EEELb0ELb{int(single_tail)}EEE*.asm"
    paths = list(directory.glob(pattern))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"Expected one {quant} partial-column kernel, got {len(paths)}")
    path = paths[0]
    loops = [DPAS.findall(loop) for _, loop in k_loops(path.read_text())]
    if any(set(widths) != {"8"} for widths in loops):
        raise ValueError(f"Expected execution-size-8 DPAS in every {quant} partial-column loop")
    return {"dpas_by_tile": sorted(map(len, loops)), "scratch_bytes_per_thread": scratch_bytes(path, metadata)}


def fused_scratch(metadata):
    report = {}
    for quant, trait in TRAITS.items():
        # Cover both row traits, both launch geometries, and all emitted K split sizes.
        row_traits = trait.removesuffix("0") + "[01]EEE"
        normal = re.compile(r"xmx_mul_mat_submitINS_.*" + row_traits + r"Li\d+ELi\d+ELb1ELb0ELb[01]EEE")
        tail = re.compile(r"xmx_mul_mat_submit_partial_columnsINS_.*" + row_traits + r"Lb1ELb[01]EEE")
        sizes = [sum(map(int, SCRATCH.findall(entry))) for symbol, entry in metadata.items()
                 if normal.search(symbol) or tail.search(symbol)]
        if not sizes:
            raise ValueError(f"No fused software kernels for {quant}")
        report[quant] = {"compiled_kernels": len(sizes), "scratch_bytes_per_thread": max(sizes)}
    return report


def fused_activation_loads(directory):
    counts = {}
    for quant in ("q8_0", "q4_k"):
        suffix = "EEELi8ELi1ELb1ELb0ELb1EEE"
        paths = list(directory.glob(f".text.*xmx_mul_mat_submit*{TRAITS[quant]}{suffix}*.asm"))
        if len(paths) != 1:
            raise ValueError(f"Expected one tiled eight-column GLU kernel for {quant}")
        counts[quant] = [len(ACTIVATION_LOAD.findall(loop)) for _, loop in k_loops(paths[0].read_text())]
    return counts


def inspect(directory, metadata):
    wide = {quant: kernel_metrics(directory, metadata, quant, 16, True) for quant in TRAITS}
    single = kernel_metrics(directory, metadata, "q8_0", 1, False)
    partial = {quant: partial_column_metrics(directory, metadata, quant, False) for quant in TRAITS}
    single_tail = {quant: partial_column_metrics(directory, metadata, quant, True) for quant in TRAITS}
    q8_dpas = wide["q8_0"]["dpas"]
    has_partial = partial["q8_0"] is not None
    tail_dpas = dict(zip(range(2, 9), partial["q8_0"]["dpas_by_tile"])) if has_partial else {}
    if has_partial:
        tail_dpas[1] = single["dpas"]
    boundary = {}
    for columns in (32, 33, 40, 41, 47, 48):
        full, remainder = divmod(columns, 16)
        tail_width = 0
        if remainder:
            tail_width = remainder if has_partial and remainder <= 8 else 16
        boundary[columns] = {"column_tiles": full + bool(tail_width),
                             "padded_columns": full * 16 + tail_width - columns,
                             "q8_0_dpas_per_row_tile": full * q8_dpas +
                             (tail_dpas.get(tail_width, q8_dpas) if tail_width else 0)}
    return {"per_k256": {"tiled_16_columns": wide, "q8_0_single_column": single,
                        "partial_column_kernels": partial, "single_column_tail_kernels": single_tail,
                        "current_column_boundary": boundary,
                        "fused_activation_loads_by_projection": fused_activation_loads(directory)}}


def check_budgets(kernels):
    failures = []

    def limit(case, metrics, name, maximum):
        actual = metrics[name]
        if actual > maximum:
            failures.append(f"{case}: {name}={actual} exceeds {maximum}")

    for quant, metrics in kernels["tiled_16_columns"].items():
        limit(quant, metrics, "activation_block_loads", 80 if quant in ("q8_0", "q4_k") else 144)
        limit(quant, metrics, "scratch_bytes_per_thread", 0)
        limit(f"{quant} 16 columns", metrics, "dpas", 16 * DPAS_PER_COLUMN[quant])
    single = kernels["q8_0_single_column"]
    limit("q8_0 single column", single, "fp16_scale_gathers", 0)
    if single["packed_scale_gathers"] != 1:
        failures.append("q8_0 single column: expected one packed scale gather")
    for quant, counts in kernels["fused_activation_loads_by_projection"].items():
        if len(counts) != 2 or any(not 0 < count <= 40 for count in counts):
            failures.append(f"{quant} fused: expected at most 40 activation loads per projection, got {counts}")
    for kind, widths in (("partial_column_kernels", (*range(2, 9), 16)), ("single_column_tail_kernels", (1, 16))):
        for quant, metrics in kernels[kind].items():
            expected = [width * DPAS_PER_COLUMN[quant] for width in widths]
            if metrics is None or metrics["dpas_by_tile"] != expected:
                failures.append(f"{quant}: missing exact-width DPAS paths for {kind}")
            if metrics is not None:
                limit(f"{quant} {kind}", metrics, "scratch_bytes_per_thread", 0)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="xmx_kernel_tests executable or extracted zebin")
    parser.add_argument("--zebin", action="store_true", help="Read an extracted device image directly")
    parser.add_argument("--fused-only", action="store_true", help="Check scratch in fused kernels in a row-load or SLM build")
    args = parser.parse_args()
    try:
        image = args.image.resolve(strict=True)
        with TemporaryDirectory(prefix="xmx-arrow-lake-isa-") as tmp:
            directory = Path(tmp)
            if not args.zebin:
                zebin = directory / "device.bin"
                # Supply a throwaway objcopy output to leave the input executable alone.
                run("objcopy", "--dump-section", f"{AOT_SECTION}={zebin}", str(image), str(directory / "host"))
                image = zebin
            dump = directory / "isa"
            run("ocloc", "disasm", "-file", str(image), "-dump", str(dump))
            entries = re.split(r"^  - name:\s+(\S+)\s*\n", (dump / ".ze_info").read_text(), flags=re.MULTILINE)
            metadata = dict(zip(entries[1::2], entries[2::2]))
            report = {} if args.fused_only else inspect(dump, metadata)
            report["fused_scratch"] = fused_scratch(metadata)
        print(json.dumps(report, indent=2))
        failures = [] if args.fused_only else check_budgets(report["per_k256"])
        for quant, metrics in report["fused_scratch"].items():
            scratch = metrics["scratch_bytes_per_thread"]
            if scratch:
                failures.append(f"{quant} fused: scratch_bytes_per_thread={scratch} exceeds 0")
        for failure in failures:
            print(failure, file=sys.stderr)
        return int(bool(failures))
    except (OSError, ValueError) as error:
        print(f"ISA check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
