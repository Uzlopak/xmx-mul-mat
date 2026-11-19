# Repository Guidelines

## Project Structure & Module Organization

This is a standalone C++17/XMX kernel investigation project. Public headers live in `include/xmx/`; fixtures and CPU reference logic are in `src/`. Tests are under `tests/`: `cpu_reference.cpp` is CPU-only, while `kernel_tests.cpp` and `dpas_probe.cpp` require Intel oneAPI SYCL. Historical logs are in `tests/history/`. `references/` contains pinned Xe2 and Xe-LPG+/Arrow Lake documentation; `third_party/llama.cpp/` preserves the extracted kernel. Build outputs belong in `build/`, `build-cpu/`, or another ignored directory.

## Build, Test, and Development Commands

Use the repository scripts from its root:

```bash
./build.sh                              # Release build, SYCL/GPU targets, safe CTest checks
./build.sh --cpu-only --build-dir build-cpu
./build.sh --aot 12.74.4 --build-dir build-aot
./run-tests.sh                           # Repeat configured CTest checks
python3 scripts/check_provenance.py --strict
```

`--cpu-only` uses the system C++ compiler; the default and AOT builds require Intel `icpx`/oneAPI. CTest deliberately does not launch GPU kernels. Run GPU cases explicitly and cautiously with `./run-tests.sh --gpu ...`; start with a small DPAS probe as documented in `README.md`.

## Coding Style & Naming Conventions

Follow the existing C++ style: four-space indentation, braces on the same line, C++17 features, and descriptive `snake_case` functions/variables. Keep public APIs and kernel traits in `include/xmx/`, and place reusable test data or CPU calculations in `src/`. Compile warnings are enabled with `-Wall -Wextra -Wpedantic`; preserve warning-clean code. Shell scripts should remain Bash-compatible and pass `bash -n`.

## Testing Guidelines

Run the CPU/reference suite through `./build.sh --cpu-only` or `ctest --test-dir build-cpu --output-on-failure`. GPU correctness is opt-in: use `--run-gpu` for a single case and `--suite` only after small cases pass. GPU tests compare against deterministic CPU references; do not treat a successful compile as runtime validation. Preserve relevant output and kernel logs under `tests/history/` when investigating failures.

## Commit & Pull Request Guidelines

There is no existing commit history to establish a project-specific convention. Use concise imperative subjects, preferably scoped (for example, `tests: add q6_k tail case`), and keep unrelated changes separate. Pull requests should explain the kernel or build impact, list exact validation commands and results, identify hardware/oneAPI versions for GPU work, and include failure logs or benchmark JSON when relevant. Call out changes to vendored sources, provenance, or licensing metadata explicitly.

## Safety and Configuration Notes

Do not run GPU suites in unattended loops. The current extracted Xe2 kernel has previously caused an Arrow Lake-H/Arc device reset; stop after a device-loss error and retain `journalctl -k` output. Do not modify the preserved original kernel or provenance files without documenting the reason.

## Xe-LPG+ Porting Target

The primary development target is to rewrite the XMX kernel for the Xe-LPG+ architecture used by this machine. The current bring-up keeps `XMX_ENABLE_SLM=OFF` and `XMX_ENABLE_TRANSPOSED_2D_LOADS=OFF`: one work-item performs the K reduction and weights/scales use row-by-row block loads. Re-enable either feature only after the isolated path is validated. Use `cmake -DXMX_ENABLE_SLM=ON` or `-DXMX_ENABLE_TRANSPOSED_2D_LOADS=ON` for explicit experiments. Validate changes incrementally, starting with the smallest DPAS probe:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
./build/xmx_dpas_probe --run-gpu --n 8 --accumulator
```

The expected result is `"passed":true`. If it succeeds, test the 16-wide variant:

```bash
./build/xmx_dpas_probe --run-gpu --n 16 --accumulator
```

On the current target, the 16-wide probe may compile but is not a substitute for the Xe-LPG+ execution-size-8 path; the recorded attempt produced correct values only for the first half. Treat that as an architecture/runtime finding and keep the kernel on the split 8-wide form unless hardware evidence proves otherwise.

If both probes pass, run a small full matrix-kernel comparison against the CPU reference:

```bash
./run-tests.sh --gpu --type q8_0 --rows 16 --cols 1 --k 1024
```

Stop immediately on `DEVICE_LOST` or a driver reset. The unchanged full kernel has already reset this GPU; passing a DPAS probe alone does not establish that the complete kernel works.
