# Arrow Lake-H benchmark archive

This directory contains the original timing loops and data for the [Q8_0 column sweep](q8-column-sweep/) and the [small GLU tile sweep](small-glu-tiles/), plus the [Q8_0 tail cache timing study](q8-tail-cache/) and the [long-K SLM comparison](slm-long-k/). The programs build against the [kernel snapshot](snapshot/xmx/kernel.hpp) used for the sweeps and reuse the repository's deterministic fixtures and CPU reference. The snapshot preserves the dispatch policy before the later narrow GLU changes. The unfused sweep also carries its [original exact-tail comparison](q8-column-sweep/legacy-column-tiles.hpp).

## Build

Run from the repository root with Intel oneAPI. These binaries target Arrow Lake-H (`12.74.4`) and use software loads. The SLM study explicitly instantiates both K-reduction paths.

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake -S tests/history/benchmarks -B build-benchmarks/row \
    -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DXMX_ENABLE_GATHER_TRANSPOSE=OFF
cmake --build build-benchmarks/row --parallel 2
cmake -S tests/history/benchmarks -B build-benchmarks/gather \
    -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DXMX_ENABLE_GATHER_TRANSPOSE=ON
cmake --build build-benchmarks/gather --parallel 2
```

## Run one block

Run the small DPAS probe first. The runner requires AC power and access to the GPU and kernel journal. It checks every completed batch against the CPU reference, verifies output guards, and stops on a failure. Use a single monitored invocation at a time.

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 build-benchmarks/row/xmx_benchmark_dpas_probe --run-gpu --n 8 --accumulator

python3 tests/history/benchmarks/run.py columns --build-dir build-benchmarks/row \
    --output-dir build-benchmarks/replay-columns --rows 4096 --first 1 --last 16

python3 tests/history/benchmarks/run.py glu --build-dir build-benchmarks/row \
    --output-dir build-benchmarks/replay-glu --quant q4_k
```

The archived column sweep covered both builds, 1,024 and 4,096 weight rows, and the four blocks 1–16, 17–32, 33–48, and 49–64. The GLU sweep covered both builds and the six formats `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, and `q8_0` at 4,096 weight rows. Each GLU invocation covers SwiGLU and GeGLU. The default K is 4,096 for both studies.

## Recreate the CSV and chart

After collecting all blocks for a study, pass its run directory to the corresponding scripts. Plotting uses Matplotlib.

```bash
python3 tests/history/benchmarks/q8-column-sweep/summarize.py build-benchmarks/replay-columns
python3 tests/history/benchmarks/q8-column-sweep/plot.py build-benchmarks/replay-columns

python3 tests/history/benchmarks/small-glu-tiles/summarize.py build-benchmarks/replay-glu
python3 tests/history/benchmarks/small-glu-tiles/plot.py build-benchmarks/replay-glu
```

## Q8_0 tail cache timing

The program compares grouped and interleaved orders of the same requests for widths 34–40. Each order runs with the generic tail kernel and with a separate device entry for each exact tail. It uses GPU event timestamps and validates each variant against the CPU reference before timing.

```bash
bash tests/history/benchmarks/q8-tail-cache/run.sh build-benchmarks/row \
    build-benchmarks/replay-cache/row
bash tests/history/benchmarks/q8-tail-cache/run.sh build-benchmarks/gather \
    build-benchmarks/replay-cache/gather
```

## SLM at long K

Both K-reduction paths run in the same AOT program and use the same Q8_0 inputs. Each run takes a row count, column count, and K. The dedicated fixture build permits up to 64M weights for the longer cases.

```bash
bash tests/history/benchmarks/slm-long-k/run.sh build-benchmarks/row \
    build-benchmarks/replay-slm/row-16x1-k512 16 1 512 1 1
bash tests/history/benchmarks/slm-long-k/run.sh build-benchmarks/row \
    build-benchmarks/replay-slm/row-4096x1-k16384 4096 1 16384
```

The archived comparison used 4,096 rows, one and sixteen columns, K values 1,024, 4,096, and 16,384, and both load configurations. [Summary CSV](slm-long-k/summary.csv)
