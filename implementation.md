# Xe-LPG+ XMX Kernel Port — Implementation Handoff

## Objective

Rewrite `include/xmx/kernel.hpp` for the Xe-LPG+ architecture in this machine (Arrow Lake-H / Intel Arc Graphics, PCI ID `8086:7d51`, IP `12.74.4`). Use Xe-LPG+-compatible DPAS instructions, operand layouts, accumulator types, register usage, and memory rules. Use `references/xe-lpg-plus/` and the installed oneAPI headers as authoritative. Preserve the CPU reference, deterministic fixtures, public interfaces, provenance metadata, the original snapshot in `third_party/llama.cpp/`, and CPU-only behavior.

Implement incrementally: validate the primitive DPAS path, port `Q8_0`, validate all Q8_0 shapes and tails, then port and validate `Q4_K`, `Q6_K`, and remaining paths. Compare every GPU result with the CPU reference, including relative L2 error, maximum absolute error, NaN/Inf checks, and tail handling. Do not claim completion from compilation alone. Performance is a later phase.

For ordinary failures, continue debugging from the smallest failing case. For `DEVICE_LOST`, GPU hangs, driver resets, or similar hardware instability, stop GPU execution immediately, preserve output and `journalctl -k`, and do not automatically retry variants.

## Final Status (Authoritative)

The Xe-LPG+ port covers all six quantization formats in the extracted Xe2 kernel. The [validation matrix](#latest-arrow-lake-aot-validation) records 1,014 GPU comparisons and zero scratch across 432 compiled GLU variants. It covers row loads, LSC gathers, short-row layouts, SwiGLU, GeGLU, and SLM. The default build uses row loads with SLM and native 2D loads disabled.

The porting fixes use the full activation stride for row-load pitches, lay out paired activations as `[group][row][K]`, retain strided-select VNNI weight layout, and use execution-size-8 DPAS with `RepeatCount=1` per output column. CPU build/tests, CTest, provenance, and formatting checks pass. The later [performance follow-up](#arrow-lake-performance-follow-up) records current tiling and minimum corrections.

The unfused software path without SLM uses 16-column work-group tiles. Remainders of one to eight columns use one exact-width tile. Remainders of nine to fifteen columns use one 16-column tile. Fused GLU uses work-group tiles up to eight columns. The software build without SLM or native 2D loads submits up to 64 GLU columns at a time. The optional SLM and native 2D builds retain the 16-column submission limit. A partial tile clamps software activation loads to the last valid column and bounds all output stores to the request. The native 2D path retains its hardware height bound.

The `native_2d_geometry` host test covers the launch geometry, tile decomposition, and `xmx_activation_source_row` bounds. GPU comparisons use output guard columns to detect writes beyond a requested tail.

Each 16-row weight dot product uses two execution-size-8 DPAS operations. The native 16-wide probe failed on its second half, and the tested Level Zero driver reports no native 2D block I/O on this device. [Device capability and software loads](#transposed-load-follow-up)

## Transposed Load Follow-up

On the Arc 140T (PCI ID `8086:7d51`, revision `03`, IP `12.74.4`), Level Zero driver `1.18.40196+8` reported `has_2d_block_io_support=false`. The offline compiler's extension query for `12.74.4` also omitted `cl_intel_subgroup_2d_block_io`, which it listed for Lunar Lake. The launcher now checks the ESIMD device capability before submitting kernels built with `XMX_ENABLE_TRANSPOSED_2D_LOADS=ON`. The earlier synthetic probe used a 32-byte surface width and pitch. Intel's SPIR-V extension specifies a 64-byte minimum width. [ESIMD load_2d requirements](references/xe-lpg-plus/esimd/sycl_ext_intel_esimd_functions.md#load_2d---load-2d-block) [Intel SPIR-V 2D block I/O](https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_2d_block_io.html)

With native 2D enabled, the launcher checks the layout before selecting a kernel. If a surface is too narrow or violates the native alignment, stride or extent requirements, it uses a fallback kernel. The fallback clamps activation reads for partial column tiles. Layouts eligible for native loads still require a device that advertises 2D block I/O. The `native_2d_geometry` CTest exercises the host-side decision without a GPU launch.

`XMX_ENABLE_GATHER_TRANSPOSE=ON` uses LSC vector gathers for weights and packed scales. Each lane reads four or eight adjacent dwords from one row. The hardware returns the dwords in `[k_dword][row]` order, which matches the DPAS B layout. The gather clamps tail lanes to the last valid row. The Arrow Lake AOT binary uses SIMD8 `load.ugm.d32x4.a64` and `load.ugm.d32x8.a64` messages. The default configuration uses row loads.

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake -S . -B build-arl-gather -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
    -DXMX_DEVICE_ARCH=12.74.4 -DXMX_ENABLE_SLM=OFF \
    -DXMX_ENABLE_TRANSPOSED_2D_LOADS=OFF -DXMX_ENABLE_GATHER_TRANSPOSE=ON
cmake --build build-arl-gather --parallel 2
ctest --test-dir build-arl-gather --output-on-failure
export ONEAPI_DEVICE_SELECTOR=level_zero:0
./run-tests.sh --build-dir build-arl-gather --gpu --type q8_0 --rows 16 --cols 1 --k 1024
```

In the initial gather bring-up, the build passed all 45 GPU comparisons at K=1024 with `relative_l2=0` and `max_abs=0`. It also passed Q6_K with one row and Q8_0 with 33 rows, 16 columns and K=32768. The kernel journal showed no GPU resets during the comparisons. The native configuration returned the capability error before kernel submission.

### Kernel Timing on Arc 140T

This bring-up comparison predates the later spill, tiling, and GLU changes. Both builds used oneAPI 2026.2, AOT target `12.74.4` and SLM disabled. Each result is the median of five batches of 64 launches per variant. The run order alternated between row loads and gathers. Times include submission and synchronization and exclude upload and fixture generation. Every timed case passed the CPU comparison.

| Quantization | Rows | Columns | K | Row loads (µs) | LSC gathers (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1024 | 1 | 4096 | 458.11 | 231.80 | 1.98× |
| Q8_0 | 1024 | 16 | 4096 | 2044.10 | 1118.96 | 1.83× |
| Q4_K | 1024 | 1 | 4096 | 579.20 | 242.55 | 2.39× |
| Q4_K | 1024 | 16 | 4096 | 1755.57 | 1245.19 | 1.41× |
| Q6_K | 1024 | 1 | 4096 | 864.73 | 466.16 | 1.86× |
| Q6_K | 1024 | 16 | 4096 | 3752.38 | 2958.10 | 1.27× |
| Q8_0 | 128 | 4 | 2048 | 370.54 | 150.40 | 2.46× |

To repeat a timing, append `--iterations 64` to the GPU test with the shape from the table. Configure a second build directory with `XMX_ENABLE_GATHER_TRANSPOSE=OFF` for the row-load baseline.

### Register Spill Follow-up

The port began accumulating one activation column at a time, so each DPAS result and scaled term covers 16 rows. The build also passes automatic GRF selection explicitly to the Arrow Lake offline compiler and to the JIT compiler. At this stage, all 15 unfused Arrow Lake AOT variants in the test harness (Q8_0, Q4_K and Q6_K at 1, 2, 4, 8 and 16 columns) had zero scratch in both load configurations.

With LSC gathers, the adjusted AOT build passed all 45 cases at K=1024. Both JIT load modes also passed all 45 cases with runtime compiler warnings enabled and no spill reports. The 16-column timings below use 1024 rows and K=4096. Each median comes from five interleaved batches of 64 launches per build, following the same timing method as above.

| Quantization | Previous gather build (µs) | Spill-free gather build (µs) | Speedup |
| --- | ---: | ---: | ---: |
| Q8_0 | 1800.93 | 472.11 | 3.81× |
| Q4_K | 1690.02 | 464.29 | 3.64× |
| Q6_K | 2804.59 | 541.48 | 5.18× |

### 32- and 64-column launches

This unfused-kernel milestone introduced a two-dimensional work-group grid with 16-row by 16-column tiles and one submission. The last tile used bounded input loads and output stores. The test CLI began using the normal column dispatcher, including 17- and 33-column partial tiles.

Both Arrow Lake AOT binaries had zero scratch across the 30 compiled kernels tested at this stage. In each load mode, each quantization used the same compiled kernel for 32- and 64-column launches.

Both AOT load modes and the default JIT build passed all 81 CPU-reference cases at K=1024. The previous single-work-item 64-column Q6_K kernel failed the CPU comparison. For timing, the baseline uses two or four separate 16-column gather launches. The comparison used 1024 rows and K=4096 on AC power in performance mode. Each median comes from five interleaved batches of 32 matrix operations.

| Quantization | Columns | Separate launches (µs) | Tiled launch (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 32 | 897.37 | 508.94 | 1.76× |
| Q4_K | 32 | 914.78 | 482.10 | 1.90× |
| Q6_K | 32 | 1039.50 | 584.81 | 1.78× |
| Q8_0 | 64 | 1624.58 | 584.81 | 2.78× |
| Q4_K | 64 | 1764.31 | 665.61 | 2.65× |
| Q6_K | 64 | 2189.48 | 749.46 | 2.92× |

## Arrow Lake Performance Follow-up

The measurements below use Arc 140T with oneAPI 2026.2, AOT target `12.74.4`, LSC gathers enabled, and SLM and native 2D loads disabled. Each comparison used 1,024 rows, K=4,096, varying scales, and seven interleaved batches of 128 launches per build. Latency is the median per-launch time, including submission and synchronization. Every timed shape passed its CPU reference comparison. Each row compares the specific baseline and candidate measured together.

### Loads and Column Boundaries

- Pairing adjacent activation reads into one 64-byte block load reduced the 16-column activation load count from 144 to 80 per K=256 for Q8_0 and Q4_K. Both stayed spill-free. The Q6_K prototype allocated 448 scratch bytes per thread, so the kernel keeps its separate loads.
- The packed Q8_0 scale gather reads all eight FP16 scales for a superblock in one message per 16 weight rows. The earlier kernel used eight gather messages. The single-column median went from 185.0 to 147.1 µs, a 20.5% latency reduction in the combined scale and tile comparison. A single-column request uses the same tile geometry in both builds.
- The initial unfused exact-tail schedule used tail tiles of up to eight columns. At 33 columns, the decomposition was 16 + 16 + 1, which reduced Q8_0 from 768 to 528 DPAS instructions per K=256 across the column tiles for one weight-row tile. At 47 columns, 16 + 16 + 8 + 7 added a work-group that loaded the weights. Q8_0 medians for 32, 33, and 47 columns were 348.6 → 333.7 µs, 293.6 → 292.1 µs, and 354.1 → 360.1 µs, respectively.
- In the Q8_0 16-column loop, the two builds each had 256 DPAS instructions. Moves decreased from 386 to 357 and `sync.nop` increased from 178 to 181. The Q8_0 dot products need both 8-row results, and the format has no minimum correction.
- A follow-up held packed scales fixed and compared the exact 8 + 7 tail with a padded 16-column tile. The padded tile computes one extra column and loads tail weights in one work-group. It took 305.4 µs versus 319.3 µs at 31 columns, and 331.3 µs versus 318.9 µs at 47 columns. The results varied by shape.

### Q4_K Minimum Correction

The Q4_K minimum correction depends on the sum of the 32 quantized activations in each group and column. The 16 weight rows have their own minimum scales. The current kernel computes the activation sum with one execution-size-8 DPAS against ones, extracts one lane, and lets the vector multiply read that scalar for all rows. The two 8-row DPAS instructions that compute the weight dot products still produce 16 distinct results. [Q4_K epilogue](include/xmx/kernel.hpp)

The 16-column disassembly below counts instructions in the K=256 loop body. Both one-DPAS experiments retain the same K-loop back edge and inner unrolling as the baseline. The vector experiment generated an integer-to-float conversion and an extra register move to build the replicated vector. The neighboring `sync.nop` instructions carry scoreboard dependencies. Each fully unrolled group and column repeats the extra work. The counts describe the generated schedule. A partially unrolled variant was outside this comparison.

| Minimum correction | Total DPAS | DPAS with ones | `mov` | `sync.nop` | Scratch bytes per thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two DPAS against ones | 512 | 256 | 683 | 216 | 0 |
| One DPAS with vector replication | 384 | 128 | 810 | 273 | 0 |
| One DPAS with scalar broadcast | 384 | 128 | 657 | 178 | 0 |

| Experiment | Columns | Before (µs) | After (µs) |
| --- | ---: | ---: | ---: |
| DP4A reduction replaced by two DPAS | 16 | 359.8 | 333.3 |
| One DPAS with vector replication | 4 | 199.8 | 186.9 |
| One DPAS with vector replication | 16 | 339.0 | 415.1 |
| One DPAS with vector replication | 33 | 391.4 | 421.5 |
| Packed Q4_K weight scales | 4 | 182.8 | 172.3 |
| Packed Q4_K weight scales | 16 | 319.7 | 327.6 |
| Packed Q4_K weight scales | 33 | 384.3 | 370.4 |
| One DPAS with scalar broadcast | 4 | 198.5 | 175.6 |
| One DPAS with scalar broadcast | 16 | 338.9 | 309.8 |
| One DPAS with scalar broadcast | 33 | 354.0 | 322.1 |

The scalar form reduced latency by 11.5%, 8.6%, and 9.0% at 4, 16, and 33 columns. The DP4A reduction and the two-DPAS baseline both used zero scratch. Packed scales reduced five Q4_K metadata messages to two but showed mixed latency. The retained optimization is the scalar form. Moves and synchronization counts help explain results, while output correctness and latency determine which version ships.

The final scalar build passed 99/99 GPU cases with varying scales at K=1,024 in the gather configuration, with zero observed numerical error. A row-load check at 17 rows and 33 columns also matched exactly. The offline ISA test checks the 384-DPAS budget at 16 columns, the exact-width tails, and zero scratch on those paths. [ISA test](tests/arrow_lake_isa.py)

### Other Minimum Corrections

Q5_K uses the same all-ones scalar correction as Q4_K. Q2_K packs a separate minimum coefficient for each 16-activation half and weight row. One DPAS with a fixed mask puts the two 16-element activation sums in separate lanes. The kernel multiplies each sum by its row-specific minimum code, adds the two products as integers, and applies the row's minimum scale. Both formats now use 384 DPAS per K=256 at 16 columns, down from 512, with zero scratch.

| Quantization | Columns | Before (µs) | After (µs) | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | 4 | 179.5 | 174.1 | 3.0% |
| Q2_K | 16 | 354.6 | 317.0 | 10.6% |
| Q2_K | 33 | 377.1 | 373.4 | 1.0% |
| Q5_K | 4 | 202.2 | 185.8 | 8.1% |
| Q5_K | 16 | 467.1 | 465.5 | 0.3% |
| Q5_K | 33 | 448.5 | 423.9 | 5.5% |

Q2_K showed the clearest gain at 16 columns. Q5_K improved at 4 and 33 columns. The smaller differences were within run-to-run variation in this comparison.

The expanded gather suite passed 165/165 CPU comparisons at K=1,024 with varying scales. All Q2_K cases matched exactly. Q5_K matched exactly in 27 of 33 cases and its maximum absolute error was `9.53674e-07` across the full set. The default row-load build also matched exactly for both formats at 17 rows and 33 columns. The ISA test covers both formats at 16 columns and through the exact-width tails.

## Xe-LPG+ Functional Coverage

The Arrow Lake-H AOT builds used oneAPI 2026.2 and target `12.74.4`. This coverage milestone preceded the GLU register-spill fix. The three build configurations used default row loads, LSC gathers, and LSC gathers with `XMX_ENABLE_SLM=ON`, respectively. All seven suites used `--vary-scales`. Each compared Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0 against the CPU reference for three row counts and eleven column counts, with output guards for column tails.

| Load path | SLM | GPU suite options beyond `--suite --vary-scales` | Passed | Max absolute error |
| --- | --- | --- | ---: | ---: |
| Rows | Off | `--k 1024` | 198/198 | 9.54e-07 |
| LSC gathers | Off | `--k 1024` | 198/198 | 9.54e-07 |
| LSC gathers | Off | `--k 1024 --short-rows --glu swiglu` | 198/198 | 1.53e-05 |
| Rows | Off | `--k 256 --short-rows --glu geglu` | 198/198 | 1.91e-06 |
| LSC gathers | On | `--k 1280` | 198/198 | 9.54e-07 |
| LSC gathers | On | `--k 2304 --glu swiglu` | 198/198 | 1.53e-05 |
| LSC gathers | On | `--k 2304 --short-rows --glu geglu` | 198/198 | 1.53e-05 |

The kernel journal showed no device resets in the 1,386 suite cases. Targeted tests also passed for Q3_K with one row and K=256, 65 columns across submissions, K=32,768, and an uneven SLM split at K=4,352. The fused test uses separate up and gate weights with shared activations. The CPU reference evaluates both projections and the chosen activation.

An AOT Q8_0 build with native 2D enabled passed through the software fallback at K=256, 17 rows, and 17 columns. The device rejected `--require-native-2d` before submission because it reports no native 2D block I/O.

### Fused GLU Register Spills

The original fused schedule kept both projections' weight tiles and scales live alongside the two result accumulators. It used up to 5,056 bytes of scratch per work-item with row loads and 4,384 bytes with LSC gathers. The spill fix computes the up projection and then the gate projection within the same submission. Each projection loads the shared input from memory, and only one set of weight operands and scales is live at a time. GLU work-groups handle at most eight activation columns.

The final work-group can cover a partial eight-column tile. The kernel clamps software input loads to the last valid column and bounds both block stores and scatter stores to the output width. A Q8_0 regression at 15 columns exercised both output layouts with 16 and 17 weight rows. [Prototype failure and fix](tests/history/glu-tail-2026-11-18.log)

The Arrow Lake AOT ISA check reads scratch allocation sizes from `.ze_info` and requires zero scratch for the fused software kernels. It found zero scratch in all 432 emitted fused variants across row-load and gather builds, each with SLM off and on. Both short-row settings are included. The previous gather binary fails the limit for all six formats.

The timings compare two Xe-LPG+ builds on Arrow Lake-H, before and after the GLU spill fix, using LSC gathers with SLM disabled, 1,024 weight rows, K=4,096, and varying scales. Both builds used the 16-column submission limit. Each median comes from seven interleaved batches of 128 launches per build. The timings include submission and synchronization. All 224 batches passed their CPU reference comparison, and the kernel journal reported no device errors.

| Format | SwiGLU at 16 columns (µs) | Speedup | GeGLU at 16 columns (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | 723.0 → 392.1 | 1.84× | 748.9 → 403.4 | 1.86× |
| Q3_K | 1,634.6 → 404.1 | 4.04× | 1,583.6 → 408.9 | 3.87× |
| Q4_K | 893.2 → 392.3 | 2.28× | 925.2 → 406.4 | 2.28× |
| Q5_K | 1,423.5 → 398.1 | 3.58× | 1,476.8 → 362.3 | 4.08× |
| Q6_K | 1,560.9 → 412.9 | 3.78× | 1,572.1 → 458.0 | 3.43× |
| Q8_0 | 1,008.7 → 428.8 | 2.35× | 992.1 → 438.2 | 2.26× |

| SwiGLU format | Columns | Before (µs) | After (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Q5_K | 4 | 347.7 | 306.2 | 1.14× |
| Q8_0 | 4 | 362.0 | 338.3 | 1.07× |
| Q5_K | 33 | 3,012.2 | 934.0 | 3.23× |
| Q8_0 | 33 | 2,182.8 | 1,096.7 | 1.99× |

The final kernel was faster in all seven paired batches for each of the 16- and 33-column comparisons. Both four-column cases were faster in six of seven pairs.

These timings cover the standalone GLU kernel with LSC gathers and SLM disabled. The follow-up below includes row-load and SLM timings. End-to-end model throughput remains unmeasured. To reproduce one measured shape, use the same options with each AOT build.

```bash
./run-tests.sh --build-dir build-arl-gather --gpu --type q5_k --glu swiglu \
    --rows 1024 --cols 16 --k 4096 --vary-scales --iterations 128
```

### GLU Submission and Activation Loads

The no-SLM software launcher now submits up to 64 GLU columns at once. Each work-group still handles up to eight columns. The cap-only prototype used byte-identical eight-column GPU code for Q8_0 and Q5_K. The SLM and native 2D builds keep their existing chunk size and K splitting. Q8_0 and Q4_K again use paired 64-byte activation loads in each GLU projection. The ISA check counts 40 activation loads per projection per K=256 in both gather kernels, down from 72.

The combined build used seven interleaved batches of 128 launches per variant with 1,024 weight rows and K=4,096. The launcher and paired-load change together reduced Q8_0 SwiGLU latency at 64 columns from 1,568.1 to 791.1 µs with gathers and from 2,126.7 to 1,287.3 µs with row loads. Both improved in all seven pairs. With the 64-column cap fixed in both builds, paired loads alone reduced the gather median from 937.9 to 768.9 µs, also faster in all seven pairs. At 16 columns the final Q8_0 comparison was essentially flat, and the two Q4_K runs gave mixed median results.

A follow-up used the same before-and-after builds and timing method for SwiGLU across all six formats at 64 columns. The AOT builds ran on Arrow Lake-H with SLM disabled. Every 64-column case improved in all seven pairs.

| Format | Gathers before → after (µs) | Speedup | Row loads before → after (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | 1,337.0 → 1,012.2 | 1.32× | 1,893.1 → 1,211.8 | 1.56× |
| Q3_K | 1,457.8 → 1,180.1 | 1.24× | 1,820.2 → 1,322.2 | 1.38× |
| Q4_K | 1,412.4 → 886.6 | 1.59× | 1,692.5 → 1,057.1 | 1.60× |
| Q5_K | 1,379.0 → 1,021.6 | 1.35× | 1,938.5 → 1,297.6 | 1.49× |
| Q6_K | 1,601.6 → 1,159.8 | 1.38× | 5,597.1 → 4,039.3 | 1.39× |
| Q8_0 | 1,607.9 → 780.0 | 2.06× | 2,157.6 → 1,251.0 | 1.72× |

The Q8_0 single-column control had byte-identical GPU code in both revisions of each load configuration. Its medians were 288.5 → 265.5 µs with gathers and 798.6 → 778.3 µs with row loads. Those differences indicate the variation of this timing method on the control shape. All 196 batches passed their CPU reference checks and the kernel journal reported no device errors.

Additional measurements supported keeping SLM opt-in. With gathers, Q5_K GeGLU at 16 rows, one column, and K=4,096 went from 212.3 to 62.9 µs with SLM. Q8_0 SwiGLU at 1,024 rows and 16 columns went from 389.9 to 299.4 µs at the same K. The short Q5_K case at K=256 improved in four of seven pairs, which leaves the crossover unresolved.

A Q8_0 prototype shared each loaded activation group between up and gate and streamed one weight group at a time. The eight-column gather kernel emitted 40 activation loads in total per K=256 and used zero scratch, down from 40 loads per projection. At 64 columns the median moved from 755.6 to 734.7 µs with gathers and from 1,254.3 to 1,212.3 µs with row loads. The prototype was faster in three of seven gather pairs and five of seven row-load pairs. The one-column gather comparison was essentially flat. These measurements leave the shared-load schedule experimental.

A second Q8_0 prototype reused the activations across 128 consecutive K values, running the up and gate projections in turn on each group of 128. The tiled eight-column gather kernel also emitted 40 activation loads per K=256 with zero scratch. At 64 columns, 1,024 weight rows, and K=4,096, the median increased from 803.8 to 1,146.6 µs with gathers and from 1,216.1 to 1,806.0 µs with row loads. All seven pairs regressed in each build. A repeat of the per-32 schedule with gathers at the same shape was 0.7% faster at the median of the paired changes, with four of seven pairs faster. The production schedule still loads the activations separately for the two projections.

#### Three-column GLU

A Q8_0 benchmark compared the exact three-column kernel with the bounded eight-column GLU kernel for the same three requested columns. Both issued one work-group per 16 weight rows in one submission. The test used 4,096 weight rows, K=4,096, varying scales, SLM disabled, and seven alternating pairs of 128 launches on AC power. All 56 batches passed the CPU-reference check and preserved all five output guard columns.

| Operation | Load path | Exact three (µs) | Padded eight (µs) | Padded faster pairs |
| --- | --- | ---: | ---: | ---: |
| SwiGLU | Row loads | 1,585.7 | 1,078.4 | 6/7 |
| GeGLU | Row loads | 1,481.0 | 981.3 | 7/7 |
| SwiGLU | Gathers | 987.5 | 929.7 | 5/7 |
| GeGLU | Gathers | 1,008.0 | 980.0 | 4/7 |

The AOT device code uses 96 DPAS instructions per K=256 with three columns and 256 with eight, counting both projections. Both kernels use zero scratch. The wider build uses paired 64-byte activation loads and 256 GRFs per work-item. The exact build uses separate 32-byte activation loads and 128 GRFs. The timings measure the combined changes in the generated code.

#### GLU Sweep from One to Eight Columns

The [benchmark programs and replay commands](tests/history/benchmarks/README.md) are archived with the measurement data.

All six formats used the exact kernel and a bounded eight-column tile for requests of one through seven columns. Eight requested columns used its exact kernel. Q8_0 also tested every intermediate tile width. The sweep used SwiGLU and GeGLU, both load paths, 4,096 weight rows, K=4,096, and varying scales. Each schedule ran in eight rounds of 32 launches with rotated ordering on AC power. The 444 cells produced 3,552 successful CPU-reference and output guard checks. The AOT scratch check passed in both load builds.

| Quantization | Padded eight vs exact, row loads | Padded eight vs exact, gathers |
| --- | ---: | ---: |
| Q2_K | +26.4% | +31.2% |
| Q3_K | +28.2% | +32.4% |
| Q4_K | +4.4% | −1.8% |
| Q5_K | +9.1% | +5.9% |
| Q6_K | +110.1% | +3.0% |
| Q8_0 | −1.3% | −1.3% |

Each entry is the median of the paired latency changes for the 14 combinations of requested width and GLU operation. Positive values indicate higher latency for padding. Q6_K row loads favored exact widths by at least 5% in the median paired change and in at least six of eight rounds for all 14 combinations. At three Q8_0 SwiGLU columns with row loads, the exact kernel took 1,515 µs, and tile widths four, six, and eight took 976, 968, and 999 µs respectively. The same shape with GeGLU took 1,601 µs at width three and 977 µs at width four. [Summary CSV](tests/history/benchmarks/small-glu-tiles/small-glu-tiles.csv) · [Raw samples](tests/history/benchmarks/small-glu-tiles/small-glu-tiles-samples.csv) · [Chart](tests/history/benchmarks/small-glu-tiles/small-glu-tiles.svg)

#### Narrow GLU Dispatch

The host launcher selects a common bounded four-column kernel for regular-row Q8_0 GLU requests of two through four columns with row loads. Regular-row Q4_K GLU requests of one column reuse the existing bounded eight-column kernel with either load path. Both rules apply to the software configuration with SLM and native 2D loads disabled. The launcher passes the original request width to the device, where activation loads and output stores use that bound. [Tile selection](include/xmx/kernel.hpp#L1027)

The Arrow Lake-H AOT row-load image contains one bounded Q8_0 entry for widths two through four. The Q4_K single-column rule uses the existing bounded eight-column entry in both images. Both AOT builds passed all six CTest checks. The SwiGLU and GeGLU GPU suites passed 258 cases each in both load configurations, with row tails and output guards for partial GLU tiles.

The integrated builds were compared with the previous dispatcher at 4,096 weight rows and K=4,096 on AC power. Each case used seven alternating pairs of 128 launches. All 84 timing batches passed the CPU reference comparison.

| Format | GLU | Load path | Columns | Previous (µs) | Integrated (µs) | Faster pairs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Q8_0 | SwiGLU | Row loads | 2 | 1,478.2 | 1,006.8 | 7/7 |
| Q8_0 | GeGLU | Row loads | 3 | 1,579.8 | 1,024.2 | 7/7 |
| Q8_0 | SwiGLU | Row loads | 4 | 1,030.1 | 1,005.2 | 5/7 |
| Q8_0 | GeGLU | Row loads | 4 | 996.3 | 1,013.4 | 3/7 |
| Q4_K | SwiGLU | Row loads | 1 | 962.3 | 568.0 | 7/7 |
| Q4_K | SwiGLU | Gathers | 1 | 578.3 | 525.3 | 5/7 |

### SLM at 4,096 Weight Rows

The unfused Q8_0 study explicitly instantiated the same launch geometries as the SLM-off and SLM-on software builds, within one AOT binary per load path. Both used identical weights, activations, work-group tiles, and a single submission per matrix operation. Each comparison ran nine alternating pairs of 32 launches on AC power with varying scales. All 216 timed batches passed the CPU reference and output guard checks.

| K | Row loads, one column | Row loads, 16 columns | Gathers, one column | Gathers, 16 columns |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | −15.8% | −7.4% | −12.7% | +2.0% |
| 4,096 | −8.7% | −7.2% | −6.3% | −7.7% |
| 16,384 | −23.7% | −23.0% | −14.3% | −10.8% |

These are the paired median latency changes when enabling SLM. At K=16,384, SLM improved all nine pairs for every measured shape and load path. At K=4,096, it improved at least six of nine pairs in each case. With short K=1,024 and 16 columns, the gather result was mixed. The measurements cover regular-row Q8_0 at 4,096 rows and two column widths. [Measurement CSV](tests/history/benchmarks/slm-long-k/summary.csv) · [Benchmark program](tests/history/benchmarks/slm-long-k/benchmark.cpp)

### Column Width and Tail Code Size

The extracted Xe2 launcher selects exact widths 1–8 and rounds wider unfused requests to 16, 32, or 64 columns per submission. It uses `NC=64` and `ncv=47` for 47 columns, with stores bounded by `ncv`. The original fused launcher has a 16-column submission cap in this project. The Xe-LPG+ host dispatcher retains the same rounding tiers. Unfused software work-groups handle up to 16 columns. The initial Xe-LPG+ exact-tail decomposition for 47 columns was `16 + 16 + 8 + 7`. The last two tiles used the generic exact-tail kernel. [Xe2 launcher](third_party/llama.cpp/xmx-kernel.original.inc#L773) · [Xe-LPG+ dispatcher](include/xmx/kernel.hpp#L1092)

The original Arrow Lake-H exact-tail AOT builds with SLM and native 2D loads disabled each emitted 252 XMX kernel entries across the six formats and the two row layouts. Unfused code accounted for 120 regular entries and 24 tail entries. GLU accounted for 108 entries. The 32- and 64-column host tiers reuse the same tiled device kernel. Each generic unfused tail entry embedded widths 1–8 and 16. Together those 12 entries occupied 2.20 MiB with gathers and 3.74 MiB with row loads, about 23% of each build's XMX kernel `.text`. The other 12 tail entries each embedded widths 1 and 16 for requests with a one-column remainder.

For regular-row Q8_0, the generic tail entry occupies 118.25 KiB with gathers and 327.19 KiB with row loads. A prototype with an exact `8 + 7` tail emits 58.50 and 127.94 KiB respectively. A prototype using bounded eight-column work-groups for both tail tiles emits 44.50 and 90.56 KiB. Rounding the whole grid to 16-column work-groups reuses the existing tiled kernel. At 33 columns this last option computes 15 discarded columns, and at 47 columns it computes one.

These Q8_0 follow-up measurements used 1,024 weight rows, K=4,096, varying scales, and seven alternating pairs of 128 launches. Both sides used the same AOT test binary. Every batch passed the CPU reference check.

| Tail policy | Load path | Columns | Exact-tail baseline (µs) | Prototype median (µs) | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Round to 16 | Gathers | 47 | 372.1 | 366.1 | 3/7 |
| Round to 16 | Row loads | 47 | 418.6 | 448.4 | 4/7 |
| Round to 16 | Row loads | 33 | 379.7 | 403.1 | 3/7 |
| Bound two eight-column tiles | Row loads | 47 | 421.9 | 428.7 | 3/7 |
| Specialize for exact `8 + 7` | Gathers | 47 | 384.8 | 350.3 | 6/7 |

The exact `8 + 7` prototype also improved six of seven pairs in the initial 47-column gather comparison. At 63 columns with gathers it improved four of seven pairs, and the initial 47-column row-load median rose from 389.7 to 411.7 µs. VTune could not collect GPU hardware counters with the available permissions, so these runs measure latency and code size only.

#### Direct Comparison at 47 Columns

A follow-up benchmark tested three Q8_0 schedules at exactly 47 columns, 1,024 weight rows, and K=4,096. Each schedule used one submission and the same test binary for a given load path. The test ran on AC power and used all six possible schedule orders, with 128 launches per schedule in each round. The mixed schedule reused the spill-free 32-column implementation described below. The two rounded schedules computed 48 columns and stored 47.

| Work-groups per row tile | Row loads (µs) | Gathers (µs) |
| --- | ---: | ---: |
| `16 + 16 + 8 + 7` | 411.6 | 396.9 |
| `16 + 16 + 16` | 403.5 | 375.9 |
| `32 + 16` | 646.0 | 538.3 |

The mixed schedule was slower in all six rounds in both builds despite zero scratch. Rounding to three 16-column groups beat the exact tail in four of six row-load rounds and five of six gather rounds. The median per-round reductions were 1.0% and 4.1%, respectively. All 36 timing batches passed the CPU reference and output guard checks.

#### Work-Group Count at 40 and 41 Columns

The unfused software launcher now uses `16 + 16 + 8` at 40 columns and `16 + 16 + 16` at 41. Both use three work-groups per row tile in one submission. The last work-group at 41 columns computes 16 columns and stores nine. For Q8_0 this raises the total DPAS count per K=256 and row tile from 656 to 768 compared with `16 + 16 + 8 + 1`. The last tile reuses one set of weights across all nine valid columns. The new policy reuses the existing bounded 16-column kernel for remainders of nine to fifteen. Exact one-column tails use their existing kernel, and the generic exact-tail kernel now contains widths 2–8 and 16.

The 41-column Q8_0 benchmark ran at 1,024 weight rows, K=4,096, and varying scales on AC power. It used seven alternating pairs of 128 launches per variant. With row loads, the median fell from 448.2 to 396.1 µs and the rounded schedule was faster in five of seven pairs. With gathers, the median fell from 359.9 to 343.5 µs and the rounded schedule was faster in all seven pairs. All 28 batches passed the CPU reference and output guard checks.

The integrated AOT builds passed all six CTest checks each. Both unfused GPU suites passed 234 CPU-reference and output guard checks at K=1,024 with varying scales, including 40 and 41 columns across all six formats. The tiled Q8_0 device code is byte-identical between the rounded benchmark and the integrated build for each load path.

With the revised policy, the regular-row Q8_0 generic exact-tail kernel still occupies 111.12 KiB with gathers and 295.69 KiB with row loads. It handles remainders of two through eight. Remainders of nine through fifteen use the bounded full-tile kernel.

The same Q8_0 timing method at 15 columns compared the production AOT builds before and after the change. The original `8 + 7` split took 315.5 µs with row loads and 268.4 µs with gathers. The single 16-column work-group took 390.0 and 324.7 µs, respectively. The original split won six of seven paired batches in both builds. All 28 batches passed the CPU-reference and output guard checks.

#### Q8_0 Sweep from 1 to 64 Columns

The Arrow Lake-H sweep compared the current unfused dispatcher, the earlier exact-tail policy, and a policy that rounds to 16-column tiles for requests above eight columns. Equal tile decompositions shared a measurement. Every width ran with 1,024 and 4,096 weight rows, K=4,096, varying scales, and both software load paths. Each distinct schedule used seven alternating rounds of 32 launches on AC power. All 3,248 batches passed the CPU-reference and output guard checks. [Summary CSV](tests/history/benchmarks/q8-column-sweep/column-sweep.csv) · [Raw samples](tests/history/benchmarks/q8-column-sweep/column-sweep-samples.csv) · [Chart](tests/history/benchmarks/q8-column-sweep/column-sweep.svg)

For whole requests of 9–15 columns at 1,024 weight rows, two exact tiles won at least six of seven rounds for every width and both load paths. Their paired median latency was 13–35% lower than the bounded 16-column tile. At 4,096 rows its direction reversed for every width. At 15 columns the two builds allocate 256 GRFs per work-item and use zero scratch. The change in relative throughput with row count is consistent with more independent work-groups benefiting smaller launches, while the wider tile reuses its weight data across all columns.

| Weight rows | Load path | `8 + 7` at 15 columns (µs) | `16` at 15 columns (µs) |
| ---: | --- | ---: | ---: |
| 1,024 | Row loads | 306.5 | 434.5 |
| 1,024 | Gathers | 255.4 | 315.6 |
| 4,096 | Row loads | 841.1 | 550.1 |
| 4,096 | Gathers | 721.3 | 590.7 |

At 41 columns and 1,024 rows, the exact split won six of seven row-load rounds and the padded tile won six of seven gather rounds. At 4,096 rows the padded tile won six of seven row-load rounds and all gather rounds. Over the complete 4,096-row sweep, the current policy had lower paired median latency than the exact policy for every width where their decompositions differed with row loads. The median paired difference across those widths was 14.7% relative to the current policy. The corresponding difference for gathers was 7.9%. The sweep samples two row counts and one K value. It gives the measured endpoints of the narrow-request crossover.

#### Q8_0 Tail Cache Timing

The controlled timing study used 4,096 weight rows and K=4,096. Every batch submitted widths 34–40 eight times each, with the same weights and activation and output addresses. Each width used two full work-groups and one exact tail per row tile. The twelve rounds rotated the four combinations of kernel and order. GPU events supplied execution time. Both kernels passed the CPU reference and output guard checks at every width.

| Load path and run | Generic kernel, mixed vs grouped | Fixed-tail kernels, mixed vs grouped |
| --- | ---: | ---: |
| Row loads | +6.8% | +0.5% |
| Row loads, repeat | −0.9% | +8.1% |
| Gathers | +4.0% | +3.3% |
| Row loads, 2026-11-19 | −0.9% | +2.3% |
| Gathers, 2026-11-19 | +1.7% | −0.4% |

The table reports paired median latency changes. The initial row-load ordering penalty for the generic kernel did not reproduce in either repeat. The fixed-tail entries occupy 80–88 KiB with row loads and 34–43 KiB with gathers, versus about 296 and 111 KiB for the generic entry. These sizes describe individual entries. The test measures sequence ordering and device latency. Hardware instruction-cache misses remain unmeasured. [Program](tests/history/benchmarks/q8-tail-cache/benchmark.cpp) · [Initial row loads](tests/history/benchmarks/q8-tail-cache/timing-row.jsonl) · [First row-load repeat](tests/history/benchmarks/q8-tail-cache/timing-row-repeat.jsonl) · [Initial gathers](tests/history/benchmarks/q8-tail-cache/timing-gather.jsonl) · [November 19 row loads](tests/history/benchmarks/q8-tail-cache/timing-row-2026-11-19.jsonl) · [November 19 gathers](tests/history/benchmarks/q8-tail-cache/timing-gather-2026-11-19.jsonl)

Locally built Intel Metrics Discovery 1.16.199 and Metrics Library 1.0.249 enabled Level Zero to enumerate instruction-cache hit and miss groups `Ext500`–`Ext503` on the Arrow Lake-H GPU. With Xe observation access enabled, the direct Metrics Discovery stream still returned `CC_ERROR_NOT_SUPPORTED`. That version's Xe backend has no Arrow Lake entry in its OA report-format selector. [Metrics Discovery Xe backend](https://github.com/intel/metrics-discovery/blob/master/instrumentation/metrics_discovery/linux/md_driver_ifc_linux_xe.cpp)

### Wider Work-Groups and DPAS Repeat Count

A regular-row Q8_0 prototype processed 32 columns per unfused work-group and kept one K reduction per work-item. At 64 columns it used two work-groups per row tile instead of four. With LSC gathers, each work-group issued 16 quantized-weight gathers and one scale gather per K=256 at both widths. The 32-column version therefore issued half as many weight gathers for the same request. Direct widening used 2,688 scratch bytes per work-item with gathers and 3,296 with row loads. A revised schedule loaded two consecutive weight groups and reused them across four eight-column blocks, holding one activation pair at a time. This version used zero scratch in both builds.

A second prototype kept the 16-column geometry and used `dpas.8x2` for two neighboring activation columns at a time. The execution size remained eight. This reduced the static DPAS instruction count per K=256 from 256 to 128 while computing the same dot products. Scratch stayed at zero. With gathers, `mov` rose from 365 to 580 and `sync.nop` fell from 183 to 128. With row loads, `mov` rose from 1,542 to 1,761 and `sync.nop` fell from 163 to 134.

The Arrow Lake-H measurements below use 64 columns, K=4,096, varying scales, and seven alternating pairs of 128 launches. The harness checked AC power and the kernel journal for each batch. Every batch passed the CPU reference check. Each row compares its own baseline and prototype.

| Prototype | Load path | Weight rows | 16-column baseline (µs) | Prototype (µs) |
| --- | --- | ---: | ---: | ---: |
| Two 32-column work-groups | Row loads | 1,024 | 443.5 | 670.1 |
| Two 32-column work-groups | Gathers | 1,024 | 394.2 | 560.1 |
| Two 32-column work-groups | Row loads | 4,096 | 1,524.6 | 1,765.3 |
| DPAS repeat count 2 | Row loads | 1,024 | 445.8 | 463.0 |
| DPAS repeat count 2 | Gathers | 1,024 | 364.9 | 444.7 |

### CPU and XMX Microbenchmark

Q8_0 SwiGLU was timed on the Core Ultra 7 265H and its Arrow Lake-H GPU with 1,024 weight rows, K=4,096, and varying scales. Each result is the median of seven batches. The CPU reference uses scalar double-precision dot products. The optimized CPU comparison uses oneMKL FP32 GEMV or GEMM with pre-dequantized weights and activations. The table reports the best median among the tested 1, 4, 8, and 16 thread settings. XMX used SLM off and the quantized representation. Timing excludes fixture setup, dequantization, and GPU transfers. XMX timing includes submission and synchronization.

| Columns | Scalar CPU reference (ms) | Best tested oneMKL (ms) | XMX row loads (ms) | XMX LSC gathers (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5.934 | 0.963, 1 thread | 0.806 | 0.276 |
| 64 | 254.143 | 4.817, 16 threads | 1.250 | 0.737 |

The test compares the standalone kernel with a CPU baseline for the same output. A quantized CPU kernel and model throughput are separate measurements.

### Latest Arrow Lake AOT Validation

The final GLU build passed all 1,014 CPU-reference comparisons across five suites, including 816 fused cases. Each suite covered all six formats and used `--vary-scales`. Each fused suite also covered 65 columns at 17 rows for all six formats. The kernel journal reported no resets.

| Load path | SLM | GPU suite options beyond `--suite --vary-scales` | Passed | Max absolute error |
| --- | --- | --- | ---: | ---: |
| LSC gathers | Off | `--k 1024` | 198/198 | 9.54e-07 |
| LSC gathers | Off | `--k 1024 --short-rows --glu swiglu` | 204/204 | 1.53e-05 |
| Rows | Off | `--k 256 --glu geglu` | 204/204 | 1.91e-06 |
| Rows | On | `--k 2304 --glu swiglu` | 204/204 | 1.53e-05 |
| LSC gathers | On | `--k 1280 --short-rows --glu geglu` | 204/204 | 3.81e-06 |

CTest passed 6/6 checks in each of the four Arrow Lake AOT configurations, and the CPU-only reference suite passed separately. The gather build without SLM runs `arrow_lake_isa`, which checks the unfused instruction limits, the fused activation-load counts, and the fused scratch budget. The other three builds run `arrow_lake_fused_scratch`. Each build emitted 108 fused software kernels with zero scratch. [CTest configuration](CMakeLists.txt)

## Work Completed

The original `xmx_term` used a 512-byte INT8 B operand, which makes oneAPI deduce execution size 16. The current kernel gathers each eight-row half from the VNNI layout into a 256-byte operand. Each half runs `xmx::dpas<8, 1, int>` for one activation column. The two results form the 16-row weight term, which the format-specific epilogue scales and corrects. [DPAS implementation](include/xmx/kernel.hpp)

The kernel unpacks weights to signed INT8 for the dot products. DPAS returns INT32. The epilogue converts the stored FP16 scales to FP32, and output accumulation uses FP32. At depth eight, BF16 DPAS covers 16 K elements per instruction and INT8 DPAS covers 32. A BF16 kernel would use different operands and packing. The installed oneAPI API requires execution size 16 for TF32. [DPAS operand formats](references/xe-lpg-plus/visa/DPAS.md) · [Installed oneAPI header](references/xe-lpg-plus/installed-oneapi-2026.2/esimd/xmx/dpas.hpp)

With `XMX_ENABLE_SLM=ON`, work-items store their K partial sums in shared local memory. After a work-group barrier, one work-item combines the sums and applies GLU when enabled. Without SLM, one work-item runs the full K reduction. [Latest SLM test coverage](#latest-arrow-lake-aot-validation)

## Verification Evidence

The following commands run the CPU-only and default row-load AOT checks. The [gather build instructions](#transposed-load-follow-up) cover the other load mode. Repeat the relevant CMake configuration in a separate build directory with `-DXMX_ENABLE_SLM=ON` to check the optional reduction.

```bash
./build.sh --cpu-only --build-dir build-cpu
./build.sh --aot 12.74.4 --build-dir build-aot
ctest --test-dir build-aot --output-on-failure
python3 scripts/check_provenance.py
bash -n build.sh run-tests.sh
```

The immutable upstream snapshot is `third_party/llama.cpp/xmx-kernel.original.inc`. The provenance check reports `Upstream snapshot: verified` and `Working kernel: modified for development`. The `--strict` option requires an unchanged working kernel and therefore exits nonzero for the port.

## Historical Bring-up

The bring-up ran with elevated host execution because the sandbox lacked `/dev/dri`. The DPAS probe recorded these results before the full-kernel tests.

```text
DPAS N=8, accumulator: passed=true
DPAS N=16, accumulator: passed=false, first_mismatch=32, actual=7
```

The 16-wide probe produced correct values for the first 32 elements and left the second half at the initialized accumulator value. The random execution-size-8 probe and an independent split-half gather probe passed with nonuniform signed INT8 operands.

The first full Q8_0 test at 16 rows, one column, and K=1,024 returned `UR_RESULT_ERROR_DEVICE_LOST`. The driver recorded a CCS reset, a timed-out job, and a device coredump. A rerun after recovery with SLM disabled also reset the device. An isolated transposed `load_2d` probe later reported device loss. [Bring-up logs](tests/history/xe-lpg-port-attempt-2026-11-18.md)

During the row-load bring-up, the test first completed with `relative_l2=1.87486` and `max_abs=6.21411`. The verified fix used the full activation pitch in dword units, retained the strided VNNI weight layout, and placed activation rows contiguously within each K group. The smallest Q8_0 case then matched exactly.

The original 45-case suite for Q8_0, Q4_K, and Q6_K passed at K values 1,024, 2,048, 4,096, 8,192, 16,384, and 32,768. Those 270 exact comparisons and the [latest AOT validation](#latest-arrow-lake-aot-validation) were successive milestones.
