# Xe-LPG+ Port Validation Attempt

Target: Arrow Lake-H / Arc Graphics, PCI ID `8086:7d51`, IP `12.74.4`.
The commands ran with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.

## Results

The smallest execution-size-8 DPAS probe passed:

```text
{"dpas_n":8,"accumulator":true,"expected":39,"first_mismatch":-1,"passed":true}
```

The native execution-size-16 probe did not work on this target. The first 32 results were correct, while the second half retained the initialized accumulator value:

```text
{"dpas_n":16,"accumulator":true,"expected":39,"first_mismatch":32,"actual":7,"passed":false}
```

The small full-kernel test then failed with device loss:

```text
Device: Intel(R) Arc(TM) Graphics
Launching q8_0 rows=16 cols=1 K=1024
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

## Kernel Diagnostics

The corresponding kernel log reported:

```text
xe 0000:00:02.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs
xe 0000:00:02.0: [drm] Tile0: GT0: Timedout job ... in xmx_kernel_test
xe 0000:00:02.0: [drm] Xe device coredump has been created
xe 0000:00:02.0: [drm] exec queue reset detected
```

No further GPU variants were launched after the reset. The full-kernel port remains unverified; this record does not establish correctness or identify whether the reset is caused by the kernel, its memory layout, or the driver.

## Staged Probe After Static Load Change

With the current build (`XMX_ENABLE_SLM=OFF`, `XMX_ENABLE_TRANSPOSED_2D_LOADS=OFF`), the authorized staged probe produced:

```text
n=8: passed=true
n=16: passed=false, first_mismatch=32, actual=7
```

The n=16 gate remains failed, so the Q8_0 full-kernel test was intentionally not launched for this configuration.

## Activation Load Isolation

The default path was changed to use row-by-row block loads for Q8 activation bytes and half2 scales as well as for weights/scales. Device-only IR contains no `__esimd_lsc_load2d` markers. CPU/CTest/provenance checks remained green. This was not GPU-run after the previous CCS reset.

Static review then corrected the new helper's pitch units: its `uint32_t*` arithmetic now receives uint32-element pitches rather than byte pitches. The corrected version builds and passes CTest/provenance checks; it has not been GPU-run.

## Numerical Debugging After Pitch Fix

The corrected variant was run on the GPU and completed without reset, but Q8_0 still mismatched (`relative_l2=1.87486`, `max_abs=6.21411`). Static review then found that the original strided `select<W, XMX_ROWS>` already formed VNNI order, while the new activation helper used the wrong stride and interleaved horizontal rows. The code now retains the existing weight layout and uses contiguous activation placement. This correction has not been GPU-run because the later hardware `load_2d` probe caused device loss.

## No-SLM Follow-up

After the device recovered, the default `XMX_ENABLE_SLM=OFF` build was run with the same small case. It failed identically with `UR_RESULT_ERROR_DEVICE_LOST`. The kernel log again reported a CCS engine reset and a timed-out job in `xmx_kernel_test`. This shows that SLM/barrier traffic is not the sole cause. No further GPU work was attempted.

## Hardware Transposed-Load Safety Result

A standalone synthetic row-load comparison invoked the hardware transposed `load_2d` operation and terminated with `UR_RESULT_ERROR_DEVICE_LOST`. No further GPU execution is permitted in this attempt. The default row-load path, with the existing strided-select VNNI construction for weights and contiguous activation rows, remains mandatory; debugging now continues only through host/static analysis.

Before that safety event, a random nonuniform s8×s8 N=8 probe passed all outputs, and an independent N=16 two-half VNNI gather probe passed all outputs. Those probes validate the primitive DPAS and split-gather contracts, but not the complete Q8 quantization kernel.

## Successful Xe-LPG+ Validation After Recovery

The corrected default configuration subsequently passed:

- Q8_0 staged cases including rows 16/17/128, columns 1/2/4/8/16, and K=1024/2048/4096.
- Q4_K and Q6_K representative tail/multi-column cases.
- The complete configured GPU suite: 45 cases across Q8_0, Q4_K, and Q6_K, rows 16/17/128, columns 1/2/4/8/16, K=1024.

All cases reported `relative_l2=0`, `max_abs=0`, `passed=true`; no reset occurred. Multi-column correctness required using the full activation stride and group-major paired-load layout.

The complete 45-case suite was additionally run at K=2048, 4096, 8192, 16384, and 32768. All five suites completed with 45/45 `passed=true`, `relative_l2=0`, and `max_abs=0`; no device loss or reset occurred. The native n=16 primitive probe remains failed on its second half, so production uses split execution-size-8 DPAS terms.

## Partial-Tile Bounds Fix

Static review identified an OOB risk for non-power-of-two `ncv` values rounded to a larger `NC` tile. The software activation row loaders now clamp invalid source rows to the last valid row whenever the native 2D path is disabled; output stores continue to honor `ncv`. Build, CTest, provenance, and the native 2D geometry test pass after this change. No new GPU run was needed because the fix only changes invalid-column reads, and GPU execution remains subject to the documented staged policy.

The `native_2d_geometry` CTest now explicitly covers `ncv=9`/`NC=16` source-row mapping: rows 8, 9, and 15 map to 8, while a valid row 15 in a full 16-row tile remains 15.
