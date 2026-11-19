# Isolated Arrow Lake-H XMX kernel test

Source: kernel-test.cpp. It includes the existing, unchanged `ggml/src/ggml-sycl/esimd.hpp` from integration `3bfc80f3da00ea10e8182ce707d4e3c22597a038` and calls xmx_mul_mat_launch directly, bypassing architecture dispatch only inside this separate executable. No llama.cpp source or active binaries changed.

Planned: 45 correctness cases across Q4_K/Q6_K/Q8_0, 16/17/128 weight rows, and 1/2/4/8/16 activation columns, K=1024. CPU quantizers prepare quantized inputs; a CPU dot product supplies the expected result. No performance result is claimed.

Compilation succeeded with oneAPI 2026.2. The first case (Q4_K, 16 rows, one column) failed with UR_RESULT_ERROR_DEVICE_LOST before producing output. xe logged a compute-engine reset and timed-out job in kernel-test at 12:07:55. Zero correctness cases completed. No speed benchmark or full-model test followed. The harness has not been validated on a known-supported GPU, so this does not by itself identify the failing instruction or prove that Arrow Lake-H cannot use an adapted kernel. Do not enable the architecture gate based on compiler success alone.

The earlier compile-only DPAS probes in ../without-dspine/arl-dpas-probe all compiled for 12.74.4, but were not executed. Runtime compatibility remains unresolved. Active build remains build-fp16-no-dspine, integration is clean. User stopped their chat themselves before this test; no user process was killed.

Evidence: build.log, runtime.log, correctness.jsonl (empty), runtime-result.json.
