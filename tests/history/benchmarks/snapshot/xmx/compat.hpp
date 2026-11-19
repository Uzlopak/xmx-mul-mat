// SPDX-License-Identifier: MIT
#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace dpct { using queue_ptr = sycl::queue *; }
inline constexpr int QK_K = 256;
inline constexpr int QK8_0 = 32;
inline constexpr int QK8_1 = 32;
inline constexpr int GGML_GLU_OP_GEGLU = 1;
inline constexpr int GGML_GLU_OP_SWIGLU = 2;
#define GGML_SYCL_XMX_GLU_MAX_COLS 16
#define GGML_SYCL_MMVQ_HAS_XMX
#ifndef XMX_ENABLE_SLM
#define XMX_ENABLE_SLM 0
#endif
#ifndef XMX_ENABLE_TRANSPOSED_2D_LOADS
#define XMX_ENABLE_TRANSPOSED_2D_LOADS 0
#endif
#ifndef XMX_ENABLE_GATHER_TRANSPOSE
#define XMX_ENABLE_GATHER_TRANSPOSE 0
#endif

namespace xmx {
[[noreturn]] inline void fail(const char * format, ...) {
    char message[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    throw std::invalid_argument(message);
}
}
#define GGML_ASSERT(condition) do { if (!(condition)) xmx::fail("Assertion failed: %s", #condition); } while (0)
#define GGML_ABORT(...) xmx::fail(__VA_ARGS__)
