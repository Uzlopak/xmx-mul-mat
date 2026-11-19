// SPDX-License-Identifier: MIT
#include "xmx/kernel.hpp"
#include <iostream>

namespace kernels = ggml_sycl_esimd;

struct Geometry {
    int k;
    int stride;
    int rows = 16;
    int columns = 2;
    uintptr_t weights = 0x1000;
    uintptr_t activations = 0x2000;
    uintptr_t gate = 0x3000;
};

template <typename T, bool FUSED = false>
static bool expect(const char * name, bool expected, const Geometry & g) {
    const bool actual = kernels::xmx_native_2d_geometry<T, FUSED>(g.weights, g.gate, g.activations, g.k,
                                                                g.rows, g.columns, g.stride);
    if (actual != expected) {
        std::cerr << name << " expected " << std::boolalpha << expected << ", got " << actual << '\n';
    }
    return actual == expected;
}

int main() {
    using q4 = kernels::xmx_traits_q4_k<false>;
    using q6 = kernels::xmx_traits_q6_k<false>;
    using q8 = kernels::xmx_traits_q8_0<false>;
    using sycl::ext::oneapi::experimental::architecture;
    static_assert(kernels::xmx_supports_dpas8(architecture::intel_gpu_arl_h));
    static_assert(kernels::xmx_supports_dpas8(architecture::intel_gpu_acm_g10));
    static_assert(!kernels::xmx_supports_dpas8(architecture::intel_gpu_lnl_m));
    static_assert(!kernels::xmx_supports_dpas8(architecture::intel_gpu_pvc));
    static_assert(!kernels::xmx_supports_dpas8(architecture::unknown));
    constexpr int max_extent = 1 << 24;

    bool passed = expect<q8>("32-byte activation scale surface", false, {256, 288});
    const auto expect_row = [&](const char * name, int actual, int expected) {
        if (actual != expected) {
            std::cerr << name << " expected " << expected << ", got " << actual << '\n';
            return false;
        }
        return true;
    };
    // ncv=9 is rounded to NC=16. The second RC=8 group starts at row 8;
    // rows 9..15 must remain in-bounds for an exactly sized activation tile.
    passed &= expect_row("partial tile first valid row", kernels::xmx_activation_source_row(8, 9), 8);
    passed &= expect_row("partial tile first invalid row", kernels::xmx_activation_source_row(9, 9), 8);
    passed &= expect_row("partial tile final invalid row", kernels::xmx_activation_source_row(15, 9), 8);
    passed &= expect_row("full tile row unchanged", kernels::xmx_activation_source_row(15, 16), 15);
    passed &= expect_row("33 columns use three tiles", kernels::xmx_column_tile_count(33), 3);
    passed &= expect_row("33-column tail starts at column 32", kernels::xmx_column_tile_at(33, 2).first, 32);
    passed &= expect_row("33-column tail computes one column", kernels::xmx_column_tile_at(33, 2).width, 1);
    passed &= expect_row("40 columns use three tiles", kernels::xmx_column_tile_count(40), 3);
    passed &= expect_row("40-column tail computes eight columns", kernels::xmx_column_tile_at(40, 2).width, 8);
    passed &= expect_row("41 columns use three tiles", kernels::xmx_column_tile_count(41), 3);
    passed &= expect_row("41-column tail computes sixteen columns", kernels::xmx_column_tile_at(41, 2).width, 16);
    constexpr int tile_columns = kernels::XMX_COLUMN_TILE;
    for (int columns = 1; columns <= kernels::XMX_CHUNK; ++columns) {
        const int tiles = kernels::xmx_column_tile_count(columns);
        passed &= expect_row("column tile count is minimal", tiles, (columns + tile_columns - 1) / tile_columns);
        int next = 0;
        for (int i = 0; i < tiles; ++i) {
            const auto tile = kernels::xmx_column_tile_at(columns, i);
            passed &= expect_row("column tiles are contiguous", tile.first, next);
            const int valid_columns = std::min(tile_columns, columns - next);
            const int expected_width = valid_columns <= 8 ? valid_columns : tile_columns;
            passed &= expect_row("column tile computes the expected width", tile.width, expected_width);
            next += valid_columns;
        }
        passed &= expect_row("column tiles cover the requested columns", next, columns);
    }
    passed &= expect<q8>("64-byte activation scale surface", true, {512, 576});
    passed &= expect<q4>("512 K Q4 weights", true, {512, 576});
    passed &= expect<q6>("32-byte Q6 weight scale surface", false, {512, 576});
    passed &= expect<q6>("64-byte Q6 weight scale surface", true, {1024, 1152});
    passed &= expect<q6::short_row_traits>("2D activations with short weight rows", true, {512, 576});
    passed &= expect<q8>("padded activation stride", true, {512, 592});
    passed &= expect<q8>("stride shorter than quants and scales", false, {512, 560});
    passed &= expect<q8>("stride needs 16-byte alignment", false, {512, 580});
    passed &= expect<q8>("maximum stride", true, {512, max_extent});
    passed &= expect<q8>("stride beyond maximum", false, {512, max_extent + 16});
    passed &= expect<q8>("maximum height", true, {512, 576, max_extent});
    passed &= expect<q8>("height beyond maximum", false, {512, 576, max_extent + 1});
    passed &= expect<q8>("zero activation columns", false, {512, 576, 16, 0});
    passed &= expect<q8>("unmatched K block size", false, {513, 592});
    passed &= expect<q8>("missing weight base", false, {512, 576, 16, 2, 0});
    passed &= expect<q8>("misaligned weight base", false, {512, 576, 16, 2, 0x1004});
    passed &= expect<q8>("misaligned activation base", false, {512, 576, 16, 2, 0x1000, 0x2004});
    passed &= expect<q8, true>("misaligned fused gate base", false, {512, 576, 16, 2, 0x1000, 0x2000, 0x3004});
    return passed ? 0 : 1;
}
