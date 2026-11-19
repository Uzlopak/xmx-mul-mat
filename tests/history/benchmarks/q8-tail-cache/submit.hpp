// SPDX-License-Identifier: MIT
#pragma once
#include "xmx/kernel.hpp"

namespace ggml_sycl_esimd {

template <typename T, bool FUSED, bool SINGLE_COLUMN_TAIL, int TAIL = 0>
static sycl::event xmx_cache_submit(const void *    vx,
                                               const void *    vg,
                                               const char *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       ncv,
                                               const int       stride_y_bytes,
                                               const int       ldd,
                                               const int       glu_op,
                                               dpct::queue_ptr stream) {
    const size_t row_tiles = (nrows + XMX_ROWS - 1) / XMX_ROWS;
    const size_t column_tiles = xmx_column_tile_count(ncv);
    constexpr auto properties = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size_automatic };
    return stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(row_tiles, column_tiles), sycl::range<2>(1, 1)),
                         properties, [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
                             const int column_tile = it.get_group(1);
                             const auto tile = SINGLE_COLUMN_TAIL
                                 ? xmx_column_tile{column_tile * XMX_COLUMN_TILE,
                                                   column_tile == ncv / XMX_COLUMN_TILE ? 1 : XMX_COLUMN_TILE}
                                 : xmx_column_tile_at(ncv, column_tile);
                             const char * y = vy + size_t(tile.first) * stride_y_bytes;
                             float * d = dst + size_t(tile.first) * ldd;
                             const int row_tile = it.get_group(0);
                             auto multiply = [&](auto width) [[intel::sycl_explicit_simd]] {
                                 constexpr int columns = decltype(width)::value;
                                 xmx_mul_mat<T, columns, 1, FUSED, false>(
                                     vx, vg, y, d, ncols, nrows, columns, stride_y_bytes, ldd, glu_op, row_tile, 0, 1);
                             };
                             if constexpr (SINGLE_COLUMN_TAIL) {
                                 if (tile.width == 1) {
                                     multiply(std::integral_constant<int, 1>{});
                                 } else {
                                     multiply(std::integral_constant<int, XMX_COLUMN_TILE>{});
                                 }
                             } else if constexpr (TAIL != 0) {
                                 if (tile.width == XMX_COLUMN_TILE) {
                                     multiply(std::integral_constant<int, XMX_COLUMN_TILE>{});
                                 } else {
                                     multiply(std::integral_constant<int, TAIL>{});
                                 }
                             } else {
                                 switch (tile.width) {
                                     case XMX_COLUMN_TILE: multiply(std::integral_constant<int, XMX_COLUMN_TILE>{}); break;
                                     case 8: multiply(std::integral_constant<int, 8>{}); break;
                                     case 7: multiply(std::integral_constant<int, 7>{}); break;
                                     case 6: multiply(std::integral_constant<int, 6>{}); break;
                                     case 5: multiply(std::integral_constant<int, 5>{}); break;
                                     case 4: multiply(std::integral_constant<int, 4>{}); break;
                                     case 3: multiply(std::integral_constant<int, 3>{}); break;
                                     case 2: multiply(std::integral_constant<int, 2>{}); break;
                                 }
                             }
                         });
    });
}

} // namespace ggml_sycl_esimd
