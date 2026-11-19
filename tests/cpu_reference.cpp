// SPDX-License-Identifier: MIT
#include "xmx/fixtures.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static uint16_t read_half(const std::vector<uint8_t> & bytes, size_t offset) {
    uint16_t bits;
    std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
    return bits;
}

static void require_varying_scale(const std::vector<uint8_t> & bytes, size_t offset,
                                  size_t block_stride, size_t row_stride, const char * message) {
    const auto first = read_half(bytes, offset);
    const auto next_block = read_half(bytes, offset + block_stride);
    const auto next_row = read_half(bytes, offset + row_stride);
    require(first != next_block && first != next_row && next_block != next_row, message);
}

static void check_varying_scales(xmx::Type type) {
    const auto f = xmx::make_fixture_with_varying_scales(type, 17, 17, 1024);
    const size_t blocks_per_row = f.k / 256;
    const size_t blocks = f.rows * blocks_per_row;
    if (type == xmx::Type::q8_0) {
        require_varying_scale(f.weights, size_t(f.rows) * f.k, 2, 2 * (f.k / 32), "Q8_0 d varies by row and block");
    } else if (type == xmx::Type::q3_k) {
        require_varying_scale(f.weights, blocks * 108, 2, 2 * blocks_per_row, "Q3_K d varies by row and block");
    } else if (type == xmx::Type::q6_k) {
        require_varying_scale(f.weights, blocks * 208, 2, 2 * blocks_per_row, "Q6_K d varies by row and block");
    } else {
        const size_t block_bytes = type == xmx::Type::q2_k ? 80 : (type == xmx::Type::q5_k ? 172 : 140);
        require_varying_scale(f.weights, blocks * block_bytes, 4, 4 * blocks_per_row, "d varies by row and block");
        require_varying_scale(f.weights, blocks * block_bytes + 2, 4, 4 * blocks_per_row, "dmin varies by row and block");
    }
    require_varying_scale(f.activations, f.k, 4, f.activation_stride, "activation d varies by column and block");
    const auto constant = xmx::make_fixture(type, f.rows, f.columns, f.k);
    require(!xmx::compare(constant.expected, f.expected).passed, "must detect reuse of constant scales");
}

int main() {
    try {
        require(xmx::half_to_float(0x3c00) == 1, "half 1");
        require(xmx::half_to_float(0xc000) == -2, "half -2");
        require(xmx::half_to_float(1) == std::ldexp(1.0f, -24), "half subnormal");
        for (unsigned i = 0; i < 65536; ++i) {
            if ((i & 0x7c00) == 0x7c00 && (i & 1023)) continue;
            require(xmx::float_to_half(xmx::half_to_float(uint16_t(i))) == i, "half roundtrip");
        }
        int cases = 0;
        for (auto type : {xmx::Type::q2_k, xmx::Type::q3_k, xmx::Type::q4_k, xmx::Type::q5_k,
                          xmx::Type::q6_k, xmx::Type::q8_0}) {
            for (int rows : {16, 17, 128}) for (int cols : {1, 2, 4, 8, 16}) {
                const auto f = xmx::make_fixture(type, rows, cols, 1024);
                require(f.expected.size() == size_t(rows * cols), "reference shape");
                require(f.activation_stride % 64 == 0, "activation alignment");
                require(xmx::compare(f.expected, f.expected).passed, "reference comparison");
                auto wrong = f.expected;
                wrong[0] += 100;
                require(!xmx::compare(wrong, f.expected).passed, "must detect wrong output");
                wrong[0] = std::numeric_limits<float>::quiet_NaN();
                require(!xmx::compare(wrong, f.expected).passed, "must reject NaN");
                ++cases;
            }
            const auto f = xmx::make_fixture(type, 16, 1, 1024);
            if (type == xmx::Type::q8_0) {
                const auto q = static_cast<int8_t>(f.weights[0]);
                require(f.decoded_weights[0] == float(q) / 64, "Q8 first weight");
            } else if (type == xmx::Type::q2_k) {
                const size_t blocks = 16 * 1024 / 256;
                for (int i : {0, 16}) {
                    const uint8_t code = f.weights[blocks * 64 + i / 16];
                    const int q = f.weights[i] & 3;
                    const float expected = float((code & 15) * q) / 1024 - float(code >> 4) / 2048;
                    require(f.decoded_weights[i] == expected, "Q2 half-block scale and minimum");
                }
            } else if (type == xmx::Type::q3_k) {
                const size_t blocks = 16 * 1024 / 256;
                const size_t codes = blocks * 96;
                const int scale = (f.weights[codes] & 15) + 16 * (f.weights[codes + 8] & 3) - 32;
                const int quant = (f.weights[0] & 3) + 4 * (f.weights[blocks * 64] & 1) - 4;
                require(f.decoded_weights[0] == float(scale * quant) / 1024, "Q3 first weight");
            } else if (type == xmx::Type::q4_k || type == xmx::Type::q5_k) {
                const size_t blocks = 16 * 1024 / 256;
                const bool q5 = type == xmx::Type::q5_k;
                const size_t codes = blocks * (q5 ? 160 : 128);
                const int scale = f.weights[codes] & 63;
                const int minimum = f.weights[codes + 4] & 63;
                const int high = q5 ? 16 * (f.weights[blocks * 128] & 1) : 0;
                const float expected = float(scale * ((f.weights[0] & 15) + high)) / 1024 - float(minimum) / 2048;
                require(f.decoded_weights[0] == expected, "Q4 or Q5 first weight");
            } else {
                const size_t blocks = 16 * 1024 / 256;
                const int q = (f.weights[0] & 15) | ((f.weights[blocks * 128] & 3) << 4);
                const int scale = static_cast<int8_t>(f.weights[blocks * 192]);
                require(f.decoded_weights[0] == float((q - 32) * scale) / 4096, "Q6 first weight");
            }
            check_varying_scales(type);
        }
        bool rejected = false;
        try { xmx::make_fixture(xmx::Type::q8_0, 1, 1, 300); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "must reject unsupported K");
        const auto up = xmx::make_fixture(xmx::Type::q3_k, 1, 65, 256);
        auto gate = xmx::make_fixture(xmx::Type::q3_k, 1, 1, 256, 43);
        const auto swiglu = xmx::glu_reference(up, gate, xmx::Glu::swiglu);
        const auto geglu = xmx::glu_reference(up, gate, xmx::Glu::geglu);
        require(swiglu.size() == up.expected.size(), "GLU chunked output shape");
        require(swiglu != geglu, "GLU activation choice affects the output");
        std::fill(gate.decoded_weights.begin(), gate.decoded_weights.end(), 0.0f);
        require(xmx::glu_reference(up, gate, xmx::Glu::swiglu) == std::vector<float>(up.columns, 0.0f),
                "zero gate must zero every output column");
        std::cout << "PASS: half conversions, quantized layouts, " << cases
                  << " reference fixtures, varying scales, error checks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
