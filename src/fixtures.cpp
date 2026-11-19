// SPDX-License-Identifier: MIT
#include "xmx/fixtures.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace xmx {
Type parse_type(const std::string & name) {
    if (name == "q2_k" || name == "q2_K") return Type::q2_k;
    if (name == "q3_k" || name == "q3_K") return Type::q3_k;
    if (name == "q4_k" || name == "q4_K") return Type::q4_k;
    if (name == "q5_k" || name == "q5_K") return Type::q5_k;
    if (name == "q6_k" || name == "q6_K") return Type::q6_k;
    if (name == "q8_0") return Type::q8_0;
    throw std::invalid_argument("Unknown type: " + name);
}
const char * type_name(Type type) {
    switch (type) {
        case Type::q2_k: return "q2_K";
        case Type::q3_k: return "q3_K";
        case Type::q4_k: return "q4_K";
        case Type::q5_k: return "q5_K";
        case Type::q6_k: return "q6_K";
        case Type::q8_0: return "q8_0";
    }
    throw std::invalid_argument("Invalid type");
}
float half_to_float(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    float value;
    if (exponent == 31) value = fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else if (exponent == 0) value = std::ldexp(float(fraction), -24);
    else value = std::ldexp(float(1024 + fraction), int(exponent) - 25);
    return bits & 0x8000 ? -value : value;
}
uint16_t float_to_half(float value) {
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    value = std::abs(value);
    if (std::isnan(value)) return sign | 0x7e00;
    if (std::isinf(value) || value >= 65520) return sign | 0x7c00;
    if (value < std::ldexp(1.0f, -14)) return sign | uint16_t(std::nearbyint(std::ldexp(double(value), 24)));
    int exponent;
    const double mantissa = std::frexp(double(value), &exponent);
    unsigned fraction = unsigned(std::nearbyint((mantissa * 2 - 1) * 1024));
    if (fraction == 1024) { fraction = 0; ++exponent; }
    return sign | uint16_t((exponent + 14) << 10) | uint16_t(fraction);
}

namespace {
constexpr int QK_K = 256;
constexpr int QK8_0 = 32;
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
struct block_q4_K { uint16_t d, dmin; uint8_t scales[12], qs[128]; };
struct block_q6_K { uint8_t ql[128], qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(block_q8_0) == 34);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q6_K) == 210);
#define GGML_RESTRICT
#define GGML_FP16_TO_FP32(value) half_to_float(value)
#include "../third_party/llama.cpp/dequant-reference.inc"
#undef GGML_FP16_TO_FP32
#undef GGML_RESTRICT

void store_half(std::vector<uint8_t> & data, size_t offset, float value) {
    const auto bits = float_to_half(value);
    std::memcpy(data.data() + offset, &bits, sizeof(bits));
}

double dot_product(const float * weights, const float * activations, int k) {
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += double(weights[i]) * activations[i];
    return sum;
}
}

static Fixture make_fixture(Type type, int rows, int columns, int k, unsigned seed, bool vary_scales) {
    if (rows < 1 || rows > 16384 || columns < 1 || columns > 128 || k < QK_K || k > 32768 || k % QK_K)
        throw std::invalid_argument("Require rows 1..16384, columns 1..128, K a multiple of 256 up to 32768");
    if (size_t(rows) * k > 16 * 1024 * 1024) throw std::invalid_argument("Fixture exceeds 16M weights");
    Fixture f{type, rows, columns, k, k + k / 32 * 4, {}, {}, {}, {}, {}};
    f.decoded_weights.resize(size_t(rows) * k);
    f.decoded_activations.resize(size_t(columns) * k);
    std::mt19937 rng(seed);
    auto byte = [&] { return uint8_t(rng() & 255); };
    // Powers of two keep the varied scales exactly representable in FP16.
    const auto scale = [vary_scales](float base, size_t row, size_t block) {
        return vary_scales ? std::ldexp(base, -int((2 * row + block) % 3)) : base;
    };
    const size_t blocks_per_row = k / QK_K;
    const size_t blocks = size_t(rows) * blocks_per_row;
    if (type == Type::q4_k) {
        std::vector<block_q4_K> raw(blocks);
        f.weights.resize(blocks * sizeof(block_q4_K));
        for (size_t b = 0; b < blocks; ++b) {
            const size_t row = b / blocks_per_row;
            const size_t block = b % blocks_per_row;
            raw[b].d = float_to_half(scale(1.0f / 1024, row, block));
            raw[b].dmin = float_to_half(scale(1.0f / 2048, 2 * row, 2 * block));
            for (auto & v : raw[b].qs) v = byte();
            for (auto & v : raw[b].scales) v = byte();
            std::memcpy(f.weights.data() + b * 128, raw[b].qs, 128);
            std::memcpy(f.weights.data() + blocks * 128 + b * 12, raw[b].scales, 12);
            std::memcpy(f.weights.data() + blocks * 140 + b * 4, &raw[b], 4);
        }
        dequantize_row_q4_K(raw.data(), f.decoded_weights.data(), int64_t(rows) * k);
    } else if (type == Type::q2_k || type == Type::q5_k) {
        const bool q2 = type == Type::q2_k;
        const size_t low_bytes = q2 ? 64 : 128;
        const size_t high_bytes = q2 ? 0 : 32;
        const size_t code_bytes = q2 ? 16 : 12;
        const size_t high_base = blocks * low_bytes;
        const size_t code_base = blocks * (low_bytes + high_bytes);
        const size_t dm_base = code_base + blocks * code_bytes;
        f.weights.resize(dm_base + blocks * 4);
        for (size_t b = 0; b < blocks; ++b) {
            const size_t row = b / blocks_per_row;
            const size_t block = b % blocks_per_row;
            uint8_t * low = f.weights.data() + b * low_bytes;
            uint8_t * high = f.weights.data() + high_base + b * high_bytes;
            uint8_t * codes = f.weights.data() + code_base + b * code_bytes;
            std::generate_n(low, low_bytes, byte);
            std::generate_n(high, high_bytes, byte);
            std::generate_n(codes, code_bytes, byte);
            const float d = scale(1.0f / 1024, row, block);
            const float dmin = scale(1.0f / 2048, 2 * row, 2 * block);
            store_half(f.weights, dm_base + b * 4, d);
            store_half(f.weights, dm_base + b * 4 + 2, dmin);
            float * decoded = f.decoded_weights.data() + b * QK_K;
            if (q2) {
                for (int group = 0; group < 16; ++group) {
                    const int shift = 2 * ((group / 2) % 4);
                    const int offset = 32 * (group / 8) + 16 * (group % 2);
                    const float weight_scale = d * (codes[group] & 15);
                    const float minimum = dmin * (codes[group] >> 4);
                    for (int i = 0; i < 16; ++i) {
                        const int quant = (low[offset + i] >> shift) & 3;
                        decoded[group * 16 + i] = weight_scale * quant - minimum;
                    }
                }
            } else {
                for (int group = 0; group < 8; ++group) {
                    uint8_t scale_code, min_code;
                    get_scale_min_k4(group, codes, &scale_code, &min_code);
                    const int shift = 4 * (group % 2);
                    const int offset = 32 * (group / 2);
                    for (int i = 0; i < 32; ++i) {
                        const int nibble = (low[offset + i] >> shift) & 15;
                        const int high_bit = (high[i] >> group) & 1;
                        decoded[group * 32 + i] = d * scale_code * (nibble + 16 * high_bit) - dmin * min_code;
                    }
                }
            }
        }
    } else if (type == Type::q3_k) {
        const size_t scales_base = blocks * 96;
        const size_t d_base = blocks * 108;
        f.weights.resize(blocks * 110);
        for (size_t b = 0; b < blocks; ++b) {
            uint8_t * quants = f.weights.data() + b * 64;
            uint8_t * high_bits = f.weights.data() + blocks * 64 + b * 32;
            uint8_t * codes = f.weights.data() + scales_base + b * 12;
            const float d = scale(1.0f / 1024, b / blocks_per_row, b % blocks_per_row);
            store_half(f.weights, d_base + b * 2, d);
            for (int group = 0; group < 16; ++group) {
                const int scale_code = byte() & 63;
                codes[group % 8] |= (scale_code & 15) << (4 * (group / 8));
                codes[8 + group % 4] |= (scale_code >> 4) << (2 * (group / 4));
                for (int i = 0; i < 16; ++i) {
                    const int k_index = 16 * group + i;
                    const int code = byte() & 7;
                    quants[32 * (k_index / 128) + k_index % 32] |= (code & 3) << (2 * ((k_index / 32) % 4));
                    high_bits[k_index % 32] |= (code >> 2) << (k_index / 32);
                    f.decoded_weights[b * QK_K + k_index] = d * (scale_code - 32) * (code - 4);
                }
            }
        }
    } else if (type == Type::q6_k) {
        std::vector<block_q6_K> raw(blocks);
        f.weights.resize(blocks * sizeof(block_q6_K));
        for (size_t b = 0; b < blocks; ++b) {
            raw[b].d = float_to_half(scale(1.0f / 4096, b / blocks_per_row, b % blocks_per_row));
            for (auto & v : raw[b].ql) v = byte();
            for (auto & v : raw[b].qh) v = byte();
            for (auto & v : raw[b].scales) v = int8_t(int(rng() % 63) - 31);
            std::memcpy(f.weights.data() + b * 128, raw[b].ql, 128);
            std::memcpy(f.weights.data() + blocks * 128 + b * 64, raw[b].qh, 64);
            std::memcpy(f.weights.data() + blocks * 192 + b * 16, raw[b].scales, 16);
            std::memcpy(f.weights.data() + blocks * 208 + b * 2, &raw[b].d, 2);
        }
        dequantize_row_q6_K(raw.data(), f.decoded_weights.data(), int64_t(rows) * k);
    } else {
        const size_t n = size_t(rows) * k / 32;
        const size_t q8_blocks_per_row = k / QK8_0;
        std::vector<block_q8_0> raw(n);
        f.weights.resize(n * sizeof(block_q8_0));
        for (size_t b = 0; b < n; ++b) {
            raw[b].d = float_to_half(scale(1.0f / 64, b / q8_blocks_per_row, b % q8_blocks_per_row));
            for (auto & v : raw[b].qs) v = int8_t(int(rng() % 63) - 31);
            std::memcpy(f.weights.data() + b * 32, raw[b].qs, 32);
            std::memcpy(f.weights.data() + n * 32 + b * 2, &raw[b].d, 2);
        }
        dequantize_row_q8_0(raw.data(), f.decoded_weights.data(), int64_t(rows) * k);
    }
    f.activations.resize(size_t(columns) * f.activation_stride);
    for (int c = 0; c < columns; ++c) for (int b = 0; b < k / 32; ++b) {
        const float d = scale(1.0f / 64, c, b);
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int8_t q = int8_t(int(rng() % 63) - 31);
            sum += q;
            std::memcpy(f.activations.data() + size_t(c) * f.activation_stride + b * 32 + i, &q, 1);
            f.decoded_activations[size_t(c) * k + b * 32 + i] = float(q) * d;
        }
        store_half(f.activations, size_t(c) * f.activation_stride + k + b * 4, d);
        store_half(f.activations, size_t(c) * f.activation_stride + k + b * 4 + 2, float(sum) * d);
    }
    f.expected.resize(size_t(rows) * columns);
    for (int c = 0; c < columns; ++c) for (int r = 0; r < rows; ++r) {
        f.expected[size_t(c) * rows + r] = float(dot_product(
            f.decoded_weights.data() + size_t(r) * k, f.decoded_activations.data() + size_t(c) * k, k));
    }
    return f;
}

Fixture make_fixture(Type type, int rows, int columns, int k, unsigned seed) {
    return make_fixture(type, rows, columns, k, seed, false);
}

Fixture make_fixture_with_varying_scales(Type type, int rows, int columns, int k, unsigned seed) {
    return make_fixture(type, rows, columns, k, seed, true);
}

std::vector<float> glu_reference(const Fixture & up, const Fixture & gate, Glu operation) {
    if (up.rows != gate.rows || up.k != gate.k) throw std::invalid_argument("GLU weight shape mismatch");
    auto expected = up.expected;
    for (int c = 0; c < up.columns; ++c) for (int r = 0; r < up.rows; ++r) {
        const double g = dot_product(gate.decoded_weights.data() + size_t(r) * gate.k,
                                     up.decoded_activations.data() + size_t(c) * up.k, up.k);
        const double activation = operation == Glu::swiglu
            ? g / (1.0 + std::exp(-g))
            : 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * g * (1.0 + 0.044715 * g * g)));
        expected[size_t(c) * up.rows + r] *= activation;
    }
    return expected;
}

Comparison compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size() || actual.empty()) throw std::invalid_argument("Output size mismatch");
    double err2 = 0, ref2 = 0, max_error = 0, max_ref = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return {false, INFINITY, INFINITY};
        const double error = std::abs(double(actual[i]) - expected[i]);
        err2 += error * error; ref2 += double(expected[i]) * expected[i];
        max_error = std::max(max_error, error); max_ref = std::max(max_ref, std::abs(double(expected[i])));
    }
    const double relative = std::sqrt(err2 / std::max(ref2, 1e-30));
    return {relative <= 1e-4 && max_error <= 1e-3 * (1 + max_ref), relative, max_error};
}
}
