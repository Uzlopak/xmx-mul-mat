#include "esimd.hpp"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace kernels = ggml_sycl_esimd;

static void run_case(sycl::queue & queue, ggml_type type, int rows, int cols) {
    const int k = 1024;
    const int stride = k + k / 32 * 4;
    std::mt19937 rng(42 + rows + cols);
    std::normal_distribution<float> distribution(0, 0.25f);
    std::vector<float> w(rows * k), y(cols * k), decoded(rows * k), yd(cols * k);
    for (auto & value : w) value = distribution(rng);
    for (auto & value : y) value = distribution(rng);
    const int blocks = rows * (k / (type == GGML_TYPE_Q8_0 ? 32 : 256));
    std::vector<uint8_t> weights(ggml_row_size(type, k) * rows);
    if (type == GGML_TYPE_Q8_0) {
        std::vector<block_q8_0> raw(blocks);
        quantize_row_q8_0_ref(w.data(), raw.data(), rows * k);
        dequantize_row_q8_0(raw.data(), decoded.data(), rows * k);
        for (int b = 0; b < blocks; ++b) {
            std::memcpy(weights.data() + b * 32, raw[b].qs, 32);
            std::memcpy(weights.data() + blocks * 32 + b * 2, &raw[b].d, 2);
        }
    } else if (type == GGML_TYPE_Q4_K) {
        std::vector<block_q4_K> raw(blocks);
        quantize_row_q4_K_ref(w.data(), raw.data(), rows * k);
        dequantize_row_q4_K(raw.data(), decoded.data(), rows * k);
        for (int b = 0; b < blocks; ++b) {
            std::memcpy(weights.data() + b * 128, raw[b].qs, 128);
            std::memcpy(weights.data() + blocks * 128 + b * 12, raw[b].scales, 12);
            std::memcpy(weights.data() + blocks * 140 + b * 4, &raw[b], 4);
        }
    } else {
        std::vector<block_q6_K> raw(blocks);
        quantize_row_q6_K_ref(w.data(), raw.data(), rows * k);
        dequantize_row_q6_K(raw.data(), decoded.data(), rows * k);
        for (int b = 0; b < blocks; ++b) {
            std::memcpy(weights.data() + b * 128, raw[b].ql, 128);
            std::memcpy(weights.data() + blocks * 128 + b * 64, raw[b].qh, 64);
            std::memcpy(weights.data() + blocks * 192 + b * 16, raw[b].scales, 16);
            std::memcpy(weights.data() + blocks * 208 + b * 2, &raw[b].d, 2);
        }
    }
    std::vector<uint8_t> activation(cols * stride);
    for (int c = 0; c < cols; ++c) {
        std::vector<block_q8_1> raw(k / 32);
        quantize_row_q8_1_ref(y.data() + c * k, raw.data(), k);
        for (int b = 0; b < k / 32; ++b) {
            std::memcpy(activation.data() + c * stride + b * 32, raw[b].qs, 32);
            std::memcpy(activation.data() + c * stride + k + b * 4, &raw[b], 4);
            ggml_fp16_t bits; std::memcpy(&bits, &raw[b], sizeof(bits));
            const float d = ggml_fp16_to_fp32(bits);
            for (int i = 0; i < 32; ++i) yd[c * k + b * 32 + i] = d * raw[b].qs[i];
        }
    }
    auto * dw = sycl::aligned_alloc_device<uint8_t>(64, weights.size(), queue);
    auto * dy = sycl::aligned_alloc_device<uint8_t>(64, activation.size(), queue);
    auto * output = sycl::malloc_device<float>(rows * cols, queue);
    if (!dw || !dy || !output) throw std::runtime_error("USM allocation failed");
    queue.memcpy(dw, weights.data(), weights.size()).wait();
    queue.memcpy(dy, activation.data(), activation.size()).wait();
    auto launch = [&] {
#define LAUNCH(N) \
        case N: \
            if (type == GGML_TYPE_Q8_0) kernels::xmx_mul_mat_launch<kernels::xmx_traits_q8_0<false>, N, false>(dw, nullptr, (const char *) dy, output, k, rows, cols, stride, rows, 0, &queue); \
            else if (type == GGML_TYPE_Q4_K) kernels::xmx_mul_mat_launch<kernels::xmx_traits_q4_k<false>, N, false>(dw, nullptr, (const char *) dy, output, k, rows, cols, stride, rows, 0, &queue); \
            else kernels::xmx_mul_mat_launch<kernels::xmx_traits_q6_k<false>, N, false>(dw, nullptr, (const char *) dy, output, k, rows, cols, stride, rows, 0, &queue); \
            break
        switch (cols) { LAUNCH(1); LAUNCH(2); LAUNCH(4); LAUNCH(8); LAUNCH(16); }
#undef LAUNCH
    };
    launch(); queue.wait_and_throw();
    std::vector<float> actual(rows * cols);
    queue.memcpy(actual.data(), output, actual.size() * sizeof(float)).wait();
    double error2 = 0, reference2 = 0, max_error = 0, max_reference = 0;
    for (int c = 0; c < cols; ++c) for (int r = 0; r < rows; ++r) {
        double reference = 0;
        for (int i = 0; i < k; ++i) reference += double(decoded[r * k + i]) * yd[c * k + i];
        const double error = std::abs(actual[c * rows + r] - reference);
        if (!std::isfinite(actual[c * rows + r])) throw std::runtime_error("Non-finite output");
        error2 += error * error; reference2 += reference * reference;
        max_error = std::max(max_error, error); max_reference = std::max(max_reference, std::abs(reference));
    }
    const double relative = std::sqrt(error2 / std::max(reference2, 1e-30));
    const bool pass = relative <= 1e-4 && max_error <= 1e-3 * (1 + max_reference);
    std::cout << "{\"type\":\"" << ggml_type_name(type) << "\",\"rows\":" << rows << ",\"columns\":" << cols
              << ",\"k\":" << k << ",\"relative_l2\":" << relative << ",\"max_abs\":" << max_error
              << ",\"passed\":" << (pass ? "true" : "false") << "}" << std::endl;
    sycl::free(dw, queue); sycl::free(dy, queue); sycl::free(output, queue);
    if (!pass) throw std::runtime_error("CPU reference mismatch");
}
int main() {
    try {
        sycl::queue queue(sycl::gpu_selector_v, [](sycl::exception_list errors) { for (auto e : errors) std::rethrow_exception(e); }, sycl::property::queue::in_order{});
        std::cerr << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0})
            for (int rows : {16, 17, 128}) for (int cols : {1, 2, 4, 8, 16}) run_case(queue, type, rows, cols);
    } catch (const std::exception & error) { std::cerr << error.what() << std::endl; return 1; }
}
