// SPDX-License-Identifier: MIT
#include "xmx/kernel.hpp"
#include "xmx/fixtures.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace kernels = ggml_sycl_esimd;
using Traits = kernels::xmx_traits_q8_0<false>;

template <typename T> struct Buffer {
    sycl::queue & queue;
    T * data;
    Buffer(sycl::queue & q, size_t size) : queue(q), data(sycl::aligned_alloc_device<T>(64, size, q)) {
        if (!data) throw std::runtime_error("Device allocation failed");
    }
    ~Buffer() { try { sycl::free(data, queue); } catch (...) {} }
    Buffer(const Buffer &) = delete;
};

struct Launch {
    sycl::queue & queue;
    const void * weights;
    const char * activation;
    float * output;
    int k;
    int rows;
    int columns;
    int stride;

    template <int Width, bool Slm> void run_case() const {
        constexpr int max_ks = Slm ? kernels::xmx_n_split<Width>() : 1;
        kernels::xmx_mul_mat_launch_ks<Traits, Width, max_ks, false>(
            weights, nullptr, activation, output, k, rows, columns, stride, rows, 0, &queue);
    }

    void run(bool slm) const {
        if (columns == 1) {
            if (slm) run_case<1, true>();
            else run_case<1, false>();
        } else {
            if (slm) run_case<16, true>();
            else run_case<16, false>();
        }
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 7 || std::string(argv[1]) != "--run-gpu") {
            std::cerr << "Usage: slm-study --run-gpu rows columns K rounds iterations\n";
            return 2;
        }
        const int rows = std::stoi(argv[2]), columns = std::stoi(argv[3]), k = std::stoi(argv[4]);
        const int rounds = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
        if ((columns != 1 && columns != 16) || rounds < 1 || rounds > 12 || iterations < 1 || iterations > 64) {
            throw std::invalid_argument("Expected one or sixteen columns and bounded rounds and iterations");
        }
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto e : errors) std::rethrow_exception(e);
        }, sycl::property::queue::in_order{});
        const auto fixture = xmx::make_fixture_with_varying_scales(xmx::Type::q8_0, rows, columns, k);
        Buffer<uint8_t> weights(q, fixture.weights.size()), activation(q, fixture.activations.size());
        const size_t output_size = fixture.expected.size();
        const size_t guard_size = size_t(rows) * 16;
        Buffer<float> output(q, output_size + guard_size);
        q.memcpy(weights.data, fixture.weights.data(), fixture.weights.size()).wait_and_throw();
        q.memcpy(activation.data, fixture.activations.data(), fixture.activations.size()).wait_and_throw();
        const Launch launch{q, weights.data, reinterpret_cast<const char *>(activation.data), output.data,
                            k, rows, columns, fixture.activation_stride};
        std::vector<float> result(output_size), guard(guard_size);
        constexpr float sentinel = std::numeric_limits<float>::max();
        const int bpr = k / QK_K;
        const int max_ks = columns == 1 ? kernels::xmx_n_split<1>() : kernels::xmx_n_split<16>();
        const int k_rounds = (bpr + max_ks - 1) / max_ks;
        const int split = (bpr + k_rounds - 1) / k_rounds;
        for (int round = 0; round < rounds; ++round) {
            for (int i = 0; i < 2; ++i) {
                const bool slm = (i + round) % 2 != 0;
                q.fill(output.data, sentinel, output_size + guard_size).wait_and_throw();
                launch.run(slm);
                q.wait_and_throw();
                auto start = std::chrono::steady_clock::now();
                for (int j = 0; j < iterations; ++j) launch.run(slm);
                q.wait_and_throw();
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / iterations;
                q.memcpy(result.data(), output.data, output_size * sizeof(float)).wait_and_throw();
                q.memcpy(guard.data(), output.data + output_size, guard_size * sizeof(float)).wait_and_throw();
                const auto comparison = xmx::compare(result, fixture.expected);
                if (!comparison.passed ||
                    !std::all_of(guard.begin(), guard.end(), [=](float value) { return value == sentinel; })) {
                    throw std::runtime_error("CPU reference or guard check failed");
                }
                std::cout << "{\"round\":" << round << ",\"slm\":" << (slm ? "true" : "false")
                          << ",\"rows\":" << rows << ",\"columns\":" << columns << ",\"k\":" << k
                          << ",\"work_items_per_group\":" << (slm ? split : 1) << ",\"iterations\":" << iterations
                          << ",\"microseconds\":" << std::setprecision(9) << us
                          << ",\"relative_l2\":" << comparison.relative_l2 << ",\"passed\":true}" << std::endl;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
