// SPDX-License-Identifier: MIT
#include "xmx/kernel.hpp"
#include "xmx/fixtures.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace kernels = ggml_sycl_esimd;

template <typename T> struct DeviceBuffer {
    sycl::queue & queue;
    T * pointer;
    DeviceBuffer(sycl::queue & q, size_t count) : queue(q), pointer(sycl::aligned_alloc_device<T>(64, count, q)) {
        if (!pointer) throw std::runtime_error("Device allocation failed");
    }
    ~DeviceBuffer() { try { sycl::free(pointer, queue); } catch (...) {} }
    DeviceBuffer(const DeviceBuffer &) = delete;
};

struct Launch {
    sycl::queue & queue;
    const uint8_t * weights;
    const uint8_t * gate;
    const char * activations;
    float * output;
    int k;
    int rows;
    int columns;
    int stride;
    int operation;

    template <typename Traits, int Width> void padded() const {
        kernels::xmx_mul_mat_submit<Traits, Width, 1, true, false, true>(
            weights, gate, activations, output, k, rows, columns, stride, rows, operation, &queue);
    }

    template <typename Traits> void run(int width) const {
        if (width == columns) {
            kernels::xmx_mul_mat_ncols<Traits, true>(
                weights, gate, activations, output, k, rows, columns, stride, rows, operation, &queue);
            return;
        }
        if constexpr (Traits::plain) {
            switch (width) {
                case 2: padded<Traits, 2>(); return;
                case 3: padded<Traits, 3>(); return;
                case 4: padded<Traits, 4>(); return;
                case 5: padded<Traits, 5>(); return;
                case 6: padded<Traits, 6>(); return;
                case 7: padded<Traits, 7>(); return;
            }
        }
        if (width != 8) throw std::logic_error("Unsupported padded width");
        padded<Traits, 8>();
    }
};

template <typename Traits>
static void sweep(sycl::queue & q, xmx::Type type, int rows, int k, int first, int last, int rounds, int iterations) {
    const auto up = xmx::make_fixture_with_varying_scales(type, rows, last, k);
    auto gate = xmx::make_fixture_with_varying_scales(type, rows, 1, k, 43);
    const auto swiglu = xmx::glu_reference(up, gate, xmx::Glu::swiglu);
    const auto geglu = xmx::glu_reference(up, gate, xmx::Glu::geglu);
    DeviceBuffer<uint8_t> weights(q, up.weights.size()), gate_weights(q, gate.weights.size());
    q.memcpy(weights.pointer, up.weights.data(), up.weights.size()).wait_and_throw();
    q.memcpy(gate_weights.pointer, gate.weights.data(), gate.weights.size()).wait_and_throw();

    constexpr int max_width = kernels::XMX_FUSED_COLUMN_TILE;
    constexpr float sentinel = std::numeric_limits<float>::max();
    const size_t guarded_size = size_t(max_width) * rows;
    for (int columns = first; columns <= last; ++columns) {
        const size_t activation_size = size_t(columns) * up.activation_stride;
        const size_t output_size = size_t(columns) * rows;
        DeviceBuffer<uint8_t> activations(q, activation_size);
        DeviceBuffer<float> output(q, guarded_size);
        q.memcpy(activations.pointer, up.activations.data(), activation_size).wait_and_throw();
        std::vector<float> actual(output_size), guard(guarded_size - output_size);
        std::vector<int> widths{columns};
        if constexpr (Traits::plain) {
            for (int width = columns + 1; width <= max_width; ++width) widths.push_back(width);
        } else {
            if (columns < max_width) widths.push_back(max_width);
        }

        for (int round = 0; round < rounds; ++round) {
            for (int act = 0; act < 2; ++act) {
                const bool swish = ((round + act) % 2 == 0);
                const auto & reference = swish ? swiglu : geglu;
                const std::vector<float> expected(reference.begin(), reference.begin() + output_size);
                const Launch launch{q, weights.pointer, gate_weights.pointer,
                    reinterpret_cast<const char *>(activations.pointer), output.pointer, k, rows, columns,
                    up.activation_stride, swish ? GGML_GLU_OP_SWIGLU : GGML_GLU_OP_GEGLU};
                for (size_t i = 0; i < widths.size(); ++i) {
                    const int width = widths[(i + round) % widths.size()];
                    q.fill(output.pointer, sentinel, guarded_size).wait_and_throw();
                    launch.run<Traits>(width);
                    q.wait_and_throw();
                    const auto start = std::chrono::steady_clock::now();
                    for (int j = 0; j < iterations; ++j) launch.run<Traits>(width);
                    q.wait_and_throw();
                    const double us =
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() /
                        iterations;
                    q.memcpy(actual.data(), output.pointer, output_size * sizeof(float)).wait_and_throw();
                    if (!guard.empty()) {
                        q.memcpy(guard.data(), output.pointer + output_size, guard.size() * sizeof(float)).wait_and_throw();
                        if (!std::all_of(guard.begin(), guard.end(), [=](float v) { return v == sentinel; })) {
                            throw std::runtime_error("Kernel wrote to a guard column");
                        }
                    }
                    const auto comparison = xmx::compare(actual, expected);
                    if (!comparison.passed) throw std::runtime_error("CPU reference mismatch");
                    std::cout << "{\"type\":\"" << xmx::type_name(type) << "\",\"glu\":\""
                              << (swish ? "swiglu" : "geglu") << "\",\"rows\":" << rows << ",\"k\":" << k
                              << ",\"columns\":" << columns << ",\"tile_width\":" << width << ",\"round\":" << round
                              << ",\"iterations\":" << iterations << ",\"microseconds\":" << std::setprecision(9)
                              << us << ",\"relative_l2\":" << comparison.relative_l2 << ",\"passed\":true}"
                              << std::endl;
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc != 9 || std::string_view(argv[1]) != "--run-gpu") {
            std::cerr << "Usage: xmx_kernel_tests --run-gpu type rows k first_width last_width rounds iterations\n";
            return 2;
        }
        const auto type = xmx::parse_type(argv[2]);
        const int rows = std::stoi(argv[3]), k = std::stoi(argv[4]);
        const int first = std::stoi(argv[5]), last = std::stoi(argv[6]);
        const int rounds = std::stoi(argv[7]), iterations = std::stoi(argv[8]);
        if (first < 1 || last > 8 || last < first || rounds < 1 || rounds > 8 || iterations < 1 || iterations > 32) {
            throw std::invalid_argument("Expected bounded widths, rounds and iterations");
        }
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto e : errors) std::rethrow_exception(e);
        }, sycl::property::queue::in_order{});
        std::cerr << "Device " << q.get_device().get_info<sycl::info::device::name>() << '\n';
        switch (type) {
            case xmx::Type::q2_k: sweep<kernels::xmx_traits_q2_k<false>>(q, type, rows, k, first, last, rounds, iterations); break;
            case xmx::Type::q3_k: sweep<kernels::xmx_traits_q3_k<false>>(q, type, rows, k, first, last, rounds, iterations); break;
            case xmx::Type::q4_k: sweep<kernels::xmx_traits_q4_k<false>>(q, type, rows, k, first, last, rounds, iterations); break;
            case xmx::Type::q5_k: sweep<kernels::xmx_traits_q5_k<false>>(q, type, rows, k, first, last, rounds, iterations); break;
            case xmx::Type::q6_k: sweep<kernels::xmx_traits_q6_k<false>>(q, type, rows, k, first, last, rounds, iterations); break;
            case xmx::Type::q8_0: sweep<kernels::xmx_traits_q8_0<false>>(q, type, rows, k, first, last, rounds, iterations); break;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
