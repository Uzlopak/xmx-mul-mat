// SPDX-License-Identifier: MIT
#include "xmx/kernel.hpp"
#include "legacy-column-tiles.hpp"
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
using Traits = kernels::xmx_traits_q8_0<false>;

template <typename T> struct DeviceBuffer {
    sycl::queue & queue;
    T * pointer;
    DeviceBuffer(sycl::queue & q, size_t count) : queue(q), pointer(sycl::aligned_alloc_device<T>(64, count, q)) {
        if (!pointer) throw std::runtime_error("Device allocation failed");
    }
    ~DeviceBuffer() { try { sycl::free(pointer, queue); } catch (...) {} }
    DeviceBuffer(const DeviceBuffer &) = delete;
};

enum class Schedule { current, exact, padded };

static const char * schedule_name(Schedule schedule) {
    switch (schedule) {
        case Schedule::current: return "current";
        case Schedule::exact: return "exact";
        case Schedule::padded: return "padded";
    }
    throw std::logic_error("Invalid column schedule");
}

int main(int argc, char ** argv) {
    try {
        if (argc != 8 || std::string_view(argv[1]) != "--run-gpu") {
            std::cerr << "Usage: xmx_kernel_tests --run-gpu rows k first_column last_column rounds iterations\n";
            return 2;
        }
        const int rows = std::stoi(argv[2]), k = std::stoi(argv[3]);
        const int first = std::stoi(argv[4]), last = std::stoi(argv[5]);
        const int rounds = std::stoi(argv[6]), iterations = std::stoi(argv[7]);
        if (first < 1 || last > 64 || last < first || last - first >= 16 ||
            rounds < 1 || rounds > 9 || iterations < 1 || iterations > 64) {
            throw std::invalid_argument("Expected a range of at most 16 widths in 1..64 and bounded timing rounds");
        }
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto e : errors) std::rethrow_exception(e);
        }, sycl::property::queue::in_order{});
        std::cerr << "Device " << q.get_device().get_info<sycl::info::device::name>() << '\n';
        const auto fixture = xmx::make_fixture_with_varying_scales(xmx::Type::q8_0, rows, last, k);
        DeviceBuffer<uint8_t> weights(q, fixture.weights.size());
        q.memcpy(weights.pointer, fixture.weights.data(), fixture.weights.size()).wait_and_throw();

        constexpr int tile = kernels::XMX_COLUMN_TILE;
        constexpr float sentinel = std::numeric_limits<float>::max();
        for (int columns = first; columns <= last; ++columns) {
            const size_t activation_size = size_t(columns) * fixture.activation_stride;
            const size_t output_size = size_t(columns) * rows;
            const size_t guarded_size = output_size + size_t(tile) * rows;
            const std::vector<float> expected(fixture.expected.begin(), fixture.expected.begin() + output_size);
            std::vector<float> actual(output_size), guard(guarded_size - output_size);
            DeviceBuffer<uint8_t> activations(q, activation_size);
            DeviceBuffer<float> output(q, guarded_size);
            q.memcpy(activations.pointer, fixture.activations.data(), activation_size).wait_and_throw();
            const char * input = reinterpret_cast<const char *>(activations.pointer);
            const int remainder = columns % tile;
            const bool compare_schedules = columns > tile / 2 && remainder != 0;
            const std::array<Schedule, 2> plans{
                Schedule::current, remainder > tile / 2 ? Schedule::exact : Schedule::padded,
            };

            auto launch = [&](Schedule schedule) {
                switch (schedule) {
                    case Schedule::current:
                        kernels::xmx_mul_mat_ncols<Traits>(
                            weights.pointer, nullptr, input, output.pointer, k, rows, columns,
                            fixture.activation_stride, rows, 0, &q);
                        break;
                    case Schedule::exact:
                        kernels::xmx_mul_mat_submit_legacy_columns<Traits, false, false>(
                            weights.pointer, nullptr, input, output.pointer, k, rows, columns,
                            fixture.activation_stride, rows, 0, &q);
                        break;
                    case Schedule::padded:
                        kernels::xmx_mul_mat_submit<Traits, tile, 1, false, false, true>(
                            weights.pointer, nullptr, input, output.pointer, k, rows, columns,
                            fixture.activation_stride, rows, 0, &q);
                        break;
                }
            };

            const int distinct = compare_schedules ? 2 : 1;
            for (int round = 0; round < rounds; ++round) {
                for (int i = 0; i < distinct; ++i) {
                    const auto schedule = plans[(round + i) % distinct];
                    q.fill(output.pointer, sentinel, guarded_size).wait_and_throw();
                    launch(schedule);
                    q.wait_and_throw();
                    const auto start = std::chrono::steady_clock::now();
                    for (int j = 0; j < iterations; ++j) launch(schedule);
                    q.wait_and_throw();
                    const double microseconds =
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() /
                        iterations;
                    q.memcpy(actual.data(), output.pointer, output_size * sizeof(float)).wait_and_throw();
                    q.memcpy(guard.data(), output.pointer + output_size, guard.size() * sizeof(float)).wait_and_throw();
                    if (!std::all_of(guard.begin(), guard.end(), [=](float value) { return value == sentinel; })) {
                        throw std::runtime_error("Kernel wrote to padded output columns");
                    }
                    const auto comparison = xmx::compare(actual, expected);
                    if (!comparison.passed) throw std::runtime_error("CPU reference mismatch");
                    std::cout << "{\"rows\":" << rows << ",\"columns\":" << columns << ",\"k\":" << k
                              << ",\"round\":" << round << ",\"iterations\":" << iterations
                              << ",\"schedule\":\"" << schedule_name(schedule) << "\",\"microseconds\":" << std::setprecision(8)
                              << microseconds << ",\"relative_l2\":" << comparison.relative_l2 << ",\"passed\":true}"
                              << std::endl;
                }
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
