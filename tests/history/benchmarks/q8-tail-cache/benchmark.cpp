// SPDX-License-Identifier: MIT
#include "submit.hpp"
#include "xmx/fixtures.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
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
    sycl::queue & q;
    const void * weights;
    const char * activations;
    float * output;
    int k;
    int rows;
    int stride;

    template <int Tail> sycl::event submit(int columns) const {
        return kernels::xmx_cache_submit<Traits, false, false, Tail>(
            weights, nullptr, activations, output, k, rows, columns, stride, rows, 0, &q);
    }

    sycl::event run(int columns, bool fixed) const {
        if (!fixed) return submit<0>(columns);
        switch (columns - 32) {
            case 2: return submit<2>(columns);
            case 3: return submit<3>(columns);
            case 4: return submit<4>(columns);
            case 5: return submit<5>(columns);
            case 6: return submit<6>(columns);
            case 7: return submit<7>(columns);
            case 8: return submit<8>(columns);
        }
        throw std::logic_error("Unsupported fixed tail");
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 6 || std::string(argv[1]) != "--run-gpu") {
            std::cerr << "Usage: cache-study --run-gpu rows K rounds repetitions-per-width\n";
            return 2;
        }
        const int rows = std::stoi(argv[2]), k = std::stoi(argv[3]);
        const int rounds = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        if (rounds < 1 || rounds > 16 || repeats < 1 || repeats > 32) {
            throw std::invalid_argument("Expected bounded rounds and repeats");
        }
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto e : errors) std::rethrow_exception(e);
        }, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        const auto fixture = xmx::make_fixture_with_varying_scales(xmx::Type::q8_0, rows, 40, k);
        Buffer<uint8_t> weights(q, fixture.weights.size()), activations(q, fixture.activations.size());
        Buffer<float> output(q, size_t(48) * rows);
        const Launch launch{q, weights.data, reinterpret_cast<const char *>(activations.data), output.data,
                            k, rows, fixture.activation_stride};
        q.memcpy(weights.data, fixture.weights.data(), fixture.weights.size()).wait_and_throw();
        q.memcpy(activations.data, fixture.activations.data(), fixture.activations.size()).wait_and_throw();
        constexpr float sentinel = std::numeric_limits<float>::max();
        std::vector<float> result(size_t(48) * rows);
        std::array<int, 7> widths{34, 35, 36, 37, 38, 39, 40};
        for (bool fixed : {false, true}) {
            for (int width : widths) {
                q.fill(output.data, sentinel, result.size()).wait_and_throw();
                launch.run(width, fixed).wait_and_throw();
                q.memcpy(result.data(), output.data, result.size() * sizeof(float)).wait_and_throw();
                const size_t size = size_t(width) * rows;
                const std::vector<float> actual(result.begin(), result.begin() + size);
                const std::vector<float> expected(fixture.expected.begin(), fixture.expected.begin() + size);
                if (!xmx::compare(actual, expected).passed ||
                    !std::all_of(result.begin() + size, result.end(), [=](float n) { return n == sentinel; })) {
                    throw std::runtime_error("CPU reference or guard check failed");
                }
            }
        }
        std::cerr << "Reference and guards passed for 14 variants\n";
        using Start = sycl::info::event_profiling::command_start;
        using End = sycl::info::event_profiling::command_end;
        std::mt19937 rng(4242);
        for (int round = 0; round < rounds; ++round) {
            std::shuffle(widths.begin(), widths.end(), rng);
            for (int offset = 0; offset < 4; ++offset) {
                const int mode = (round + offset) % 4;
                const bool fixed = mode >= 2, mixed = mode % 2 != 0;
                auto begin = std::chrono::steady_clock::now();
                std::vector<sycl::event> events;
                events.reserve(widths.size() * repeats);
                for (size_t i = 0; i < widths.size() * repeats; ++i) {
                    const size_t index = mixed ? i % widths.size() : i / repeats;
                    events.push_back(launch.run(widths[index], fixed));
                }
                q.wait_and_throw();
                const double wall = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin).count() / events.size();
                const auto first = events.front().get_profiling_info<Start>();
                const auto last = events.back().get_profiling_info<End>();
                uint64_t busy = 0;
                for (auto & event : events) {
                    busy += event.get_profiling_info<End>() - event.get_profiling_info<Start>();
                }
                std::cout << "{\"round\":" << round << ",\"kernel\":\"" << (fixed ? "fixed" : "generic")
                          << "\",\"order\":\"" << (mixed ? "mixed" : "grouped") << "\",\"launches\":" << events.size()
                          << ",\"gpu_us\":" << double(busy) / events.size() / 1000
                          << ",\"device_span_us\":" << double(last - first) / events.size() / 1000
                          << ",\"wall_us\":" << wall << "}" << std::endl;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
