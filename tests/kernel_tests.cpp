// SPDX-License-Identifier: MIT
#include "xmx/kernel.hpp"
#include "xmx/fixtures.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace kernels = ggml_sycl_esimd;

struct Options {
    bool run = false, suite = false;
    bool short_rows = false, require_native_2d = false, vary_scales = false;
    xmx::Type type = xmx::Type::q8_0;
    bool selected_type = false;
    std::optional<xmx::Glu> glu;
    int rows = 16, cols = 1, k = 1024, iterations = 0;
};
static void help() {
    std::cout << "Usage: xmx_kernel_tests --run-gpu [--suite] [--type q8_0|q2_k|q3_k|q4_k|q5_k|q6_k]\n"
                 "       [--rows N] [--cols 1..128] [--k 1024] [--iterations N]\n"
                 "       [--glu swiglu|geglu] [--short-rows] [--require-native-2d] [--vary-scales]\n"
                 "       xmx_kernel_tests --check-zero-k\n"
                 "No GPU kernel is run without --run-gpu.\n"
                 "The unchanged Xe2 kernel previously reset an Arrow Lake-H compute engine.\n"
                 "--iterations reports kernel-only timings, excluding upload and input quantization.\n";
}
template <typename T> struct DeviceBuffer {
    sycl::queue & queue;
    T * pointer;
    DeviceBuffer(sycl::queue & q, size_t n) : queue(q), pointer(sycl::aligned_alloc_device<T>(64, n, q)) {
        if (!pointer) throw std::runtime_error("Device allocation failed");
    }
    ~DeviceBuffer() { try { sycl::free(pointer, queue); } catch (...) {} }
    DeviceBuffer(const DeviceBuffer &) = delete;
};

template <bool SHORT_ROWS>
static void expect_zero_k_rejection(int columns) {
    alignas(64) uint8_t weights[64]{};
    alignas(64) char activations[64]{};
    alignas(64) float output[kernels::XMX_ROWS]{};
    using Traits = kernels::xmx_traits_q8_0<SHORT_ROWS>;
    try {
        // Validation must reject this input before it needs a queue.
        kernels::xmx_mul_mat_ncols<Traits>(
            weights, nullptr, activations, output, 0, kernels::XMX_ROWS, columns, 64, kernels::XMX_ROWS, 0, nullptr);
    } catch (const std::invalid_argument & error) {
        if (std::string(error.what()).find("ncols > 0") != std::string::npos) {
            return;
        }
        throw;
    }
    throw std::runtime_error("launcher accepted zero K");
}

static void check_zero_k() {
    expect_zero_k_rejection<false>(1);
    // A wide short-row request must reach the launcher instead of returning early.
    expect_zero_k_rejection<true>(9);
    std::cout << "PASS: launcher rejected zero K for single-column and wide short-row requests\n";
}

static void run_case(sycl::queue & q, const Options & o) {
    const auto f = o.vary_scales ? xmx::make_fixture_with_varying_scales(o.type, o.rows, o.cols, o.k)
                                 : xmx::make_fixture(o.type, o.rows, o.cols, o.k);
    const char * glu_name = nullptr;
    int glu_op = 0;
    std::optional<DeviceBuffer<uint8_t>> gate_weights;
    std::vector<float> fused_expected;
    if (o.glu) {
        // The gate uses independent weights and shares the up projection's activations.
        const auto gate = o.vary_scales ? xmx::make_fixture_with_varying_scales(o.type, o.rows, 1, o.k, 43)
                                       : xmx::make_fixture(o.type, o.rows, 1, o.k, 43);
        fused_expected = xmx::glu_reference(f, gate, *o.glu);
        gate_weights.emplace(q, gate.weights.size());
        q.memcpy(gate_weights->pointer, gate.weights.data(), gate.weights.size()).wait_and_throw();
        glu_op = *o.glu == xmx::Glu::swiglu ? GGML_GLU_OP_SWIGLU : GGML_GLU_OP_GEGLU;
        glu_name = *o.glu == xmx::Glu::swiglu ? "swiglu" : "geglu";
    }
    const auto & expected = o.glu ? fused_expected : f.expected;
    const int tile_columns = o.glu ? kernels::XMX_FUSED_COLUMN_TILE : kernels::XMX_COLUMN_TILE;
    const int guard_columns = o.glu || o.cols > 8 ? (tile_columns - o.cols % tile_columns) % tile_columns : 0;
    const size_t guard_size = size_t(guard_columns) * o.rows;
    constexpr float unwritten = std::numeric_limits<float>::max();
    DeviceBuffer<uint8_t> w(q, f.weights.size()), y(q, f.activations.size());
    DeviceBuffer<float> output(q, f.expected.size() + guard_size);
    if (o.require_native_2d &&
        !kernels::xmx_native_2d_geometry<kernels::xmx_traits_q8_0<false>, false>(
            reinterpret_cast<uintptr_t>(w.pointer), 0, reinterpret_cast<uintptr_t>(y.pointer),
            o.k, o.rows, o.cols, f.activation_stride)) {
        throw std::invalid_argument("Case must be eligible for native 2D loads");
    }
    q.memcpy(w.pointer, f.weights.data(), f.weights.size()).wait();
    q.memcpy(y.pointer, f.activations.data(), f.activations.size()).wait();
    q.fill(output.pointer, unwritten, f.expected.size() + guard_size).wait_and_throw();
    auto launch_type = [&](auto traits) {
        auto launch = [&](auto mode) {
            kernels::xmx_mul_mat_ncols<decltype(traits), decltype(mode)::value>(
                w.pointer, gate_weights ? gate_weights->pointer : nullptr, reinterpret_cast<const char *>(y.pointer),
                output.pointer, o.k, o.rows, o.cols, f.activation_stride, o.rows, glu_op, &q);
        };
        if (o.glu) launch(std::true_type{});
        else launch(std::false_type{});
    };
    auto launch_rows = [&](auto traits) {
        if (o.short_rows) {
            launch_type(typename decltype(traits)::short_row_traits{});
        } else {
            launch_type(traits);
        }
    };
    auto launch = [&] {
        switch (o.type) {
            case xmx::Type::q8_0: launch_rows(kernels::xmx_traits_q8_0<false>{}); break;
            case xmx::Type::q2_k: launch_rows(kernels::xmx_traits_q2_k<false>{}); break;
            case xmx::Type::q3_k: launch_rows(kernels::xmx_traits_q3_k<false>{}); break;
            case xmx::Type::q4_k: launch_rows(kernels::xmx_traits_q4_k<false>{}); break;
            case xmx::Type::q5_k: launch_rows(kernels::xmx_traits_q5_k<false>{}); break;
            case xmx::Type::q6_k: launch_rows(kernels::xmx_traits_q6_k<false>{}); break;
        }
    };
    std::cerr << "Launching " << xmx::type_name(o.type) << " rows=" << o.rows << " cols=" << o.cols << " K=" << o.k;
    if (glu_name) std::cerr << " glu=" << glu_name;
    if (o.short_rows) std::cerr << " short_rows=true";
    std::cerr << std::endl;
    launch(); q.wait_and_throw();
    std::vector<float> actual(f.expected.size());
    q.memcpy(actual.data(), output.pointer, actual.size() * sizeof(float)).wait();
    if (guard_size) {
        // Make any stores into the padded columns observable.
        std::vector<float> guard(guard_size);
        q.memcpy(guard.data(), output.pointer + actual.size(), guard.size() * sizeof(float)).wait_and_throw();
        if (!std::all_of(guard.begin(), guard.end(), [=](float value) { return value == unwritten; })) {
            throw std::runtime_error("Kernel wrote to padded output columns");
        }
    }
    const auto comparison = xmx::compare(actual, expected);
    std::cout << "{\"type\":\"" << xmx::type_name(o.type) << "\",\"rows\":" << o.rows
              << ",\"columns\":" << o.cols << ",\"k\":" << o.k
              << ",\"relative_l2\":" << comparison.relative_l2 << ",\"max_abs\":" << comparison.max_absolute
              << ",\"passed\":" << (comparison.passed ? "true" : "false");
    if (glu_name) std::cout << ",\"glu\":\"" << glu_name << "\"";
    if (o.short_rows) std::cout << ",\"short_rows\":true";
    std::cout << "}" << std::endl;
    if (!comparison.passed) {
        const size_t limit = std::min<size_t>(actual.size(), 32);
        for (size_t i = 0; i < limit; ++i)
            std::cerr << "mismatch[" << i << "]: actual=" << actual[i] << " expected=" << expected[i] << '\n';
        throw std::runtime_error("CPU reference mismatch; remaining cases skipped");
    }
    if (o.iterations) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < o.iterations; ++i) launch();
        q.wait_and_throw();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / o.iterations;
        std::cout << "{\"scope\":\"kernel_only\",\"iterations\":" << o.iterations << ",\"microseconds_per_launch\":" << us << "}" << std::endl;
    }
}
int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-zero-k") {
            check_zero_k();
            return 0;
        }
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { help(); return 0; }
            if (arg == "--run-gpu") o.run = true;
            else if (arg == "--suite") o.suite = true;
            else if (arg == "--short-rows") o.short_rows = true;
            else if (arg == "--require-native-2d") o.require_native_2d = true;
            else if (arg == "--vary-scales") o.vary_scales = true;
            else {
                if (i + 1 == argc) throw std::invalid_argument("Missing value: " + arg);
                const std::string value = argv[++i];
                if (arg == "--type") {
                    o.type = xmx::parse_type(value);
                    o.selected_type = true;
                } else if (arg == "--glu") {
                    if (value == "swiglu") o.glu = xmx::Glu::swiglu;
                    else if (value == "geglu") o.glu = xmx::Glu::geglu;
                    else throw std::invalid_argument("--glu requires swiglu or geglu");
                }
                else if (arg == "--rows") o.rows = std::stoi(value);
                else if (arg == "--cols") o.cols = std::stoi(value);
                else if (arg == "--k") o.k = std::stoi(value);
                else if (arg == "--iterations") o.iterations = std::stoi(value);
                else throw std::invalid_argument("Unknown option: " + arg);
            }
        }
        if (!o.run) { help(); return 2; }
        if (o.iterations < 0 || o.iterations > 100000) throw std::invalid_argument("iterations must be 0..100000");
        if (o.require_native_2d && (!XMX_ENABLE_TRANSPOSED_2D_LOADS || o.suite || o.short_rows || o.glu ||
                                    o.type != xmx::Type::q8_0))
            throw std::invalid_argument("--require-native-2d requires a native 2D build and a single unfused Q8_0 case");
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) { for (auto e : errors) std::rethrow_exception(e); }, sycl::property::queue::in_order{});
        std::cerr << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        if (o.require_native_2d &&
            !q.get_device().get_info<sycl::ext::intel::esimd::info::device::has_2d_block_io_support>()) {
            throw std::invalid_argument("--require-native-2d requires a device with native 2D block I/O");
        }
        if (o.suite) {
            for (auto type : {xmx::Type::q8_0, xmx::Type::q4_k, xmx::Type::q6_k, xmx::Type::q2_k, xmx::Type::q5_k,
                              xmx::Type::q3_k}) {
                if (o.selected_type && type != o.type) continue;
                for (int rows : {16, 17, 128}) for (int cols : {1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 40, 41, 64}) {
                    Options c = o; c.type = type; c.rows = rows; c.cols = cols; run_case(q, c);
                }
                if (o.glu) {
                    Options c = o;
                    c.type = type;
                    c.rows = 17;
                    c.cols = 65;
                    run_case(q, c);
                }
            }
        } else run_case(q, o);
        return 0;
    } catch (const std::exception & error) { std::cerr << "FAIL: " << error.what() << std::endl; return 1; }
}
