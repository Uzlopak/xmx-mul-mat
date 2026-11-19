// SPDX-License-Identifier: MIT
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace es = sycl::ext::intel::esimd;
namespace matrix = sycl::ext::intel::esimd::xmx;

template <int N, bool Accumulator> static bool probe(sycl::queue & q) {
    auto * a = sycl::malloc_shared<int8_t>(128, q);
    auto * b = sycl::malloc_shared<int8_t>(N * 32, q);
    auto * c = sycl::malloc_shared<int>(4 * N, q);
    if (!a || !b || !c) throw std::runtime_error("Allocation failed");
    for (int i = 0; i < 128; ++i) a[i] = 1;
    for (int i = 0; i < N * 32; ++i) b[i] = 1;
    for (int i = 0; i < 4 * N; ++i) c[i] = 7;
    q.single_task([=]() SYCL_ESIMD_KERNEL {
        es::simd<int8_t, 128> av; av.copy_from(a);
        es::simd<int8_t, N * 32> bv; bv.copy_from(b);
        es::simd<int, 4 * N> result;
        if constexpr (Accumulator) {
            es::simd<int, 4 * N> cv; cv.copy_from(c);
            result = matrix::dpas<8, 4, int>(cv, bv, av);
        } else result = matrix::dpas<8, 4, int>(bv, av);
        result.copy_to(c);
    }).wait_and_throw();
    const int expected = Accumulator ? 39 : 32;
    bool pass = true;
    int first_mismatch = -1;
    for (int i = 0; i < 4 * N; ++i) {
        if (c[i] != expected && first_mismatch < 0) first_mismatch = i;
        pass = pass && c[i] == expected;
    }
    std::cout << "{\"dpas_n\":" << N << ",\"accumulator\":" << (Accumulator ? "true" : "false")
              << ",\"expected\":" << expected << ",\"first_mismatch\":" << first_mismatch;
    if (first_mismatch >= 0) std::cout << ",\"actual\":" << c[first_mismatch];
    std::cout << ",\"passed\":" << (pass ? "true" : "false") << "}" << std::endl;
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return pass;
}
int main(int argc, char ** argv) {
    try {
        bool run = false, accumulator = false;
        int n = 16;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: xmx_dpas_probe --run-gpu [--n 8|16] [--accumulator]\n"
                             "A primitive DPAS correctness probe; no 2D loads or SLM.\n"; return 0;
            }
            if (arg == "--run-gpu") run = true;
            else if (arg == "--accumulator") accumulator = true;
            else if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
            else throw std::invalid_argument("Unknown or incomplete option: " + arg);
        }
        if (!run) { std::cerr << "No GPU launch without --run-gpu. Use --help.\n"; return 2; }
        if (n != 8 && n != 16) throw std::invalid_argument("N must be 8 or 16");
        sycl::queue q(sycl::gpu_selector_v, [](sycl::exception_list errors) { for (auto e : errors) std::rethrow_exception(e); }, sycl::property::queue::in_order{});
        const bool pass = n == 8 ? (accumulator ? probe<8, true>(q) : probe<8, false>(q)) : (accumulator ? probe<16, true>(q) : probe<16, false>(q));
        return pass ? 0 : 1;
    } catch (const std::exception & error) { std::cerr << error.what() << std::endl; return 1; }
}
