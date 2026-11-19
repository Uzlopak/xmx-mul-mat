#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
int main() {
    sycl::queue q;
    auto * a = sycl::malloc_shared<int8_t>(128, q);
    auto * b = sycl::malloc_shared<int8_t>(NROWS * 32, q);
    auto * c = sycl::malloc_shared<int>(4 * NROWS, q);
    q.single_task([=]() SYCL_ESIMD_KERNEL {
        es::simd<int8_t, 128> av; av.copy_from(a);
        es::simd<int8_t, NROWS * 32> bv; bv.copy_from(b);
#ifdef ACCUMULATOR
        es::simd<int, 4 * NROWS> cv; cv.copy_from(c);
        auto result = xmx::dpas<8, 4, int>(cv, bv, av);
#else
        auto result = xmx::dpas<8, 4, int>(bv, av);
#endif
        result.copy_to(c);
    }).wait();
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
}
