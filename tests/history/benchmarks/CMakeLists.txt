cmake_minimum_required(VERSION 3.20)
project(xmx_benchmark_replay LANGUAGES CXX)

option(XMX_ENABLE_GATHER_TRANSPOSE "Load transposed weights with LSC gathers" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

add_library(xmx_benchmark_fixtures STATIC "${repo_root}/src/fixtures.cpp")
target_include_directories(xmx_benchmark_fixtures PUBLIC "${repo_root}/include")

add_executable(xmx_column_sweep q8-column-sweep/benchmark.cpp)
add_executable(xmx_small_glu_sweep small-glu-tiles/benchmark.cpp)
add_executable(xmx_tail_cache_study q8-tail-cache/benchmark.cpp)
add_executable(xmx_slm_long_k slm-long-k/benchmark.cpp)
add_executable(xmx_benchmark_dpas_probe "${repo_root}/tests/dpas_probe.cpp")

set(benchmarks xmx_column_sweep xmx_small_glu_sweep xmx_tail_cache_study)
# K=16,384 at 4,096 rows requires the same deterministic fixture with a larger size limit.
set(fixture_source "${repo_root}/src/fixtures.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${fixture_source}")
file(READ "${fixture_source}" slm_fixtures)
string(REPLACE "> 16 * 1024 * 1024" "> 64 * 1024 * 1024" slm_fixtures "${slm_fixtures}")
string(REPLACE "Fixture exceeds 16M weights" "Fixture exceeds 64M weights" slm_fixtures "${slm_fixtures}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/slm-fixtures.cpp" CONTENT "${slm_fixtures}" @ONLY)
add_library(xmx_benchmark_slm_fixtures STATIC "${CMAKE_CURRENT_BINARY_DIR}/slm-fixtures.cpp")
target_include_directories(xmx_benchmark_slm_fixtures PUBLIC "${repo_root}/include" PRIVATE "${repo_root}/src")

execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libsycl.so
    OUTPUT_VARIABLE sycl_runtime OUTPUT_STRIP_TRAILING_WHITESPACE)
foreach(target IN ITEMS ${benchmarks} xmx_slm_long_k xmx_benchmark_dpas_probe)
    target_compile_options(${target} PRIVATE -fsycl -fsycl-targets=spir64_gen)
    target_link_options(${target} PRIVATE -fsycl -fsycl-targets=spir64_gen
        -Xsycl-target-backend=spir64_gen
        "-device 12.74.4 -options \"-ze-intel-enable-auto-large-GRF-mode\"")
    if(EXISTS "${sycl_runtime}")
        get_filename_component(sycl_runtime_dir "${sycl_runtime}" DIRECTORY)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "${sycl_runtime_dir}")
    endif()
endforeach()

foreach(target IN ITEMS ${benchmarks} xmx_slm_long_k)
    target_include_directories(${target} PRIVATE snapshot)
    target_compile_definitions(${target} PRIVATE
        XMX_ENABLE_SLM=0 XMX_ENABLE_TRANSPOSED_2D_LOADS=0
        XMX_ENABLE_GATHER_TRANSPOSE=$<BOOL:${XMX_ENABLE_GATHER_TRANSPOSE}>)
endforeach()

foreach(target IN LISTS benchmarks)
    target_link_libraries(${target} PRIVATE xmx_benchmark_fixtures)
endforeach()
target_link_libraries(xmx_slm_long_k PRIVATE xmx_benchmark_slm_fixtures)
