cmake_minimum_required(VERSION 3.20)
project(xmx_mul_mat VERSION 0.1.0 LANGUAGES CXX)

option(XMX_BUILD_GPU "Build SYCL kernel tests and DPAS probes" ON)
option(XMX_ENABLE_SLM "Enable the experimental SLM/barrier K reduction" OFF)
option(XMX_ENABLE_TRANSPOSED_2D_LOADS "Enable native 2D loads on devices that advertise 2D block I/O" OFF)
option(XMX_ENABLE_GATHER_TRANSPOSE "Load transposed weight and scale tiles with LSC vector gathers" OFF)
set(XMX_DEVICE_ARCH "" CACHE STRING "Optional Intel GPU AOT target, e.g. 12.74.4")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

add_library(xmx_fixtures STATIC src/fixtures.cpp)
target_include_directories(xmx_fixtures PUBLIC include)
target_compile_options(xmx_fixtures PRIVATE -Wall -Wextra -Wpedantic)
add_executable(xmx_cpu_tests tests/cpu_reference.cpp)
target_link_libraries(xmx_cpu_tests PRIVATE xmx_fixtures)
add_test(NAME cpu_reference COMMAND xmx_cpu_tests)

if(XMX_BUILD_GPU)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" XMX_COMPILER_HAS_SYCL)
    if(NOT XMX_COMPILER_HAS_SYCL)
        message(FATAL_ERROR "Use Intel oneAPI icpx, or configure with -DXMX_BUILD_GPU=OFF")
    endif()
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libsycl.so
        OUTPUT_VARIABLE XMX_SYCL_RUNTIME OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${XMX_SYCL_RUNTIME}")
        get_filename_component(XMX_SYCL_RUNTIME_DIR "${XMX_SYCL_RUNTIME}" DIRECTORY)
    endif()
    foreach(name IN ITEMS kernel_tests dpas_probe native_2d_geometry)
        add_executable(xmx_${name} tests/${name}.cpp)
        target_include_directories(xmx_${name} PRIVATE include)
        target_compile_options(xmx_${name} PRIVATE -fsycl)
        target_link_options(xmx_${name} PRIVATE -fsycl)
        if(XMX_SYCL_RUNTIME_DIR)
            set_target_properties(xmx_${name} PROPERTIES BUILD_RPATH "${XMX_SYCL_RUNTIME_DIR}")
        endif()
        if(XMX_DEVICE_ARCH AND NOT name STREQUAL "native_2d_geometry")
            set(xmx_backend_options "-device ${XMX_DEVICE_ARCH}")
            if(name STREQUAL "kernel_tests" AND XMX_DEVICE_ARCH STREQUAL "12.74.4")
                # Propagate the register allocation policy to the offline compiler.
                string(APPEND xmx_backend_options " -options \"-ze-intel-enable-auto-large-GRF-mode\"")
            endif()
            target_compile_options(xmx_${name} PRIVATE -fsycl-targets=spir64_gen)
            target_link_options(xmx_${name} PRIVATE -fsycl-targets=spir64_gen
                -Xsycl-target-backend=spir64_gen "${xmx_backend_options}")
        elseif(name STREQUAL "kernel_tests")
            # Preserve the automatic GRF option in the JIT image build options.
            target_link_options(xmx_${name} PRIVATE -Xs "-ze-intel-enable-auto-large-GRF-mode")
        endif()
    endforeach()
    target_compile_definitions(xmx_kernel_tests PRIVATE
        XMX_ENABLE_SLM=$<BOOL:${XMX_ENABLE_SLM}>
        XMX_ENABLE_TRANSPOSED_2D_LOADS=$<BOOL:${XMX_ENABLE_TRANSPOSED_2D_LOADS}>
        XMX_ENABLE_GATHER_TRANSPOSE=$<BOOL:${XMX_ENABLE_GATHER_TRANSPOSE}>)
    target_link_libraries(xmx_kernel_tests PRIVATE xmx_fixtures)
    # CTest does not launch GPU kernels: the original kernel reset an Arrow Lake-H GPU.
    add_test(NAME kernel_cli_help COMMAND xmx_kernel_tests --help)
    add_test(NAME kernel_rejects_zero_k COMMAND xmx_kernel_tests --check-zero-k)
    add_test(NAME dpas_cli_help COMMAND xmx_dpas_probe --help)
    add_test(NAME native_2d_geometry COMMAND xmx_native_2d_geometry)
    if(XMX_DEVICE_ARCH STREQUAL "12.74.4" AND NOT XMX_ENABLE_TRANSPOSED_2D_LOADS)
        find_package(Python3 3.9 QUIET COMPONENTS Interpreter)
        find_program(XMX_OCLOC ocloc)
        find_program(XMX_OBJCOPY objcopy)
        if(Python3_Interpreter_FOUND AND XMX_OCLOC AND XMX_OBJCOPY)
            if(XMX_ENABLE_GATHER_TRANSPOSE AND NOT XMX_ENABLE_SLM)
                add_test(NAME arrow_lake_isa
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/arrow_lake_isa.py"
                            $<TARGET_FILE:xmx_kernel_tests>)
            else()
                add_test(NAME arrow_lake_fused_scratch
                    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/arrow_lake_isa.py"
                            --fused-only $<TARGET_FILE:xmx_kernel_tests>)
            endif()
        endif()
    endif()
endif()
