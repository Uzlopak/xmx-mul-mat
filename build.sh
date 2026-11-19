#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_build="$project_root/build"
project_jobs="${XMX_BUILD_JOBS:-2}"
project_gpu=ON
project_arch=""
while (($#)); do
    case "$1" in
        --cpu-only) project_gpu=OFF; shift ;;
        --build-dir) project_build="$2"; shift 2 ;;
        --jobs) project_jobs="$2"; shift 2 ;;
        --aot) project_arch="$2"; shift 2 ;;
        -h|--help) echo 'Usage: ./build.sh [--cpu-only] [--build-dir DIR] [--jobs N] [--aot 12.74.4]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ "$project_gpu" == ON ]]; then
    if ! command -v icpx >/dev/null; then
        project_setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
        [[ -f "$project_setvars" ]] || { echo 'Intel oneAPI icpx is required. Set ONEAPI_SETVARS or use --cpu-only.' >&2; exit 1; }
        source "$project_setvars" --force >/dev/null 2>&1
    fi
    project_compiler="$(command -v icpx)"
else
    project_compiler="${CXX:-c++}"
fi
cmake -S "$project_root" -B "$project_build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$project_compiler" -DXMX_BUILD_GPU="$project_gpu" -DXMX_DEVICE_ARCH="$project_arch"
cmake --build "$project_build" --parallel "$project_jobs"
ctest --test-dir "$project_build" --output-on-failure
