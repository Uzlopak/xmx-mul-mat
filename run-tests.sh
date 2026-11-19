#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_build="$project_root/build"
if [[ "${1:-}" == --build-dir ]]; then project_build="$2"; shift 2; fi
project_setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
if [[ -x "$project_build/xmx_kernel_tests" && -f "$project_setvars" ]]; then
    source "$project_setvars" --force >/dev/null 2>&1
fi
if [[ "${1:-}" == --gpu ]]; then
    shift
    export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
    exec "$project_build/xmx_kernel_tests" --run-gpu "$@"
fi
if (($#)); then echo 'Usage: ./run-tests.sh [--build-dir DIR] [--gpu [--suite | kernel options]]' >&2; exit 2; fi
ctest --test-dir "$project_build" --output-on-failure
