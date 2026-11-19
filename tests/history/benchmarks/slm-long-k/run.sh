#!/usr/bin/env bash
set -u
if (($# < 5 || $# > 7)); then
    echo "Usage: bash $0 BUILD_DIR OUTPUT_PREFIX ROWS COLUMNS K [ROUNDS] [ITERATIONS]" >&2
    exit 2
fi
binary="$1/xmx_slm_long_k"
prefix="$2"
rows="$3"
columns="$4"
k="$5"
rounds="${6:-9}"
iterations="${7:-32}"
mkdir -p "$(dirname "$prefix")" || exit 1
started="$(date --iso-8601=seconds)"
test "$(cat /sys/class/power_supply/AC/online)" = 1 || exit 1
result=0
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 90s "$binary" --run-gpu "$rows" "$columns" "$k" "$rounds" "$iterations" > "$prefix.jsonl" 2> "$prefix.log" || result=$?
journalctl -k --since "$started" --no-pager > "$prefix.kernel.log" || exit 1
if rg -in 'device[ _-]lost|engine reset|GPU (hang|reset|recovery)|timed.out job|exec queue reset|device coredump|xe.*wedged' "$prefix.log" "$prefix.kernel.log"; then
    exit 1
fi
if ((result)); then
    cat "$prefix.log" >&2
    exit "$result"
fi
test "$(cat /sys/class/power_supply/AC/online)" = 1 || exit 1
test "$(wc -l < "$prefix.jsonl")" = "$((rounds * 2))" || exit 1
printf '%s\n' "$prefix.jsonl"
