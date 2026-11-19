#!/usr/bin/env bash
set -u
if (($# != 2)); then
    echo "Usage: bash $0 BUILD_DIR OUTPUT_PREFIX" >&2
    exit 2
fi
binary="$1/xmx_tail_cache_study"
prefix="$2"
mkdir -p "$(dirname "$prefix")" || exit 1
started="$(date --iso-8601=seconds)"
test "$(cat /sys/class/power_supply/AC/online)" = 1 || exit 1
result=0
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 80s "$binary" --run-gpu 4096 4096 12 8 > "$prefix.jsonl" 2> "$prefix.log" || result=$?
journalctl -k --since "$started" --no-pager > "$prefix.kernel.log" || exit 1
if rg -in 'device[ _-]lost|engine reset|GPU (hang|reset|recovery)|timed.out job|exec queue reset|device coredump|xe.*wedged' "$prefix.log" "$prefix.kernel.log"; then
    exit 1
fi
cat "$prefix.log"
test "$(cat /sys/class/power_supply/AC/online)" = 1 || exit 1
test "$result" = 0 || exit "$result"
test "$(wc -l < "$prefix.jsonl")" = 48 || exit 1
printf '%s\n' "$prefix.jsonl"
