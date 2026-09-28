#!/bin/bash
# Start the dual-resident 125B server for measurement runs.
#
# Usage: MODEL=/path/to/...-00001-of-00002.gguf [OUT=dir] tools/engine-rig/scripts/run-125b.sh <tag> [env ...]
# Example: MODEL=~/models/qwen/...gguf run-125b.sh baseline "GGML_CUDA_FUSION_LOG=1"
set -u
TAG=${1:?usage: run-125b.sh <tag> [env ...]}
shift || true

MODEL=${MODEL:?set MODEL=/path/to/shard-1.gguf}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-$REPO/tools/engine-rig/logs}
LOG=$OUT/server-$TAG.log
mkdir -p "$OUT"

if pgrep -f "build-hip/bin/llama-server" >/dev/null; then
    echo "ERROR: llama-server already running (pid $(pgrep -f 'build-hip/bin/llama-server'))" >&2
    exit 1
fi

cd "$REPO"
env "$@" LLAMA_GRAPH_DUMP=2 \
    nohup ./build-hip/bin/llama-server \
        -m "$MODEL" \
        -ngl 99 -ncmoe 0 -sm layer -ts 0.53,0.47 --device ROCm1,ROCm0 \
        -c 4096 -np 1 -t 8 -lm mmap --lazy-mode on -fa auto \
        --host 127.0.0.1 --port 8090 \
        > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$OUT/server-$TAG.pid.numeric"
echo "server pid $PID, log $LOG"

for i in $(seq 1 180); do
    if grep -q "listening on" "$LOG" 2>/dev/null; then
        echo "ready after ${i}s"
        exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: server died; tail of log:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 1
done
echo "ERROR: timeout waiting for server" >&2
exit 1
