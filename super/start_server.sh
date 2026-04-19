#!/usr/bin/env bash
# Start Nemotron-3-Nano llama-server with TurboQuant KV cache compression.
#
# Model: unsloth/Nemotron-3-Nano-30B-A3B-GGUF (Q8_0, ~34GB)
# Hardware: 2x RTX 3090 (48GB total) → ~14GB leftover for KV cache
#
# Usage:
#   ./start_server.sh                  # TurboQuant: turbo2 V-cache + q8_0 K-cache, 262144 (256K) ctx
#   ./start_server.sh --safe           # Balanced: turbo3 V-cache (better quality), 262144 (256K) ctx
#   ./start_server.sh --baseline       # Mainline mode: f16 cache, 8K context (fallback)
#   ./start_server.sh --ctx-size 1048576 # Full 1M native window (may OOM on startup)
#   ./start_server.sh --autostart on   # Enable auto-start on boot (systemd user service)
#   ./start_server.sh --autostart off  # Disable auto-start on boot
#
# Compression tradeoff: turbo4 (3.8x, safest) < turbo3 (5.1x, balanced) < turbo2 (6.4x, aggressive)
# Max context uses the most aggressive V-cache compression while keeping q8_0 K-cache for
# precision-sensitive attention scoring. On 2x 3090 with Q8_0 Nano (~34GB weights), ~14GB is
# left for KV cache + compute buffers — small batch/ubatch frees another ~1GB for context.
#
# Requires the TurboQuant fork built at ~/llama.cpp-turboquant
# Mainline fallback at ~/llama.cpp.mainline (or ~/llama.cpp)

set -euo pipefail
# Unmatched globs expand to nothing (instead of the literal pattern) so the
# model-path lookup below doesn't fail when e.g. no shard files exist.
shopt -s nullglob

# ── Defaults (max-context TurboQuant) ────────────────────────────────────────
PORT=8001
HOST=0.0.0.0
N_GPU_LAYERS=99
TENSOR_SPLIT="0.5,0.5"
# Smaller batch/ubatch → smaller compute buffer → more room for KV cache.
# For an RAG workload with a single concurrent request this costs little in throughput.
BATCH_SIZE=2048
UBATCH_SIZE=512

# TurboQuant defaults: most aggressive V-cache compression, precision K-cache
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="turbo2"
# Default context: 256K. Starts cleanly on 2x 3090 without tripping the
# pipeline-parallel compute-buffer OOM that 1M causes on first allocation.
# Pass --ctx-size 1048576 to use the full native window if VRAM allows.
CTX_SIZE=262144
# Single-sequence mode dedicates the full KV cache to one request (no splitting
# across concurrent slots). RAG is a one-request-at-a-time workload.
PARALLEL=1

# Paths
TURBOQUANT_SERVER="$HOME/llama.cpp-turboquant/build/bin/llama-server"
MAINLINE_SERVER="$HOME/llama.cpp.mainline/llama-server"
MODEL_DIR="$HOME/models/nemotron-3-nano"

BASELINE=false
SAFE=false
SERVICE_NAME="llama-server"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            BASELINE=true; shift ;;
        --safe)
            SAFE=true; shift ;;
        --ctx-size)
            CTX_SIZE="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --tensor-split)
            TENSOR_SPLIT="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --ubatch-size)
            UBATCH_SIZE="$2"; shift 2 ;;
        --cache-type-k)
            CACHE_TYPE_K="$2"; shift 2 ;;
        --cache-type-v)
            CACHE_TYPE_V="$2"; shift 2 ;;
        --autostart)
            case "${2:-}" in
                on)
                    systemctl --user daemon-reload
                    systemctl --user enable "$SERVICE_NAME"
                    loginctl enable-linger "$(whoami)"
                    echo "Autostart ENABLED. Service will start on boot."
                    echo "  Status:  systemctl --user status $SERVICE_NAME"
                    echo "  Logs:    journalctl --user -u $SERVICE_NAME -f"
                    echo "  Start:   systemctl --user start $SERVICE_NAME"
                    exit 0 ;;
                off)
                    systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
                    echo "Autostart DISABLED."
                    exit 0 ;;
                *)
                    echo "Usage: --autostart on|off"; exit 1 ;;
            esac ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Find model ────────────────────────────────────────────────────────────────
# Prefer a single-file Q8_0 GGUF; fall back to the first shard if split; finally
# any .gguf in the dir. With nullglob enabled, unmatched patterns vanish instead
# of breaking the pipeline under pipefail.
CANDIDATES=(
    "$MODEL_DIR"/Nemotron-3-Nano-30B-A3B-Q8_0.gguf
    "$MODEL_DIR"/*Q8_0*-00001-of-*.gguf
    "$MODEL_DIR"/*.gguf
)
MODEL_PATH=""
for c in "${CANDIDATES[@]}"; do
    if [[ -f "$c" ]]; then MODEL_PATH="$c"; break; fi
done
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: No GGUF found in $MODEL_DIR"
    exit 1
fi

# ── Select server binary ─────────────────────────────────────────────────────
if [[ "$BASELINE" == true ]]; then
    SERVER="$MAINLINE_SERVER"
    CACHE_TYPE_K="f16"
    CACHE_TYPE_V="f16"
    CTX_SIZE=8192
    echo "Mode: BASELINE (mainline llama.cpp, f16 cache, ${CTX_SIZE} ctx)"
elif [[ "$SAFE" == true ]]; then
    SERVER="$TURBOQUANT_SERVER"
    CACHE_TYPE_V="turbo3"
    CTX_SIZE=262144
    echo "Mode: TURBOQUANT-SAFE (-ctk ${CACHE_TYPE_K} -ctv ${CACHE_TYPE_V}, ${CTX_SIZE} ctx, parallel=${PARALLEL})"
else
    SERVER="$TURBOQUANT_SERVER"
    echo "Mode: TURBOQUANT-MAX (-ctk ${CACHE_TYPE_K} -ctv ${CACHE_TYPE_V}, ${CTX_SIZE} ctx, batch=${BATCH_SIZE}/${UBATCH_SIZE}, parallel=${PARALLEL})"
fi

if [[ ! -x "$SERVER" ]]; then
    echo "ERROR: Server binary not found or not executable: $SERVER"
    exit 1
fi

# ── Kill existing server on the same port ─────────────────────────────────────
EXISTING_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Stopping existing server on port $PORT (PID: $EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 2
fi

# ── Launch llama-server in background ────────────────────────────────────────
echo "Model:  $MODEL_PATH"
echo "Server: $SERVER"
echo "Port:   $PORT"
echo ""

"$SERVER" \
    --model "$MODEL_PATH" \
    --alias "nemotron-3-nano" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --tensor-split "$TENSOR_SPLIT" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --flash-attn on \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --port "$PORT" \
    --host "$HOST" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Shutting down llama-server (PID: $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Wait for llama-server health ─────────────────────────────────────────────
echo "Waiting for llama-server to come up on :$PORT..."
for i in {1..120}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited before becoming healthy."
        exit 1
    fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "llama-server is ready."
        break
    fi
    sleep 2
done

# ── Activate rag conda env and launch Gradio app ─────────────────────────────
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rag

echo ""
echo "Starting Gradio app (http://localhost:7860)..."
cd "$APP_DIR"
python app.py
