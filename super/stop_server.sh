#!/usr/bin/env bash
# Stop the MyLabs Studio React UI and llama-server.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$APP_DIR/.run"
LLAMA_PID_FILE="$RUN_DIR/llama-server.pid"
UI_PID_FILE="$RUN_DIR/react-ui.pid"
LLAMA_PORT=8001
UI_PORT=7860
QUIET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quiet)
            QUIET=true; shift ;;
        --llm-port)
            LLAMA_PORT="$2"; shift 2 ;;
        --ui-port)
            UI_PORT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() {
    if [[ "$QUIET" != true ]]; then
        echo "$@"
    fi
}

stop_pid() {
    local name="$1"
    local pid_file="$2"

    if [[ ! -f "$pid_file" ]]; then
        return 0
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    rm -f "$pid_file"

    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    log "Stopping $name (PID: $pid)..."
    kill "$pid" 2>/dev/null || true

    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.25
    done

    log "Force stopping $name (PID: $pid)..."
    kill -9 "$pid" 2>/dev/null || true
}

stop_port() {
    local name="$1"
    local port="$2"
    local pids

    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    log "Stopping $name process(es) on port $port: $pids"
    kill $pids 2>/dev/null || true

    for _ in {1..20}; do
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [[ -z "$pids" ]]; then
            return 0
        fi
        sleep 0.25
    done

    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        log "Force stopping process(es) on port $port: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
}

stop_matching_project_processes() {
    local name="$1"
    local pattern="$2"
    local pids

    pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    local matched=()
    local pid cwd
    for pid in $pids; do
        if [[ "$pid" == "$$" ]]; then
            continue
        fi
        cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
        if [[ "$cwd" == "$APP_DIR" ]]; then
            matched+=("$pid")
        fi
    done

    if [[ "${#matched[@]}" -eq 0 ]]; then
        return 0
    fi

    log "Stopping $name project process(es): ${matched[*]}"
    kill "${matched[@]}" 2>/dev/null || true

    for _ in {1..20}; do
        local alive=()
        for pid in "${matched[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        if [[ "${#alive[@]}" -eq 0 ]]; then
            return 0
        fi
        sleep 0.25
    done

    log "Force stopping $name project process(es): ${matched[*]}"
    kill -9 "${matched[@]}" 2>/dev/null || true
}

stop_pid "React/FastAPI UI" "$UI_PID_FILE"
stop_pid "llama-server" "$LLAMA_PID_FILE"
stop_matching_project_processes "React/FastAPI UI" "python app.py"
stop_matching_project_processes "llama-server" "llama-server .*--port $LLAMA_PORT"
stop_port "React/FastAPI UI" "$UI_PORT"
stop_port "llama-server" "$LLAMA_PORT"

log "MyLabs Studio stopped."
