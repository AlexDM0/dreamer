#!/usr/bin/env bash
# Dreamer launcher for Linux / WSL:  ./run.sh   (or: ./run.sh --config my.yaml)
# Uses only the interpreter in .venv; create it first with ./setup.sh
set -euo pipefail

CONFIG="config.yaml"
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="${2:?--config needs a value}"; shift ;;
        --config=*) CONFIG="${1#*=}" ;;
        -h|--help) sed -n '2,3p' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "No .venv found: run ./setup.sh first (or ./setup.sh --mock for a GPU-free setup)." >&2
    exit 1
fi

PORT=7860
if [ -f "$CONFIG" ]; then
    p="$(sed -n 's/^port:[[:blank:]]*\([0-9][0-9]*\).*/\1/p' "$CONFIG" | head -n 1)"
    if [ -n "$p" ]; then PORT="$p"; fi
else
    echo "No $CONFIG found; using defaults. Run ./setup.sh to create config.yaml, then set admin_pin." >&2
fi

# LAN addresses for the iPad and the parent's phone (best effort).
IPS="$(hostname -I 2>/dev/null || true)"
echo
echo "Dreamer is starting. Open on the iPad / phone:"
found=0
for ip in $IPS; do
    case "$ip" in *:*|127.*|169.254.*) continue ;; esac
    echo "  kid page:  http://$ip:$PORT/      admin: http://$ip:$PORT/admin"
    found=1
done
if [ "$found" = 0 ]; then
    echo "  kid page:  http://<this-machine-ip>:$PORT/      admin: http://<this-machine-ip>:$PORT/admin"
fi
echo "Press Ctrl+C to stop."
echo

export SKETCH_CONFIG="$CONFIG"
exec "$PY" -m server
