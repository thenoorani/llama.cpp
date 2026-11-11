#!/usr/bin/env bash
# Artemis benchmark step: realistic chat session on the iPhone (15 turns, 250 tokens in / 150 out, growing
# context, 60 Hz UI load). The app waits until the phone is back at thermal "nominal" before starting.
# Writes artemis_results.json at the repository root: session_s, ui_dropped_pct, ttft_ms_p50, tg_tps, ...
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
# the Mac only drives the phone, but the phone charges from it; keep conditions fixed
if ! pmset -g batt | grep -q "AC Power"; then echo "Mac not on AC power; refusing to benchmark"; exit 1; fi
RESULTS="$PWD/artemis_results.json" UI_LOAD="${UI_LOAD:-16}" "$HERE/run.sh" session
