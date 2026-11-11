#!/usr/bin/env bash
# Drive the llama.cpp benchmark app on a USB-connected iPhone.
#
# usage: run.sh build   build the app against LLAMA_DIR's iOS xcframework and install it
#        run.sh ref     score the gate text and save the reference to REF (original code only)
#        run.sh gate    correctness gate against REF; exits non-zero on drift
#        run.sh bench   benchmark; writes numeric metrics to RESULTS
#        run.sh session multi-turn conversation with UI load; metrics to RESULTS, chart in results/
#        run.sh all     build, gate, bench
#
# env: DEVELOPMENT_TEAM (required for build), TT_DIR, LLAMA_DIR, MODEL, TEXT, REF, RESULTS, DEVICE_ID,
#      and app settings passed through: PP_REPS TG_REPS TGD_REPS DEPTH THERMAL_MAX
#      COOL_TIMEOUT_S GATE_MAX_MEAN_KL GATE_MIN_TOP1; UI_DROP_CAP (session, default 15%)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# models and frozen data live outside the repo on the runner machine
TT_DIR="${TT_DIR:-/Users/victornoorani/Projects/Turin_Tech}"
LLAMA_DIR="${LLAMA_DIR:-$(cd "$HERE/../.." && pwd)}"
MODEL="${MODEL:-$TT_DIR/models/Qwen3.5-2B-Q4_K_M.gguf}"
TEXT="${TEXT:-$TT_DIR/bench-data/ppl-code.txt}"
REF="${REF:-$TT_DIR/bench-data/ios-ref/$(basename "$MODEL" .gguf).ref}"
RESULTS="${RESULTS:-$PWD/artemis_results.json}"
BUNDLE_ID=com.victornoorani.llamabench
APP_DIR="$HERE/LlamaBenchApp"

DEVICE_ID="${DEVICE_ID:-$(xcrun devicectl list devices 2>/dev/null | grep iPhone | grep -E 'available|connected' | grep -oE '[0-9A-F]{8}-([0-9A-F]{4}-){3}[0-9A-F]{12}' | head -1 || true)}"
: "${DEVICE_ID:?no available iPhone found}"

mkdir -p "$HERE/results"
STAMP="$(date +%Y%m%d-%H%M%S)"

# Hard link (or replace) a bundled resource so the app always ships the requested file.
stage() {
    local src="$1" dst="$APP_DIR/$2"
    if [[ -z "$src" || ! -f "$src" ]]; then rm -f "$dst"; return; fi
    if [[ ! "$dst" -ef "$src" ]]; then rm -f "$dst"; ln "$src" "$dst" 2>/dev/null || cp "$src" "$dst"; fi
}

build() {
    : "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM to your Apple team id}"
    local log="$HERE/results/build-$STAMP.log"
    rsync -a --delete "$LLAMA_DIR/build-apple/llama.xcframework/" "$HERE/llama.xcframework/"
    stage "$MODEL" model.gguf
    stage "$TEXT" text.txt
    stage "$REF" ref.bin
    # Build for the specific device (hardware UDID, not the CoreDevice id) so free
    # provisioning registers it with the team.
    local udid
    udid="$(xcrun devicectl device info details --device "$DEVICE_ID" 2>/dev/null | awk '/udid:/ {print $NF; exit}')"
    (cd "$HERE" && xcodebuild build \
        -project LlamaBenchApp.xcodeproj -scheme LlamaBench -configuration Release \
        -destination "id=$udid" -derivedDataPath "$HERE/.build" \
        -allowProvisioningUpdates DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM") \
        > "$log" 2>&1 || { grep -E "error:" "$log" | head -20; echo "build failed: $log" >&2; return 1; }
    xcrun devicectl device install app --device "$DEVICE_ID" \
        "$HERE/.build/Build/Products/Release-iphoneos/LlamaBench.app" >> "$log" 2>&1 \
        || { tail -20 "$log"; echo "install failed: $log" >&2; return 1; }
    echo "installed $(basename "$MODEL") (ref: $([[ -f "$REF" ]] && basename "$REF" || echo none))"
}

# Launch the app in MODE $1 and print its ARTEMIS_* lines; fails if the app reported an error.
launch() {
    local log="$HERE/results/$1-$STAMP.log" v
    local -a vars=()
    for v in PP_REPS TG_REPS TGD_REPS DEPTH THERMAL_MAX COOL_TIMEOUT_S GATE_MAX_MEAN_KL GATE_MIN_TOP1 \
             N_GPU_LAYERS GATE_UBATCH TURNS TURN_IN TURN_OUT SESSION_MIN THERMAL_START UI UI_LOAD; do
        [[ -n "${!v:-}" ]] && vars+=("DEVICECTL_CHILD_$v=${!v}")
    done
    env ${vars[@]+"${vars[@]}"} DEVICECTL_CHILD_MODE="$1" \
        xcrun devicectl device process launch --device "$DEVICE_ID" --console --terminate-existing \
        "$BUNDLE_ID" > "$log" 2>&1 || true
    if grep -q 'reason: Locked' "$log"; then
        echo "ARTEMIS_ERROR iPhone is locked; unlock it (and set Auto-Lock to Never while benchmarking)"
        return 1
    fi
    grep -E '^ARTEMIS_(INFO|METRIC|ERROR|TURN)' "$log" || { tail -20 "$log"; echo "no output from app, full log: $log" >&2; return 1; }
    ! grep -q '^ARTEMIS_ERROR' "$log"
}

bench() {
    rm -f "$RESULTS"
    local out
    out="$(launch bench)" || { printf '%s\n' "$out"; return 1; }
    printf '%s\n' "$out"
    printf '%s\n' "$out" | python3 -I -c '
import json, sys
m = {}
for line in sys.stdin:
    if line.startswith("ARTEMIS_METRIC "):
        k, v = line.split(None, 1)[1].strip().split("=", 1)
        m[k] = float(v)
if "pp512_tps" not in m:
    sys.exit("no pp512_tps in app output")
json.dump(m, open(sys.argv[1] + ".tmp", "w"), indent=1)
' "$RESULTS"
    mv "$RESULTS.tmp" "$RESULTS"
    echo "wrote $RESULTS"
}

# Multi-turn session: per-turn timeline to results/session-*.jsonl, summary metrics to RESULTS,
# and a chart next to the timeline.
session() {
    rm -f "$RESULTS"
    local out timeline="$HERE/results/session-$STAMP.jsonl"
    out="$(launch session)" || { printf '%s\n' "$out" | grep -v '^ARTEMIS_TURN'; return 1; }
    printf '%s\n' "$out" | grep -v '^ARTEMIS_TURN'
    printf '%s\n' "$out" | sed -n 's/^ARTEMIS_TURN //p' > "$timeline"
    local -a ui_args=(--ui-drop-cap "${UI_DROP_CAP:-15}")
    [[ "${UI:-1}" == 0 ]] && ui_args=(--no-ui)
    python3 -I "$HERE/session_report.py" "$timeline" --results "$RESULTS" "${ui_args[@]}" \
        --title "$(basename "$MODEL" .gguf) · $([[ -n "${SESSION_MIN:-}" ]] && echo "${SESSION_MIN}-minute session" || echo "${TURNS:-15}-turn conversation")"
}

ref() {
    launch ref
    mkdir -p "$(dirname "$REF")"
    xcrun devicectl device copy from --device "$DEVICE_ID" --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" --source Documents/ref.bin --destination "$REF" > /dev/null
    echo "saved reference to $REF ($(wc -c < "$REF" | tr -d ' ') bytes); run build again to bundle it"
}

case "${1:-}" in
    build) build ;;
    ref)   ref ;;
    gate)  launch gate ;;
    bench) bench ;;
    session) session ;;
    all)   build && launch gate && bench ;;
    *)     sed -n '2,14p' "$0"; exit 2 ;;
esac
