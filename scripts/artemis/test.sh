#!/usr/bin/env bash
# Correctness gate: Metal ops used by Qwen3-Coder-30B-A3B vs CPU, then end-to-end perplexity on a frozen text.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${ARTEMIS_MODEL:-/Users/victornoorani/Projects/Turin_Tech/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
DATA="${ARTEMIS_DATA:-/Users/victornoorani/Projects/Turin_Tech/bench-data}"
# baseline PPL at 7df8b876b is 2.2599; allow 1% drift
PPL_MAX="${ARTEMIS_PPL_MAX:-2.2825}"

T=./build/bin/test-backend-ops
run_ops() {
    local out
    out=$("$T" test -b MTL0 -j 8 "$@" 2>&1) || { echo "$out" | grep -E 'FAIL' | head -20; return 1; }
    if echo "$out" | grep -q FAIL; then echo "$out" | grep FAIL | head -20; return 1; fi
    echo "ops ok: $* ($(echo "$out" | grep -c OK) cases)"
}
# Q4_K_M weights are q4_K/q6_K; head size is 128
run_ops -o MUL_MAT,MUL_MAT_ID -p 'type_a=(q4_K|q6_K|f16|f32),'
run_ops -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128'
run_ops -o RMS_NORM,ROPE,SOFT_MAX,GLU,GET_ROWS,SET_ROWS,ARGSORT,TOP_K,ADD,MUL,CPY,ADD_ID

ppl=$(./build/bin/llama-perplexity -m "$MODEL" -f "$DATA/ppl-code.txt" -c 512 --chunks 16 -ngl 99 -fa on 2>&1 \
    | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p')
[ -n "$ppl" ] || { echo "perplexity run produced no estimate"; exit 1; }
echo "ppl=$ppl (max $PPL_MAX)"
awk -v p="$ppl" -v m="$PPL_MAX" 'BEGIN { exit !(p <= m) }' || { echo "perplexity regression"; exit 1; }
