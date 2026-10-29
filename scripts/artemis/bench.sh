#!/usr/bin/env bash
# Benchmark Qwen3-Coder-30B-A3B prompt processing and generation speed; writes artemis_results.json.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${ARTEMIS_MODEL:-/Users/victornoorani/Projects/Turin_Tech/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
OUT=artemis_results.json
rm -f "$OUT" artemis_results.csv

# battery or Low Power Mode roughly halves GPU throughput; refuse rather than record misleading numbers
if ! pmset -g batt | grep -q "AC Power"; then
    echo "not on AC power; refusing to benchmark"; exit 1
fi
if pmset -g | grep -Eq '^[[:space:]]*(lowpowermode|powermode)[[:space:]]+1$'; then
    echo "Low Power Mode is on; refusing to benchmark"; exit 1
fi

pp=$(mktemp); tg=$(mktemp)
trap 'rm -f "$pp" "$tg"' EXIT
# -lm none: read weights into memory instead of mmap, so idle-time page eviction cannot stall the GPU
./build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -lm none -p 512,2048 -n 0 -r 10 -o json > "$pp"
./build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -lm none -p 0 -n 128 -r 3 -o json > "$tg"

/usr/bin/python3 -I - "$OUT" "$pp" "$tg" <<'EOF'
import json, os, statistics, sys
rows = [r for f in sys.argv[2:] for r in json.load(open(f))]
res = {}
for r in rows:
    key = f"pp{r['n_prompt']}_tps" if r["n_gen"] == 0 else f"tg{r['n_gen']}_tps"
    s = r["samples_ts"]
    res[key] = round(statistics.median(s), 3)
    res[key.replace("_tps", "_cv_pct")] = round(100 * statistics.stdev(s) / statistics.mean(s), 3) if len(s) > 1 else 0.0
assert {"pp512_tps", "pp2048_tps", "tg128_tps"} <= res.keys(), res
tmp = sys.argv[1] + ".tmp"
json.dump(res, open(tmp, "w"))
os.replace(tmp, sys.argv[1])
print(res)
EOF
