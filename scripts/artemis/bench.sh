#!/usr/bin/env bash
# Benchmark Qwen3-Coder-30B-A3B prompt processing and generation speed; writes artemis_results.json.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${ARTEMIS_MODEL:-/Users/victornoorani/Projects/Turin_Tech/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
OUT=artemis_results.json
rm -f "$OUT" artemis_results.csv

raw=$(mktemp)
trap 'rm -f "$raw"' EXIT
./build/bin/llama-bench -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 5 -o json > "$raw"

/usr/bin/python3 -I - "$raw" "$OUT" <<'EOF'
import json, os, sys
rows = json.load(open(sys.argv[1]))
res = {}
for r in rows:
    key = f"pp{r['n_prompt']}_tps" if r["n_gen"] == 0 else f"tg{r['n_gen']}_tps"
    res[key] = round(r["avg_ts"], 3)
    res[key.replace("_tps", "_tps_stddev")] = round(r["stddev_ts"], 3)
assert "pp512_tps" in res and "tg128_tps" in res, res
tmp = sys.argv[2] + ".tmp"
json.dump(res, open(tmp, "w"))
os.replace(tmp, sys.argv[2])
print(res)
EOF
