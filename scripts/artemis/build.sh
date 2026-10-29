#!/usr/bin/env bash
# Clean Metal build of the binaries used by the Artemis test and benchmark steps.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../.."

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > /dev/null
cmake --build build -j "$(sysctl -n hw.ncpu)" --target llama-bench llama-perplexity test-backend-ops > /dev/null
echo "build ok"
