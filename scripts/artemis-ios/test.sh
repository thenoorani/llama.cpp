#!/usr/bin/env bash
# Artemis test step: on-device correctness gate. Next-token top-32 distributions over 1,020 positions of the
# frozen text must match the reference from the original code (mean KL <= 1e-4, top-1 >= 0.99, batched and
# one-token paths).
set -euo pipefail
exec "$(dirname "$0")/run.sh" gate
