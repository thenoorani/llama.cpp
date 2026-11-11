#!/usr/bin/env bash
# Artemis build step: iOS device xcframework from this checkout (Metal library embedded), then build,
# sign and install the benchmark app on the USB-connected iPhone.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
./build-xcframework.sh ios-device > "${TMPDIR:-/tmp}/xcframework-build.log" 2>&1 \
    || { tail -40 "${TMPDIR:-/tmp}/xcframework-build.log"; echo "xcframework build failed"; exit 1; }
DEVELOPMENT_TEAM="${DEVELOPMENT_TEAM:-63CZA36J6Z}" "$HERE/run.sh" build
