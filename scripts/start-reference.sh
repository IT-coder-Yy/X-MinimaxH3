#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep Python/Torch temporary files inside the release.  This also makes the
# launcher robust on minimal WSL installations where /tmp is unavailable.
mkdir -p "${release_root}/runtime/tmp"
export TMPDIR="${TMPDIR:-${release_root}/runtime/tmp}"
source "${release_root}/scripts/_runtime.sh"
h3_configure_runtime

exec "${H3_SERVE_PYTHON}" "${release_root}/server.py" \
  --host "${H3_SERVE_HOST:-127.0.0.1}" \
  --port "${H3_SERVE_PORT:-8090}" \
  --data-dir "${H3_SERVE_DATA_DIR:-${release_root}/data/reference}" \
  --memory-profile "${H3_SERVE_MEMORY_PROFILE:-auto}" \
  "$@" \
  --engine reference
