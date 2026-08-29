#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${release_root}/.env.local" ]]; then
  source "${release_root}/.env.local"
fi
python_bin="${H3_SERVE_PYTHON:-${release_root}/runtime/venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Python runtime is missing. Run ./setup.sh first." >&2
  exit 1
fi
export PYTHONPATH="${H3_NATIVE_SPARGE_BUILD_DIR:+${H3_NATIVE_SPARGE_BUILD_DIR}:}${release_root}/backends/turbo/vendor:${release_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${1:-}" == "--full" ]]; then
  shift
  exec "${python_bin}" "${release_root}/scripts/preflight.py" "$@"
fi
exec "${python_bin}" "${release_root}/scripts/quick_check.py" "$@"
