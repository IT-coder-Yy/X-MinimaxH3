#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${release_root}/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "${release_root}/.env.local"
fi
python_bin="${H3_SERVE_PYTHON:-${release_root}/runtime/venv/bin/python}"
[[ -x "${python_bin}" ]] || {
  echo "Python runtime is missing. Run ./setup.sh first." >&2
  exit 1
}
sparse_path="${H3_NATIVE_SPARGE_BUILD_DIR:-}"

# Product configuration must not leak into tests that deliberately verify
# release-local defaults. Keep only the compiled extension on Python's import
# path; every H3/X_MINIMAXH3 setting is removed from the child process.
unset_args=()
while IFS='=' read -r name _; do
  case "${name}" in
    H3_*|X_MINIMAXH3_*) unset_args+=( -u "${name}" ) ;;
  esac
done < <(env)

cd "${release_root}"
exec env "${unset_args[@]}" \
  PYTHONPATH="${sparse_path:+${sparse_path}:}${release_root}/backends/turbo/vendor:${release_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -m unittest discover -s tests -v "$@"
