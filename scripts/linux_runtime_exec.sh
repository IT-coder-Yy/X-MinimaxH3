#!/usr/bin/env bash
set -euo pipefail

canonical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${canonical_root}/.env.local" ]]; then
  source "${canonical_root}/.env.local"
fi

linux_root="${X_MINIMAXH3_RUNTIME_ROOT:-/root/x-minimaxh3-runtime}"
linux_worktree="${linux_root}/worktree"
python_bin="${H3_SERVE_PYTHON:-${canonical_root}/runtime/venv/bin/python}"

if [[ "${X_MINIMAXH3_SYNC:-1}" == "1" || ! -d "${linux_worktree}/h3serve" ]]; then
  "${canonical_root}/scripts/sync_linux_runtime.sh"
fi
[[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 1; }

mkdir -p \
  "${linux_root}/cache/cuda" \
  "${linux_root}/cache/huggingface" \
  "${linux_root}/cache/pycache" \
  "${linux_root}/cache/torch" \
  "${linux_root}/cache/torchinductor" \
  "${linux_root}/cache/triton" \
  "${linux_root}/cache/xdg" \
  "${linux_root}/tmp"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${linux_root}/cache/cuda}"
export HF_HOME="${HF_HOME:-${linux_root}/cache/huggingface}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${linux_root}/cache/pycache}"
export TORCH_HOME="${TORCH_HOME:-${linux_root}/cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${linux_root}/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${linux_root}/cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${linux_root}/cache/xdg}"
export TMPDIR="${TMPDIR:-${linux_root}/tmp}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export H3_SERVE_PYTHON="${python_bin}"
export H3_SERVE_RUNTIME_DIR="${linux_worktree}/runtime"
export H3_SERVE_DATA_DIR="${H3_SERVE_DATA_DIR:-${canonical_root}/data}"
export H3_SERVE_OUTPUT_DIR="${H3_SERVE_OUTPUT_DIR:-${canonical_root}/output}"
export H3_SERVE_LOCAL_MODEL_CACHE="${H3_SERVE_LOCAL_MODEL_CACHE:-${linux_root}/cache/checkpoints}"
export H3_NATIVE_SPARGE_BUILD_DIR="${H3_NATIVE_SPARGE_BUILD_DIR:-${linux_root}/extensions/sparge-sm89-py310-torch213-cu133}"
if [[ -d "${H3_NATIVE_SPARGE_BUILD_DIR}" ]]; then
  export H3_NATIVE_ENABLE_SPARSE="${H3_NATIVE_ENABLE_SPARSE:-1}"
fi
export PYTHONPATH="${linux_worktree}:${linux_worktree}/backends/turbo/vendor${PYTHONPATH:+:${PYTHONPATH}}"

cd "${linux_worktree}"
if (($# == 0)); then
  set -- bash scripts/start.sh
elif [[ "$1" == -* ]]; then
  set -- bash scripts/start.sh "$@"
fi
case "$1" in
  python) shift; exec "${python_bin}" "$@" ;;
  bash) shift; exec /usr/bin/env bash "$@" ;;
  *) exec "$@" ;;
esac
