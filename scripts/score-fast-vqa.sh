#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${script_dir}/.." && pwd)"
runtime_root="${H3_FAST_VQA_ROOT:-${release_root}/runtime/quality/fastvqa}"

if [[ ! -f "${runtime_root}/FAST_VQA_3D_1_1.pth" \
   || ! -f "${runtime_root}/source/fastvqa/models/evaluator.py" \
   || ! -d "${runtime_root}/python/decord" ]]; then
  "${script_dir}/setup_fast_vqa.sh"
fi

python_bin="${H3_SERVE_PYTHON:-}"
if [[ -z "${python_bin}" && -x "${release_root}/runtime/venv/bin/python" ]]; then
  python_bin="${release_root}/runtime/venv/bin/python"
fi
if [[ -z "${python_bin}" && -x "/root/miniconda3/envs/voxcpm/bin/python" ]]; then
  python_bin="/root/miniconda3/envs/voxcpm/bin/python"
fi
if [[ -z "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi

exec "${python_bin}" "${script_dir}/score_fast_vqa.py" "$@"

