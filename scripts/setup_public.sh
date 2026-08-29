#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reuse_env=""
model_dir=""
vendor_dir=""
sparse_build_dir=""
download_models=0
accept_model_license=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

  --reuse-env PATH          Reuse a validated Python environment.
  --model-dir PATH          Reuse an existing MiniMax H3 model store.
  --vendor-dir PATH         Reuse a directory containing MiniMax-H3/ and LightX2V/.
  --sparse-build-dir PATH   Reuse a compatible compiled spas_sage_attn directory.
  --download-models         Download every release-declared weight.
  --accept-model-license    Confirm that you accepted the model publishers' licenses.
  -h, --help                Show this help.

Fresh installation:
  ./setup.sh --download-models --accept-model-license

Reuse an existing installation:
  ./setup.sh --reuse-env /path/to/env --model-dir /path/to/models \
    --vendor-dir /path/to/vendor
EOF
}

while (($#)); do
  case "$1" in
    --reuse-env) reuse_env="${2:?missing path}"; shift 2 ;;
    --model-dir) model_dir="${2:?missing path}"; shift 2 ;;
    --vendor-dir) vendor_dir="${2:?missing path}"; shift 2 ;;
    --sparse-build-dir) sparse_build_dir="${2:?missing path}"; shift 2 ;;
    --download-models) download_models=1; shift ;;
    --accept-model-license) accept_model_license=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "${release_root}/runtime" "${release_root}/data" "${release_root}/output"

if [[ -n "${reuse_env}" ]]; then
  reuse_env="$(readlink -f -- "${reuse_env}")"
  python_bin="${reuse_env}/bin/python"
  [[ -x "${python_bin}" ]] || {
    echo "The reusable environment has no bin/python: ${reuse_env}" >&2
    exit 1
  }
else
  "${release_root}/scripts/install.sh"
  python_bin="${release_root}/runtime/venv/bin/python"
fi

PYTHONPATH="${release_root}/backends/turbo/vendor${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version
import aiohttp, av, comfy_kitchen, diffusers, einops, numpy, safetensors, torch
import transformers
print(f"Validated Python {sys.version.split()[0]}, Torch {torch.__version__}")
PY

if [[ -z "${vendor_dir}" ]]; then
  vendor_dir="${release_root}/runtime/vendor"
fi
vendor_dir="$(readlink -f -- "${vendor_dir}")"
for source_name in MiniMax-H3 LightX2V; do
  [[ -d "${vendor_dir}/${source_name}" ]] || {
    echo "Missing ${vendor_dir}/${source_name}. Run without --reuse-env or pass --vendor-dir." >&2
    exit 1
  }
done

if [[ -z "${sparse_build_dir}" ]]; then
  candidate_sparse="$(dirname "${vendor_dir}")/extensions/sparge-sm89-py310-torch213-cu133"
  if [[ -d "${candidate_sparse}/spas_sage_attn" ]]; then
    sparse_build_dir="${candidate_sparse}"
  fi
fi
if [[ -n "${sparse_build_dir}" ]]; then
  sparse_build_dir="$(readlink -f -- "${sparse_build_dir}")"
  if ! PYTHONPATH="${sparse_build_dir}:${release_root}/backends/turbo/vendor${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -c 'import spas_sage_attn' >/dev/null 2>&1; then
    echo "Incompatible sparse-attention build: ${sparse_build_dir}" >&2
    exit 1
  fi
fi

if [[ -z "${model_dir}" ]]; then
  model_dir="${release_root}/models"
fi
mkdir -p "${model_dir}"
model_dir="$(readlink -f -- "${model_dir}")"

if ((download_models)); then
  ((accept_model_license)) || {
    echo "--download-models requires --accept-model-license" >&2
    exit 2
  }
  "${python_bin}" "${release_root}/scripts/download_models.py" \
    "${model_dir}" --accept-model-license
fi

config_path="${release_root}/.env.local"
{
  printf 'export H3_SERVE_PYTHON=%q\n' "${python_bin}"
  printf 'export H3_SERVE_MODEL_DIR=%q\n' "${model_dir}"
  printf 'export H3_SERVE_MINIMAX_SOURCE=%q\n' "${vendor_dir}/MiniMax-H3"
  printf 'export H3_SERVE_LIGHTX_SOURCE=%q\n' "${vendor_dir}/LightX2V"
  if [[ -n "${sparse_build_dir}" ]]; then
    printf 'export H3_NATIVE_SPARGE_BUILD_DIR=%q\n' "${sparse_build_dir}"
    printf 'export H3_NATIVE_ENABLE_SPARSE=1\n'
  fi
  printf 'export H3_SERVE_DATA_DIR=%q\n' "${release_root}/data"
  printf 'export H3_SERVE_OUTPUT_DIR=%q\n' "${release_root}/output"
} > "${config_path}"
chmod 600 "${config_path}"

echo "Configuration written to ${config_path}"
# shellcheck disable=SC1090
source "${config_path}"
PYTHONPATH="${sparse_build_dir:+${sparse_build_dir}:}${release_root}/backends/turbo/vendor:${release_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" "${release_root}/scripts/quick_check.py"
echo
echo "Setup complete. Start with: ./run.sh"
