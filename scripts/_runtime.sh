#!/usr/bin/env bash

# Resolve either a self-contained release runtime or this repository's already
# validated development runtime.  Callers must define release_root first.

h3_sparse_importable() {
  local python_bin="$1"
  local search_root="${2:-}"
  PYTHONPATH="${search_root}${search_root:+:}${PYTHONPATH:-}" \
    "${python_bin}" -c 'import torch; import spas_sage_attn' >/dev/null 2>&1
}

h3_configure_sparse_runtime() {
  local python_bin="$1"
  local sparse_commit="ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
  local runtime_root="${H3_SERVE_RUNTIME_DIR:-${release_root}/runtime}"
  local cache_root="${H3_SPARSE_CACHE_DIR:-${runtime_root}/extensions/sparge-sm89-py310-torch28-cu12}"
  local packaged_root="${release_root}/prebuilt/sparge-sm89-py310-torch28-cu126"
  local source_root="${runtime_root}/vendor/SpargeAttn"
  local legacy_root="/tmp/h3_sparge_sm89"

  # An explicitly disabled optional backend always wins. This is useful for
  # diagnosis and guarantees a dense-only service without deleting a cache.
  if [[ "${H3_NATIVE_ENABLE_SPARSE:-auto}" == "0" ]]; then
    return 0
  fi

  if [[ -n "${H3_NATIVE_SPARGE_BUILD_DIR:-}" ]]; then
    if h3_sparse_importable "${python_bin}" "${H3_NATIVE_SPARGE_BUILD_DIR}"; then
      export H3_NATIVE_ENABLE_SPARSE=1
      return 0
    fi
    echo "H3_NATIVE_SPARGE_BUILD_DIR is not compatible with the selected runtime: ${H3_NATIVE_SPARGE_BUILD_DIR}" >&2
    return 1
  fi

  if h3_sparse_importable "${python_bin}"; then
    export H3_NATIVE_ENABLE_SPARSE=1
    unset H3_NATIVE_SPARGE_BUILD_DIR
    return 0
  fi

  # The GitHub release carries the exact extension used by the validated
  # RTX 4090 runtime.  Prefer it over a machine-local cache or a source build.
  # Importing is the ABI check: an incompatible Python/Torch/CUDA combination
  # is rejected instead of being silently selected.
  if h3_sparse_importable "${python_bin}" "${packaged_root}"; then
    export H3_NATIVE_ENABLE_SPARSE=1
    export H3_NATIVE_SPARGE_BUILD_DIR="${packaged_root}"
    return 0
  fi

  if h3_sparse_importable "${python_bin}" "${cache_root}"; then
    export H3_NATIVE_ENABLE_SPARSE=1
    export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
    return 0
  fi

  # Development-only migration path for checkouts created before the packaged
  # binary was introduced.
  if h3_sparse_importable "${python_bin}" "${legacy_root}"; then
    echo "Caching the validated RTX 4090 sparse-attention extension..." >&2
    mkdir -p "${cache_root}"
    cp -a "${legacy_root}/spas_sage_attn" "${cache_root}/"
    if h3_sparse_importable "${python_bin}" "${cache_root}"; then
      export H3_NATIVE_ENABLE_SPARSE=1
      export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
      return 0
    fi
  fi

  if [[ "${H3_AUTO_BUILD_SPARSE:-1}" != "1" ]]; then
    echo "Sparse attention is unavailable and automatic compilation is disabled." >&2
    return 1
  fi
  if ! command -v git >/dev/null || ! command -v nvcc >/dev/null; then
    echo "Sparse attention needs git and the CUDA nvcc compiler for its one-time build." >&2
    return 1
  fi

  echo "No compatible sparse-attention cache was found; compiling once for RTX 4090 (SM89)..." >&2
  mkdir -p "$(dirname "${source_root}")" "${cache_root}"
  if [[ ! -d "${source_root}/.git" ]]; then
    if ! git clone --filter=blob:none https://github.com/thu-ml/SpargeAttn.git "${source_root}"; then
      echo "Could not download the pinned SpargeAttention source." >&2
      return 1
    fi
  fi
  if ! git -C "${source_root}" fetch --depth 1 origin "${sparse_commit}" \
    || ! git -C "${source_root}" checkout --detach "${sparse_commit}"; then
    echo "Could not select the audited SpargeAttention commit ${sparse_commit}." >&2
    return 1
  fi
  if ! TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="${H3_SPARSE_BUILD_JOBS:-4}" \
    "${python_bin}" -m pip install --upgrade --no-build-isolation --no-deps \
      --target "${cache_root}" "${source_root}"; then
    echo "The optional SM89 sparse-attention build failed." >&2
    return 1
  fi
  if ! h3_sparse_importable "${python_bin}" "${cache_root}"; then
    echo "The compiled sparse-attention extension is incompatible with this runtime." >&2
    return 1
  fi

  export H3_NATIVE_ENABLE_SPARSE=1
  export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
  echo "Sparse attention is ready and will be reused from ${cache_root}." >&2
}

h3_configure_runtime() {
  local runtime_root="${H3_SERVE_RUNTIME_DIR:-${release_root}/runtime}"
  local packaged_python="${runtime_root}/venv/bin/python"
  local vendor_path="${release_root}/backends/turbo/vendor"
  local python_bin=""
  local candidate
  local -a candidates=()

  if [[ -n "${H3_SERVE_PYTHON:-}" ]]; then
    candidates+=("${H3_SERVE_PYTHON}")
  elif [[ -x "${packaged_python}" ]]; then
    candidates+=("${packaged_python}")
  else
    # Development checkout: prefer the environment used for the measured
    # Native H3 runs, then consider the caller's active Python installations.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      candidates+=("${CONDA_PREFIX}/envs/voxcpm/bin/python")
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
      candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs/voxcpm/bin/python")
    fi
    candidates+=(
      "/root/miniconda3/envs/voxcpm/bin/python"
      "$(command -v python3.10 2>/dev/null || true)"
      "$(command -v python3 2>/dev/null || true)"
      "$(command -v python 2>/dev/null || true)"
    )
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if PYTHONPATH="${vendor_path}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${candidate}" -c '
import sys
assert sys.version_info[:2] == (3, 10)
import aiohttp, av, diffusers, einops, numpy, safetensors, torch, transformers
import PIL, comfy_kitchen, sageattention
' >/dev/null 2>&1; then
      python_bin="${candidate}"
      break
    fi
  done

  if [[ -z "${python_bin}" ]]; then
    echo "No compatible H3 Python runtime was found." >&2
    if [[ -n "${H3_SERVE_PYTHON:-}" ]]; then
      echo "H3_SERVE_PYTHON is incomplete: ${H3_SERVE_PYTHON}" >&2
    fi
    echo "Run ${release_root}/scripts/install.sh, or set H3_SERVE_PYTHON to the validated Python 3.10 environment." >&2
    return 1
  fi

  export H3_SERVE_PYTHON="${python_bin}"
  export PYTHONPATH="${vendor_path}${PYTHONPATH:+:${PYTHONPATH}}"

  local packaged_flashvsr_python="${runtime_root}/flashvsr-venv/bin/python"
  if [[ -z "${H3_SERVE_FLASHVSR_PYTHON:-}" && -x "${packaged_flashvsr_python}" ]]; then
    export H3_SERVE_FLASHVSR_PYTHON="${packaged_flashvsr_python}"
  fi

  # This is optional: reuse or compile the locked SM89 extension once. A build
  # failure keeps the service available with exact 100% dense attention.
  if ! h3_configure_sparse_runtime "${python_bin}"; then
    export H3_NATIVE_ENABLE_SPARSE=0
    unset H3_NATIVE_SPARGE_BUILD_DIR
    echo "Continuing with complete dense attention; sparse controls stay locked." >&2
  fi

  export H3_SERVE_MINIMAX_SOURCE="${H3_SERVE_MINIMAX_SOURCE:-${release_root}/runtime_sources/MiniMax-H3}"
  export H3_SERVE_LIGHTX_SOURCE="${H3_SERVE_LIGHTX_SOURCE:-${release_root}/runtime_sources/LightX2V}"
}
