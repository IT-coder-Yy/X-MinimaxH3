#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "${script_dir}/.." && pwd)"
runtime_root="${H3_FAST_VQA_ROOT:-${release_root}/runtime/quality/fastvqa}"
source_root="${runtime_root}/source"
python_root="${runtime_root}/python"
checkpoint="${runtime_root}/FAST_VQA_3D_1_1.pth"

source_url="https://github.com/VQAssessment/FAST-VQA-and-FasterVQA.git"
source_commit="8db452e2caa5d5d4da507bcf577c19b8114f2ebd"
checkpoint_url="https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth"
checkpoint_size="127343543"
checkpoint_sha256="8c3108647653fd48e31f3bebbe03a344d624c806d3f1af9478a4e9f5aa3038ab"

resolve_python() {
  local candidate
  for candidate in \
    "${H3_SERVE_PYTHON:-}" \
    "${release_root}/runtime/venv/bin/python" \
    "/root/miniconda3/envs/voxcpm/bin/python" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

python_bin="$(resolve_python)" || {
  echo "No suitable Python runtime was found. Set H3_SERVE_PYTHON." >&2
  exit 1
}

command -v git >/dev/null || {
  echo "git is required to install FasterVQA." >&2
  exit 1
}
command -v curl >/dev/null || {
  echo "curl is required to download the FasterVQA checkpoint." >&2
  exit 1
}

mkdir -p "${runtime_root}" "${python_root}"

if [[ ! -d "${source_root}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${source_url}" "${source_root}"
fi
git -C "${source_root}" fetch --depth 1 origin "${source_commit}"
git -C "${source_root}" checkout --detach "${source_commit}"

if ! PYTHONPATH="${python_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -c 'import decord, timm; assert decord.__version__ == "0.6.0"; assert timm.__version__ == "0.6.13"' \
  >/dev/null 2>&1; then
  "${python_bin}" -m pip install --upgrade --no-deps \
    --target "${python_root}" decord==0.6.0 timm==0.6.13
fi

checkpoint_ok=0
if [[ -f "${checkpoint}" ]]; then
  actual_size="$(stat -c '%s' "${checkpoint}")"
  actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  if [[ "${actual_size}" == "${checkpoint_size}" && "${actual_sha256}" == "${checkpoint_sha256}" ]]; then
    checkpoint_ok=1
  fi
fi

if [[ "${checkpoint_ok}" != "1" ]]; then
  rm -f "${checkpoint}.part"
  curl --fail --location --retry 3 --output "${checkpoint}.part" "${checkpoint_url}"
  actual_size="$(stat -c '%s' "${checkpoint}.part")"
  actual_sha256="$(sha256sum "${checkpoint}.part" | awk '{print $1}')"
  if [[ "${actual_size}" != "${checkpoint_size}" || "${actual_sha256}" != "${checkpoint_sha256}" ]]; then
    rm -f "${checkpoint}.part"
    echo "FasterVQA checkpoint verification failed." >&2
    exit 1
  fi
  mv "${checkpoint}.part" "${checkpoint}"
fi

echo "FasterVQA is ready."
echo "  Python:     ${python_bin}"
echo "  Source:     ${source_commit}"
echo "  Checkpoint: ${checkpoint_sha256}"
echo "  Runtime:    ${runtime_root}"

