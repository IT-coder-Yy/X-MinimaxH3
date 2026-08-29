#!/usr/bin/env bash
set -euo pipefail

# Copy only release-declared model artifacts to Linux-native storage. The
# Windows model tree remains an untouched rollback source. A .ready marker is
# written only after every byte size and SHA-256 matches models/manifest.json.
serve_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$(readlink -f -- "${serve_root}/models")"
target_root="${H3_LINUX_MODEL_ROOT:-/root/h3-model-store}"
python_bin="${H3_LINUX_PYTHON:-/root/miniconda3/envs/h3serve213cu130/bin/python}"
workers="${H3_MODEL_COPY_WORKERS:-3}"

case "${target_root}" in
  /root/h3-model-store|/root/h3-model-store/*) ;;
  *) echo "Refusing unexpected Linux model root: ${target_root}" >&2; exit 2 ;;
esac

[[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 1; }
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid worker count" >&2; exit 2; }
mkdir -p "${target_root}"
rm -f "${target_root}/.ready"

artifact_list="$(mktemp /tmp/h3-model-artifacts.XXXXXX)"
trap 'rm -f "${artifact_list}"' EXIT
"${python_bin}" - "${serve_root}/models/manifest.json" >"${artifact_list}" <<'PY'
import json
import sys
from pathlib import Path

for artifact in json.loads(Path(sys.argv[1]).read_text())["artifacts"]:
    print(
        artifact["install_path"],
        artifact["bytes"],
        artifact["sha256"],
        sep="|",
    )
PY

export source_root target_root
xargs -P "${workers}" -d '\n' -I '{}' bash -c '
  set -euo pipefail
  IFS="|" read -r install expected_size expected_hash <<< "$1"
  source_path="${source_root}/${install}"
  target_path="${target_root}/${install}"
  [[ -f "${source_path}" ]] || { echo "Missing ${source_path}" >&2; exit 1; }
  mkdir -p "$(dirname "${target_path}")"
  echo "COPY ${install}"
  rsync -L --partial --append-verify --info=name1 "${source_path}" "${target_path}"
  actual_size="$(stat -c %s "${target_path}")"
  actual_hash="$(sha256sum "${target_path}" | cut -d" " -f1)"
  [[ "${actual_size}" == "${expected_size}" ]] || { echo "Size mismatch: ${install}" >&2; exit 1; }
  [[ "${actual_hash}" == "${expected_hash}" ]] || { echo "SHA-256 mismatch: ${install}" >&2; exit 1; }
  echo "VERIFIED ${install}"
' _ '{}' <"${artifact_list}"

cp "${serve_root}/models/manifest.json" "${target_root}/manifest.json"
touch "${target_root}/.ready"
echo "Linux model store ready: ${target_root}"
