#!/usr/bin/env bash
set -euo pipefail

connector_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comfy_root="${1:-${H3_COMFYUI_ROOT:-}}"
if [[ -z "${comfy_root}" ]]; then
  echo "Usage: $0 /path/to/ComfyUI" >&2
  exit 2
fi
comfy_root="$(cd "${comfy_root}" && pwd)"
python_bin="${comfy_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" || ! -f "${comfy_root}/main.py" ]]; then
  echo "Invalid ComfyUI installation: ${comfy_root}" >&2
  exit 2
fi

# This ComfyUI process is only an HTTP client.  Keep it off the 4090 so H3
# Serve retains the full 24GB generation envelope.
exec "${python_bin}" "${comfy_root}/main.py" \
  --cpu \
  --listen "${H3_COMFYUI_HOST:-127.0.0.1}" \
  --port "${H3_COMFYUI_PORT:-8188}" \
  --disable-all-custom-nodes \
  --whitelist-custom-nodes ComfyUI-H3-Serve-Connector \
  "${@:2}"
