#!/usr/bin/env bash
set -euo pipefail

# Keep Python environments and large model weights on WSL's native ext4
# filesystem. Installing either under /mnt/c is functionally valid but makes
# wheel extraction and model loading dramatically slower on Windows.

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_root="${H3_WINDOWS_STATE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/x-minimaxh3}"

if [[ "${1:-}" == "--state-dir" ]]; then
  [[ $# -ge 3 ]] || { echo "--state-dir requires a path and an action" >&2; exit 2; }
  state_root="$2"
  shift 2
fi

[[ "${state_root}" == /* ]] || {
  echo "Windows/WSL state directory must be an absolute Linux path: ${state_root}" >&2
  exit 2
}

action="${1:-}"
[[ -n "${action}" ]] || { echo "usage: windows-wsl.sh [--state-dir PATH] install|start|stop [args...]" >&2; exit 2; }
shift

mkdir -p "${state_root}"
export H3_SERVE_RUNTIME_DIR="${state_root}/runtime"
export H3_SERVE_MODEL_DIR="${state_root}/models"
# The compact-memory path creates an execution-ordered Qwen cache of roughly
# 15 GiB. Keep it beside the selected runtime and models instead of silently
# consuming the WSL system disk when --state-dir points elsewhere.
export H3_SERVE_LOCAL_MODEL_CACHE="${H3_SERVE_LOCAL_MODEL_CACHE:-${state_root}/cache/checkpoints}"

case "${action}" in
  install) exec "${release_root}/install.sh" "$@" ;;
  start) exec "${release_root}/start.sh" "$@" ;;
  stop) exec "${release_root}/stop.sh" "$@" ;;
  *) echo "unknown Windows/WSL action: ${action}" >&2; exit 2 ;;
esac
