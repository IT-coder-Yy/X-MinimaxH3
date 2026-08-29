#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${release_root}/.env.local" ]]; then
  # Generated only by setup.sh and intentionally excluded from Git.
  source "${release_root}/.env.local"
fi
case "${release_root}" in
  /mnt/*) exec "${release_root}/scripts/linux_runtime_exec.sh" "$@" ;;
  *) exec "${release_root}/scripts/start.sh" "$@" ;;
esac
