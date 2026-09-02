#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${release_root}/.env" ]]; then
  # Centralized user configuration (host, port, API key, model paths).
  source "${release_root}/.env"
fi
if [[ -f "${release_root}/.env.local" ]]; then
  # Generated only by setup.sh and intentionally excluded from Git.
  source "${release_root}/.env.local"
fi
case "${release_root}" in
  /mnt/*)
    exec "${release_root}/scripts/linux_runtime_exec.sh" "$@"
    ;;
  *)
    # Keep a persistent copy of all service output (including tracebacks)
    # in data/service.log while still streaming it to the terminal.
    mkdir -p "${release_root}/data"
    "${release_root}/scripts/start.sh" "$@" 2>&1 | tee -a "${release_root}/data/service.log"
    ;;
esac
