#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${H3_SERVE_RUNTIME_DIR:-${release_root}/runtime}"
source "${release_root}/scripts/_process.sh"
serve_port="${H3_SERVE_PORT:-8090}"
pid_file="${H3_SERVE_PID_FILE:-${runtime_root}/h3serve-${serve_port}.pid}"
server_path="${release_root}/server.py"
flashvsr_path="${release_root}/scripts/flashvsr_worker.py"

declare -a server_pids=()
if [[ -s "${pid_file}" ]]; then
  candidate="$(<"${pid_file}")"
  if h3_is_release_server_pid "${candidate}"; then
    server_pids+=("${candidate}")
  fi
fi

# Also find a service started before PID-file support, from another shell, or
# through the same WSL mount with different path casing.
while IFS= read -r candidate; do
  [[ -n "${candidate}" ]] || continue
  already_added=0
  for known in "${server_pids[@]:-}"; do
    [[ "${known}" == "${candidate}" ]] && already_added=1
  done
  [[ "${already_added}" == 1 ]] || server_pids+=("${candidate}")
done < <(h3_find_release_server_pids)

if ((${#server_pids[@]})); then
  echo "Stopping H3 service: ${server_pids[*]}"
  kill -INT "${server_pids[@]}" 2>/dev/null || true
  for _ in {1..40}; do
    live=0
    for candidate in "${server_pids[@]}"; do
      kill -0 "${candidate}" 2>/dev/null && live=1
    done
    [[ "${live}" == 0 ]] && break
    sleep 0.25
  done
  for candidate in "${server_pids[@]}"; do
    if kill -0 "${candidate}" 2>/dev/null; then
      echo "Graceful shutdown timed out; terminating PID ${candidate}." >&2
      kill -TERM "${candidate}" 2>/dev/null || true
    fi
  done
  for _ in {1..40}; do
    live=0
    for candidate in "${server_pids[@]}"; do
      kill -0 "${candidate}" 2>/dev/null && live=1
    done
    [[ "${live}" == 0 ]] && break
    sleep 0.25
  done
fi

# A forced server termination can leave the release-local upscaler daemon.
mapfile -t upscaler_pids < <(pgrep -f -- "${flashvsr_path}" || true)
if ((${#upscaler_pids[@]})); then
  echo "Stopping FlashVSR worker: ${upscaler_pids[*]}"
  kill -TERM "${upscaler_pids[@]}" 2>/dev/null || true
fi

rm -f -- "${pid_file}"
if ss -ltnp "sport = :${serve_port}" 2>/dev/null | grep -q LISTEN; then
  echo "Port ${serve_port} is still occupied by a process outside this release." >&2
  ss -ltnp "sport = :${serve_port}" >&2 || true
  exit 1
fi
echo "H3 service is stopped; port ${serve_port} is free."
