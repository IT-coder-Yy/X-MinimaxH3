#!/usr/bin/env bash

# Process ownership helpers shared by start.sh and stop.sh.  WSL DrvFS paths
# are case-insensitive while /proc/<pid>/cmdline preserves the spelling used by
# the launching shell.  Comparing path strings therefore misclassifies the
# same release as an external process when one shell uses /mnt/c/Users and
# another uses /mnt/c/users.  Device/inode identity is stable across both.

h3_same_file() {
  local left="$1"
  local right="$2"
  [[ -e "${left}" && -e "${right}" ]] || return 1
  [[ "$(stat -Lc '%d:%i' -- "${left}" 2>/dev/null)" == \
     "$(stat -Lc '%d:%i' -- "${right}" 2>/dev/null)" ]]
}

h3_is_release_runtime_mirror() {
  local candidate_server="$1"
  local marker="$(dirname -- "${candidate_server}")/.h3-release-source"
  local source_root
  [[ -f "${marker}" ]] || return 1
  IFS= read -r source_root < "${marker}" || return 1
  h3_same_file "${source_root}" "${release_root}"
}

h3_is_release_server_pid() {
  local pid="$1"
  local target_server_path="${server_path:-${release_root}/server.py}"
  local argument candidate process_cwd
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  while IFS= read -r -d '' argument; do
    case "${argument}" in
      server.py|*/server.py)
        if [[ "${argument}" == /* ]]; then
          candidate="${argument}"
        else
          candidate="${process_cwd}/${argument}"
        fi
        if h3_same_file "${candidate}" "${target_server_path}" \
          || h3_is_release_runtime_mirror "${candidate}"; then
          return 0
        fi
        ;;
    esac
  done < "/proc/${pid}/cmdline" 2>/dev/null
  return 1
}

h3_find_release_server_pids() {
  local process_dir pid
  for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    h3_is_release_server_pid "${pid}" && printf '%s\n' "${pid}"
  done
}
