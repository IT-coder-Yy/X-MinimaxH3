#!/usr/bin/env bash
set -euo pipefail

# Keep the Git checkout and user data on Windows while executing import-heavy
# Python/CUDA code from a Linux-native mirror. Model weights are referenced by
# H3_SERVE_MODEL_DIR and are never copied by this command.
canonical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
linux_root="${X_MINIMAXH3_RUNTIME_ROOT:-/root/x-minimaxh3-runtime}"
linux_worktree="${linux_root}/worktree"

[[ "${linux_root}" != "/" && "${linux_root}" != "/root" ]] || {
  echo "Refusing unsafe runtime root: ${linux_root}" >&2
  exit 2
}
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }

mkdir -p "${linux_worktree}" "${linux_root}/cache" "${linux_root}/tmp"
rsync -a --delete-delay \
  --exclude '/.git' \
  --exclude '/.env.local' \
  --exclude '/data' \
  --exclude '/output' \
  --exclude '/models' \
  --exclude '/runtime' \
  --exclude '/tests' \
  --exclude '/docs' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${canonical_root}/" "${linux_worktree}/"

printf '%s\n' "${canonical_root}" > "${linux_worktree}/.h3-release-source"
echo "Linux runtime mirror ready: ${linux_worktree}"
