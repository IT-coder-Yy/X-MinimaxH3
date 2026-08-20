#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${root}/install.sh" "$@"
for argument in "$@"; do
  [[ "${argument}" == "--dry-run" ]] && exit 0
done
exec "${root}/start.sh"
