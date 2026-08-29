#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${release_root}/scripts/stop.sh" "$@"
