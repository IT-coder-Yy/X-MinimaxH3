#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(${release_root}/runtime/venv/bin/python -c 'from h3serve import __version__; print(__version__)' 2>/dev/null || sed -n 's/__version__ = "\([^"]*\)"/\1/p' "${release_root}/h3serve/__init__.py")"
destination="${1:-${release_root}/dist}"
archive="${destination}/x-minimaxh3-${version}-linux-x86_64-sm89.tar.gz"

mkdir -p "${destination}"
tar -C "${release_root}" -czf "${archive}" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  --exclude='runtime' --exclude='data' --exclude='output' --exclude='dist' \
  README.md README.zh-CN.md VALIDATION.md RELEASE_MANIFEST.json THIRD_PARTY_NOTICES.md LICENSE SECURITY.md CONTRIBUTING.md \
  setup.sh run.sh stop.sh doctor.sh test.sh .env.example .gitattributes .github \
  pyproject.toml requirements.txt requirements.lock requirements-flashvsr.lock \
  server.py smoke_generation.py .gitignore \
  h3serve static backends benchmarks integrations scripts tests docs patches \
  third_party_licenses models/manifest.json

echo "Built ${archive}"
