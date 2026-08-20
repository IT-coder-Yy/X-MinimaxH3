#!/usr/bin/env python3
"""Backward-compatible alias for the release doctor."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
doctor = ROOT / "scripts/doctor.py"
runtime_root = Path(os.environ.get("H3_SERVE_RUNTIME_DIR", ROOT / "runtime")).expanduser().resolve()
python = runtime_root / "venv/bin/python"
if not python.is_file():
    raise SystemExit(f"运行环境尚未安装：{python}\n请先运行 ./install.sh")
os.execv(str(python), [str(python), str(doctor), *sys.argv[1:]])
