from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServicePaths:
    """All filesystem dependencies of one release installation.

    Nothing here is inferred from the former research-repository layout. The
    release root can be copied or installed anywhere as long as its runtime is
    prepared by scripts/install.sh (or the paths are explicitly overridden).
    """

    release_root: Path
    data_dir: Path
    model_dir: Path
    output_dir: Path
    python_executable: Path
    minimax_source_dir: Path
    lightx_source_dir: Path
    turbo_curve_path: Path
    flashvsr_source_dir: Path
    flashvsr_model_dir: Path
    flashvsr_python_executable: Path

    @classmethod
    def defaults(cls, release_root: Path, *, data_dir: Path | None = None) -> "ServicePaths":
        root = release_root.resolve()
        runtime = Path(os.environ.get("H3_SERVE_RUNTIME_DIR", root / "runtime")).resolve()
        return cls(
            release_root=root,
            data_dir=(data_dir or Path(os.environ.get("H3_SERVE_DATA_DIR", root / "data"))).resolve(),
            # Keep the release-local model mount as the public path even when
            # development uses a read-only symlink to the shared weight store.
            # NativeSessionPaths resolves it only at the point of physical
            # access.  This prevents status/provenance from presenting the old
            # comparator checkout as though it were the active service root.
            model_dir=Path(
                os.environ.get("H3_SERVE_MODEL_DIR", root / "models")
            ).expanduser().absolute(),
            output_dir=Path(os.environ.get("H3_SERVE_OUTPUT_DIR", root / "output")).resolve(),
            # Preserve a virtualenv's interpreter symlink. Resolving it to the
            # base interpreter would silently discard that venv's site-packages.
            python_executable=Path(
                os.environ.get("H3_SERVE_PYTHON", sys.executable)
            ).expanduser().absolute(),
            minimax_source_dir=Path(
                os.environ.get(
                    "H3_SERVE_MINIMAX_SOURCE", runtime / "vendor/MiniMax-H3"
                )
            ).resolve(),
            lightx_source_dir=Path(
                os.environ.get(
                    "H3_SERVE_LIGHTX_SOURCE", runtime / "vendor/LightX2V"
                )
            ).resolve(),
            turbo_curve_path=Path(
                os.environ.get(
                    "H3_SERVE_TURBO_CURVE",
                    root / "backends/turbo/custom_node/h3_silu_temb_grid.safetensors",
                )
            ).resolve(),
            flashvsr_source_dir=Path(
                os.environ.get(
                    "H3_SERVE_FLASHVSR_SOURCE", root / "third_party/flashvsr"
                )
            ).resolve(),
            flashvsr_model_dir=Path(
                os.environ.get(
                    "H3_SERVE_FLASHVSR_MODELS",
                    root / "models/upscalers/flashvsr-v1.1",
                )
            ).expanduser().absolute(),
            flashvsr_python_executable=Path(
                os.environ.get(
                    "H3_SERVE_FLASHVSR_PYTHON",
                    runtime / "flashvsr-venv/bin/python",
                )
            ).expanduser().absolute(),
        )
