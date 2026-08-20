"""Executable artifact and implementation preflight for conditioning/VAEs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_checkpoint


_ARTIFACTS = {
    "text": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
}


def run_preflight(model_root: Path) -> dict:
    reports = {
        kind: audit_checkpoint(model_root / relative, kind)
        for kind, relative in _ARTIFACTS.items()
    }
    blockers = []
    if reports["text"].valid:
        blockers.append(
            "Qwen graph/SM89 loader for comfy_nvfp4_awq_single_file_v1 is not "
            "yet shipped in conditioning_vae"
        )
    if reports["video_vae"].valid:
        blockers.append(
            "Apache SGLang fused video-VAE graph is audited but not yet vendored "
            "into the standalone release"
        )
    if reports["audio_vae"].valid:
        blockers.append(
            "Apache SGLang raw-weight audio-VAE graph is audited but not yet "
            "vendored into the standalone release"
        )
    return {
        "model_root": str(model_root.resolve()),
        "artifacts_valid": all(report.valid for report in reports.values()),
        "runtime_ready": all(report.valid for report in reports.values()) and not blockers,
        "reports": {kind: report.to_dict() for kind, report in reports.items()},
        "runtime_blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="also fail until standalone model graphs/loaders are shipped",
    )
    args = parser.parse_args(argv)
    result = run_preflight(args.model_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["artifacts_valid"]:
        return 2
    if args.require_runtime and not result["runtime_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
