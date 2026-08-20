from __future__ import annotations

from pathlib import Path
from typing import Any


DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF_DIT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
FLASHVSR_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
    "posi_prompt.pth",
)


MODEL_FILES = {
    "dit": ("diffusion_models", DIT),
    "ref_dit": ("diffusion_models", REF_DIT),
    "text_encoder": ("text_encoders", TEXT_ENCODER),
    "video_vae": ("vae", VIDEO_VAE),
    "audio_vae": ("vae", AUDIO_VAE),
    "lora": ("loras", LORA),
}

ENGINE_ROLES = {
    "original": ("dit", "text_encoder", "video_vae", "audio_vae"),
    "lora": ("dit", "text_encoder", "video_vae", "audio_vae", "lora"),
    "reference": ("ref_dit", "text_encoder", "video_vae", "audio_vae"),
    "reference_lora": ("ref_dit", "text_encoder", "video_vae", "audio_vae", "lora"),
}


def model_status(model_root: Path) -> dict[str, Any]:
    model_root = model_root.resolve()
    files: dict[str, dict[str, Any]] = {}
    for role, (folder, filename) in MODEL_FILES.items():
        path = model_root / folder / filename
        exists = path.is_file()
        files[role] = {
            "filename": filename,
            "relative_path": str(Path(folder) / filename),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
        }
    engines = {
        engine: {
            "ready": all(files[role]["exists"] for role in roles),
            "required_roles": list(roles),
        }
        for engine, roles in ENGINE_ROLES.items()
    }
    upscaler_files = {
        filename: {
            "relative_path": str(Path("upscalers/flashvsr-v1.1") / filename),
            "exists": (model_root / "upscalers/flashvsr-v1.1" / filename).is_file(),
        }
        for filename in FLASHVSR_FILES
    }
    return {
        "ready": all(item["ready"] for item in engines.values()),
        "any_engine_ready": any(item["ready"] for item in engines.values()),
        "engines": engines,
        "files": files,
        "upscaler": {
            "ready": all(item["exists"] for item in upscaler_files.values()),
            "files": upscaler_files,
        },
    }
