from __future__ import annotations

from pathlib import Path
from typing import Any

from .deployment_profiles import LAUNCHER_DEFINITIONS


DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF_DIT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DIT_W4A8 = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
REF_DIT_W4A8 = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
MODEL_FILES = {
    "dit": ("diffusion_models", DIT),
    "ref_dit": ("diffusion_models", REF_DIT),
    "dit_w4a8": ("diffusion_models", DIT_W4A8),
    "ref_dit_w4a8": ("diffusion_models", REF_DIT_W4A8),
    "text_encoder": ("text_encoders", TEXT_ENCODER),
    "video_vae": ("vae", VIDEO_VAE),
    "audio_vae": ("vae", AUDIO_VAE),
    "lora": ("loras", LORA),
    "latent_upscaler": ("latent_upscale_models", LATENT_UPSCALER),
}

LAUNCHER_ROLES = {
    launcher_id: definition.required_model_roles
    for launcher_id, definition in LAUNCHER_DEFINITIONS.items()
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
    launchers = {
        launcher: {
            "ready": all(files[role]["exists"] for role in roles),
            "required_roles": list(roles),
        }
        for launcher, roles in LAUNCHER_ROLES.items()
    }
    return {
        "ready": all(item["ready"] for item in launchers.values()),
        "any_engine_ready": any(item["ready"] for item in launchers.values()),
        "engines": engines,
        "launchers": launchers,
        "files": files,
    }
