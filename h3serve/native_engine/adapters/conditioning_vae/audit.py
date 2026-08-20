"""Header-only audits for the three conditioning/VAE checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


CheckpointKind = Literal["text", "video_vae", "audio_vae"]


@dataclass(frozen=True, slots=True)
class TensorHeader:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CheckpointAudit:
    path: Path
    kind: CheckpointKind
    tensor_count: int
    layout: str
    upstream_implementation: str
    metadata: dict[str, Any]
    representative_tensors: dict[str, TensorHeader]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> "CheckpointAudit":
        if self.errors:
            raise ValueError(
                f"unsupported {self.kind} checkpoint {self.path.name}: "
                + "; ".join(self.errors)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        value["valid"] = self.valid
        return value


def _safe_open(path: Path):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint audit requires safetensors; install the release dependencies"
        ) from exc
    # `safe_open` parses the index only.  The audit never calls get_tensor(), so
    # the 0.6--15.7 GB payloads are not materialized.
    return safe_open(path, framework="pt", device="cpu")


def _header(handle: Any, key: str) -> TensorHeader | None:
    if key not in handle.keys():
        return None
    tensor = handle.get_slice(key)
    return TensorHeader(str(tensor.get_dtype()), tuple(int(v) for v in tensor.get_shape()))


def _expect(
    handle: Any,
    key: str,
    dtype: str,
    shape: tuple[int, ...],
    errors: list[str],
    found: dict[str, TensorHeader],
) -> None:
    actual = _header(handle, key)
    if actual is None:
        errors.append(f"missing tensor {key}")
        return
    found[key] = actual
    if actual.dtype != dtype or actual.shape != shape:
        errors.append(
            f"{key} is {actual.dtype}{actual.shape}, expected {dtype}{shape}"
        )


def _json_metadata(handle: Any, key: str, errors: list[str]) -> dict[str, Any]:
    raw = (handle.metadata() or {}).get(key)
    if not isinstance(raw, str):
        errors.append(f"missing safetensors metadata {key}")
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON metadata {key}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"metadata {key} must be a JSON object")
        return {}
    return value


def _audit_text(path: Path, handle: Any) -> CheckpointAudit:
    keys = list(handle.keys())
    errors: list[str] = []
    warnings: list[str] = []
    found: dict[str, TensorHeader] = {}
    if len(keys) != 2054:
        errors.append(f"tensor count is {len(keys)}, expected 2054")
    layers = sorted(
        {
            int(key.split(".")[2])
            for key in keys
            if key.startswith("model.layers.") and key.split(".")[2].isdigit()
        }
    )
    if layers != list(range(50)):
        errors.append("language checkpoint must contain exactly layers 0..49")
    counts = {
        suffix: sum(key.endswith(suffix) for key in keys)
        for suffix in (
            "comfy_quant",
            "weight_scale",
            "weight_scale_2",
            "pre_quant_scale",
        )
    }
    expected_counts = {
        "comfy_quant": 351,
        "weight_scale": 351,
        "weight_scale_2": 350,
        "pre_quant_scale": 100,
    }
    if counts != expected_counts:
        errors.append(f"quant sidecar counts are {counts}, expected {expected_counts}")
    _expect(handle, "model.embed_tokens.weight", "I8", (151936, 5120), errors, found)
    _expect(
        handle,
        "model.layers.0.self_attn.q_proj.weight",
        "U8",
        (8192, 2560),
        errors,
        found,
    )
    _expect(
        handle,
        "model.layers.0.self_attn.q_proj.weight_scale",
        "F8_E4M3",
        (8192, 320),
        errors,
        found,
    )
    _expect(
        handle,
        "model.layers.0.mlp.down_proj.pre_quant_scale",
        "BF16",
        (25600,),
        errors,
        found,
    )
    _expect(
        handle,
        "model.layers.49.mlp.down_proj.weight",
        "U8",
        (5120, 12800),
        errors,
        found,
    )
    _expect(
        handle,
        "visual.patch_embed.proj.weight",
        "BF16",
        (1152, 3, 2, 16, 16),
        errors,
        found,
    )
    if any(key.startswith("model.language_model.") for key in keys):
        errors.append("checkpoint unexpectedly uses official language_model prefix")
    warnings.append(
        "standard Transformers/SGLang linear loaders cannot consume packed NVFP4; "
        "a native SM89 quantized-linear loader is still required"
    )
    return CheckpointAudit(
        path=path,
        kind="text",
        tensor_count=len(keys),
        layout="comfy_nvfp4_awq_single_file_v1",
        upstream_implementation="SGLang Qwen3-VL graph + Apache Comfy Kitchen quant primitives",
        metadata={"language_layers": layers, "quant_sidecars": counts},
        representative_tensors=found,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _audit_video(path: Path, handle: Any) -> CheckpointAudit:
    keys = list(handle.keys())
    errors: list[str] = []
    warnings: list[str] = []
    found: dict[str, TensorHeader] = {}
    metadata = _json_metadata(handle, "minimax_h3_video_vae", errors)
    if len(keys) != 562:
        errors.append(f"tensor count is {len(keys)}, expected 562")
    if len(metadata.get("latents_mean") or ()) != 24:
        errors.append("video VAE metadata must contain 24 latent means")
    if len(metadata.get("latents_std") or ()) != 24:
        errors.append("video VAE metadata must contain 24 latent stds")
    if metadata.get("vae_clip_length") != 17 or metadata.get("vae_token_drop") != 3:
        errors.append("video VAE temporal metadata must be clip_length=17/token_drop=3")
    _expect(handle, "decoder.mask_token", "F16", (1, 1, 2048), errors, found)
    _expect(handle, "decoder.register_tokens", "F16", (1, 4, 2048), errors, found)
    _expect(
        handle,
        "decoder.transformer_blocks.0.attn.to_qkv.weight",
        "F16",
        (6144, 2048),
        errors,
        found,
    )
    _expect(handle, "quant_conv.weight", "F16", (48, 48, 1, 1, 1), errors, found)
    _expect(handle, "post_quant_conv.weight", "F16", (24, 24, 1, 1, 1), errors, found)
    warnings.append(
        "FastVideo expects split to_q/to_k/to_v and Diffusers FF keys; use the "
        "SGLang fused H3 VAE graph for this checkpoint"
    )
    return CheckpointAudit(
        path=path,
        kind="video_vae",
        tensor_count=len(keys),
        layout="sglang_fused_h3_video_vae_v1",
        upstream_implementation="SGLang fused MiniMaxH3VideoVAE (Apache-2.0)",
        metadata=metadata,
        representative_tensors=found,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _audit_audio(path: Path, handle: Any) -> CheckpointAudit:
    keys = list(handle.keys())
    errors: list[str] = []
    warnings: list[str] = []
    found: dict[str, TensorHeader] = {}
    metadata = _json_metadata(handle, "minimax_h3_audio_vae", errors)
    if len(keys) != 917:
        errors.append(f"tensor count is {len(keys)}, expected 917")
    if len(metadata.get("latents_mean") or ()) != 32:
        errors.append("audio VAE metadata must contain 32 latent means")
    if len(metadata.get("latents_std") or ()) != 32:
        errors.append("audio VAE metadata must contain 32 latent stds")
    if metadata.get("sample_rate") != 32000 or metadata.get("output_channel") != 2:
        errors.append("audio VAE must declare 32 kHz stereo output")
    _expect(handle, "decoder.conv_pre.weight", "F32", (1024, 2048, 7), errors, found)
    _expect(handle, "dec_in_proj.weight", "F32", (2048, 32, 1), errors, found)
    _expect(handle, "mean_proj.weight", "F32", (32, 32, 1), errors, found)
    if any(key.endswith("weight_g") or key.endswith("weight_v") for key in keys):
        errors.append("audio VAE checkpoint unexpectedly retains weight_norm parameters")
    warnings.append(
        "FastVideo constructs weight_g/weight_v state; the local raw-weight file "
        "needs the SGLang raw-weight graph or remove_weight_norm before loading"
    )
    return CheckpointAudit(
        path=path,
        kind="audio_vae",
        tensor_count=len(keys),
        layout="h3_audio_raw_weight_v1",
        upstream_implementation="SGLang DacAudioVAE raw-weight graph (Apache-2.0)",
        metadata=metadata,
        representative_tensors=found,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def audit_checkpoint(path: str | Path, kind: CheckpointKind) -> CheckpointAudit:
    """Validate one artifact without reading its tensor payloads."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with _safe_open(path) as handle:
        if kind == "text":
            return _audit_text(path, handle)
        if kind == "video_vae":
            return _audit_video(path, handle)
        if kind == "audio_vae":
            return _audit_audio(path, handle)
    raise ValueError(f"unsupported checkpoint kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("text", "video_vae", "audio_vae"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    report = audit_checkpoint(args.path, args.kind)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
