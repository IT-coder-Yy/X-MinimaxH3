"""Header-only compatibility audit for the current H3 checkpoint bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import H3CoreConfig
from .quantization import ComfyQuantSpec


@dataclass(frozen=True, slots=True)
class CheckpointAudit:
    path: Path
    compatible: bool
    tensor_count: int
    issues: tuple[str, ...]
    quantized_layer_count: int = 0
    lora_pair_count: int = 0

    def require_compatible(self) -> None:
        if not self.compatible:
            raise ValueError("checkpoint audit failed: " + "; ".join(self.issues))


def _safe_open(path: Path):
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for checkpoint audit") from error
    return safe_open(str(path), framework="pt", device="cpu")


def audit_pruned_convrot_checkpoint(
    path: str | Path, config: H3CoreConfig = H3CoreConfig()
) -> CheckpointAudit:
    """Validate key/shape/quant metadata without reading the 21 GB payload."""

    path = Path(path)
    issues: list[str] = []
    quantized = 0
    with _safe_open(path) as checkpoint:
        keys = set(checkpoint.keys())

        def expect(
            key: str, shape: tuple[int, ...], dtypes: tuple[str, ...] | None = None
        ) -> None:
            if key not in keys:
                issues.append(f"missing {key}")
                return
            actual = tuple(checkpoint.get_slice(key).get_shape())
            if actual != shape:
                issues.append(f"{key}: shape {actual} != {shape}")
            if dtypes is not None:
                actual_dtype = str(checkpoint.get_slice(key).get_dtype())
                if actual_dtype not in dtypes:
                    issues.append(f"{key}: dtype {actual_dtype} not in {dtypes}")

        expect(
            "adaln_t_table",
            (config.curve_grid_size, config.pruned_curve_dim),
            ("F32",),
        )
        plain_tensors = {
            "video_patch_proj.weight": ((config.hidden_size, config.video_patch_dim), ("F32",)),
            "video_patch_proj.bias": ((config.hidden_size,), ("F32",)),
            "audio_patch_proj.weight": ((config.hidden_size, config.audio_channels), ("F32",)),
            "audio_patch_proj.bias": ((config.hidden_size,), ("F32",)),
            "condition_proj.weight": ((config.hidden_size, config.text_dim), ("BF16",)),
            "condition_proj.bias": ((config.hidden_size,), ("BF16",)),
            "rope.inv_freq": ((config.rope_inv_freq_len,), ("F32",)),
            "token_refiner.final_norm.weight": ((config.hidden_size,), ("BF16",)),
            "final_layer.norm.weight": ((config.hidden_size,), ("BF16",)),
            "final_layer.adaln_proj.linear.weight": (
                (config.final_adaln_width, config.pruned_curve_dim),
                ("F16",),
            ),
            "final_layer.adaln_proj.linear.bias": ((config.final_adaln_width,), ("F16",)),
            "final_layer.video_out.weight": ((config.video_patch_dim, config.hidden_size), ("F32",)),
            "final_layer.video_out.bias": ((config.video_patch_dim,), ("F32",)),
            "final_layer.audio_out.weight": ((config.audio_channels, config.hidden_size), ("F32",)),
            "final_layer.audio_out.bias": ((config.audio_channels,), ("F32",)),
        }
        for key, (shape, dtypes) in plain_tensors.items():
            expect(key, shape, dtypes)
        for index in range(config.token_refiner_layers):
            prefix = f"token_refiner.blocks.{index}"
            refiner_tensors = {
                "norm1.weight": (config.hidden_size,),
                "norm2.weight": (config.hidden_size,),
                "attn.q_norm.weight": (config.head_dim,),
                "attn.k_norm.weight": (config.head_dim,),
                "attn.qkv_proj.weight": (3 * config.attention_width, config.hidden_size),
                "attn.out_proj.weight": (config.hidden_size, config.attention_width),
                "mlp.fc1.weight": (2 * config.ffn_size, config.hidden_size),
                "mlp.fc2.weight": (config.hidden_size, config.ffn_size),
            }
            for name, shape in refiner_tensors.items():
                expect(f"{prefix}.{name}", shape, ("BF16",))
        for index in range(config.num_layers):
            prefix = f"blocks.{index}"
            expect(f"{prefix}.norm1.weight", (config.hidden_size,))
            expect(f"{prefix}.norm2.weight", (config.hidden_size,))
            expect(f"{prefix}.attn.q_norm.weight", (config.head_dim,))
            expect(f"{prefix}.attn.k_norm.weight", (config.head_dim,))
            expect(
                f"{prefix}.adaln_proj.linear.weight",
                (config.block_adaln_width, config.pruned_curve_dim),
            )
            expect(f"{prefix}.adaln_proj.linear.bias", (config.block_adaln_width,))
            matrices = {
                "attn.qkv_proj": (3 * config.attention_width, config.hidden_size),
                "attn.out_proj": (config.hidden_size, config.attention_width),
                "mlp.fc1": (2 * config.ffn_size, config.hidden_size),
                "mlp.fc2": (config.hidden_size, config.ffn_size),
            }
            for name, shape in matrices.items():
                layer = f"{prefix}.{name}"
                expect(f"{layer}.weight", shape, ("I8",))
                expect(f"{layer}.weight_scale", (shape[0], 1), ("F32",))
                quant_key = f"{layer}.comfy_quant"
                if quant_key not in keys:
                    issues.append(f"missing {quant_key}")
                    continue
                try:
                    spec = ComfyQuantSpec.decode(checkpoint.get_tensor(quant_key))
                    spec.validate_supported()
                    if not spec.convrot:
                        issues.append(f"{layer}: ConvRot flag is false")
                    elif shape[1] % spec.convrot_groupsize:
                        issues.append(f"{layer}: input width not divisible by ConvRot group")
                    else:
                        quantized += 1
                except Exception as error:
                    issues.append(f"{quant_key}: {error}")
        tensor_count = len(keys)
    return CheckpointAudit(
        path=path,
        compatible=not issues,
        tensor_count=tensor_count,
        issues=tuple(issues),
        quantized_layer_count=quantized,
    )


def audit_larry_lora(
    path: str | Path, config: H3CoreConfig = H3CoreConfig()
) -> CheckpointAudit:
    path = Path(path)
    issues: list[str] = []
    with _safe_open(path) as checkpoint:
        keys = set(checkpoint.keys())
        suffix = ".lora_A.weight"
        modules = sorted(key[: -len(suffix)] for key in keys if key.endswith(suffix))
        expected_pairs = config.num_layers * 5 + config.token_refiner_layers * 4 + 1
        if len(modules) != expected_pairs:
            issues.append(f"LoRA pair count {len(modules)} != {expected_pairs}")
        for module in modules:
            down_key = f"{module}{suffix}"
            up_key = f"{module}.lora_B.weight"
            if up_key not in keys:
                issues.append(f"missing {up_key}")
                continue
            down_shape = tuple(checkpoint.get_slice(down_key).get_shape())
            up_shape = tuple(checkpoint.get_slice(up_key).get_shape())
            if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1]:
                issues.append(f"{module}: invalid low-rank shapes {down_shape}, {up_shape}")
        tensor_count = len(keys)
    return CheckpointAudit(
        path=path,
        compatible=not issues,
        tensor_count=tensor_count,
        issues=tuple(issues),
        lora_pair_count=len(modules),
    )
