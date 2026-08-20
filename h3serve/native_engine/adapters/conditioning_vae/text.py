"""Qwen3-VL layer-50 conditioning adapter for MiniMax H3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .audit import audit_checkpoint
from .contracts import TextConditioning
from .preprocess import prepare_keyframes

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


class UnsupportedTextCheckpoint(RuntimeError):
    """Raised when no loader claims the local packed text layout."""


def map_local_qwen_key(key: str) -> str:
    """Map the single-file local prefixes to the Apache SGLang graph."""

    if key.startswith("visual."):
        return "model.visual." + key[len("visual.") :]
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def load_local_qwen_encoder(
    checkpoint: str | Path,
    *,
    build_encoder: Callable[[], Any],
    quantized_loader: Any,
) -> Any:
    """Build and load the audited local encoder through an explicit plugin.

    This function refuses normal ``load_state_dict``.  The checkpoint stores
    packed U8 NVFP4 nibbles, FP8 block scales, tensor scales and AWQ
    ``pre_quant_scale`` values; treating those tensors as ordinary weights is a
    silent numerical corruption.
    """

    report = audit_checkpoint(checkpoint, "text").require_valid()
    supports = getattr(quantized_loader, "supports_layout", None)
    load = getattr(quantized_loader, "load", None)
    if not callable(supports) or not supports(report.layout) or not callable(load):
        raise UnsupportedTextCheckpoint(
            "local Qwen3-VL requires a loader that explicitly supports "
            f"{report.layout}; normal Transformers/SGLang state_dict loading is unsafe"
        )
    encoder = build_encoder()
    load(encoder, report.path, key_mapper=map_local_qwen_key, selected_layers=50)
    return encoder


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer(text, add_special_tokens=False)
    ids = result["input_ids"] if isinstance(result, dict) else result.input_ids
    return [int(value) for value in ids]


def _special_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or int(value) < 0:
        raise ValueError(f"tokenizer does not define MiniMax H3 token {token!r}")
    return int(value)


def _presentation(
    tokenizer: Any,
    prompt: str,
    image_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    tags: list[int] = []
    for index, count in enumerate(image_token_counts, start=1):
        if count <= 0:
            raise ValueError("Qwen image token count must be positive")
        label = _tokenize(tokenizer, f"<Picture {index}>: ")
        vision = (
            [_special_id(tokenizer, VISION_START)]
            + [_special_id(tokenizer, IMAGE_PAD)] * int(count)
            + [_special_id(tokenizer, VISION_END)]
        )
        ids.extend(label)
        tags.extend([1] * len(label))
        ids.extend(vision)
        tags.extend([0] * len(vision))
    prompt_ids = _tokenize(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("prompt produced no Qwen tokens")
    ids.extend(prompt_ids)
    tags.extend([1] * len(prompt_ids))
    return ids, tags


class H3Qwen3VLConditioner:
    """Turn a normalized service request into H3's layer-50 Qwen payload.

    ``encoder`` must expose ``encode_ids`` with the SGLang-compatible narrow
    signature.  The class intentionally does not own checkpoint construction;
    use :func:`load_local_qwen_encoder` with a layout-aware quantized loader.
    """

    def __init__(self, encoder: Any, tokenizer: Any, processor: Any) -> None:
        encode_ids = getattr(encoder, "encode_ids", None)
        if not callable(encode_ids):
            raise TypeError("Qwen encoder must expose encode_ids()")
        if tokenizer is None:
            raise TypeError("Qwen tokenizer is required")
        if processor is None or not hasattr(processor, "image_processor"):
            raise TypeError("Qwen3-VL processor with image_processor is required")
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.processor = processor

    def encode(self, request: Any) -> TextConditioning:
        import torch

        keyframes = prepare_keyframes(request)
        pixel_values = None
        image_grid_thw = None
        image_token_counts: list[int] = []
        if keyframes:
            images = [item.image for item in keyframes]
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            if int(image_grid_thw.shape[0]) != len(images):
                raise ValueError("Qwen image grid count does not match prepared keyframes")
            merge = int(self.processor.image_processor.merge_size) ** 2
            if merge <= 0:
                raise ValueError("Qwen image processor merge_size must be positive")
            image_token_counts = [
                int(image_grid_thw[index].prod().item()) // merge
                for index in range(len(images))
            ]
        ids_list, tags_list = _presentation(
            self.tokenizer, request.prompt, image_token_counts
        )
        input_ids = torch.tensor(ids_list, dtype=torch.long)
        token_tags = torch.tensor(tags_list, dtype=torch.long)
        kwargs: dict[str, Any] = {}
        if pixel_values is not None:
            kwargs.update(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        hidden = self.encoder.encode_ids(input_ids, **kwargs)
        if hidden.ndim != 2 or tuple(hidden.shape) != (len(ids_list), 5120):
            raise ValueError(
                f"Qwen layer-50 output has shape {tuple(hidden.shape)}, "
                f"expected {(len(ids_list), 5120)}"
            )
        return TextConditioning(
            hidden_states=hidden,
            token_tags=token_tags,
            input_ids=input_ids,
            keyframes=keyframes,
        )


__all__ = [
    "H3Qwen3VLConditioner",
    "UnsupportedTextCheckpoint",
    "load_local_qwen_encoder",
    "map_local_qwen_key",
]
