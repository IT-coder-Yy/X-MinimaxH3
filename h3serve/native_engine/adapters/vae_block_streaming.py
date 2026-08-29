"""Phase-local Video-VAE block residency for hard 8-GiB execution.

The MiniMax-H3 Video-VAE is a 4.9-GiB FP16 model whose ViT decoder owns most
of the weights.  Spatial tiling bounds activations but does not reduce those
resident weights.  This context evicts a small tail of equal-sized decoder
blocks and streams one block at a time while the remaining decoder stays hot.
It is intentionally restricted to decode phases whose caller evicts the full
VAE afterwards.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def stream_video_vae_decoder_tail(
    model,
    *,
    enabled: bool,
    block_count: int = 12,
    device: str = "cuda:0",
) -> Iterator[dict[str, object]]:
    """Keep the decoder hot except for a bounded streamed block tail."""

    if not enabled:
        yield {"enabled": False, "streamed_blocks": 0}
        return
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    if blocks is None or len(blocks) < block_count:
        raise TypeError("H3 Video-VAE does not expose the expected decoder blocks")
    selected = tuple(blocks[-int(block_count):])
    for block in selected:
        block.to("cpu", non_blocking=False)
    torch.cuda.empty_cache()

    handles = []
    for block in selected:
        def load(module, _inputs, *, _device=device):
            module.to(_device, non_blocking=True)

        def evict(module, _inputs, output):
            module.to("cpu", non_blocking=False)
            return output

        handles.append(block.register_forward_pre_hook(load))
        handles.append(block.register_forward_hook(evict))
    try:
        yield {
            "enabled": True,
            "streamed_blocks": len(selected),
            "resident_blocks": len(blocks) - len(selected),
            "policy": "decoder_tail_single_block_streaming_v1",
        }
    finally:
        for handle in handles:
            handle.remove()


__all__ = ["stream_video_vae_decoder_tail"]
