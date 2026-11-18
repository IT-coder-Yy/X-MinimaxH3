"""Split the streamed INT8 DiT QKV projection and attention along whole heads."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from copy import copy

import torch
from torch import nn

active_head_parallel = ContextVar("h3_active_head_parallel", default=None)


class HeadParallel:
    def __init__(self, devices):
        self.devices = devices
        self.shards = {}
        self.calls = 0
        self.pool = ThreadPoolExecutor(max_workers=len(devices) - 1)

    def __enter__(self):
        self.token = active_head_parallel.set(self)
        return self

    def __exit__(self, *exc):
        active_head_parallel.reset(self.token)
        self.pool.shutdown()
        self.shards.clear()
        torch.cuda.empty_cache()

    def run(self, attention, physical, residual, hidden, frequencies, gate, segments, options):
        from .kernels import current_attention_layer
        from .layers import FusedQKVAttention, _gated_residual_slice_
        from .lora import RuntimeLoRALinear
        from .quantization import ConvRotInt8Linear

        base = attention.qkv_proj
        if isinstance(base, RuntimeLoRALinear):
            if base.update.enabled:
                return False
            base = base.base
        layer = current_attention_layer()
        if layer is None or not isinstance(base, ConvRotInt8Linear):
            return False
        head_count = attention.num_heads // len(self.devices)
        inputs = []
        for rank, device in enumerate(self.devices):
            start, stop = rank * head_count, (rank + 1) * head_count
            key = (layer, rank)
            shard = self.shards.get(key)
            if shard is None:
                width = attention.inner_width
                lo, hi = start * attention.head_dim, stop * attention.head_dim

                def pack(tensor):
                    if tensor is None or tensor.numel() == 1:
                        return None if tensor is None else tensor.to(device)
                    return torch.cat([tensor[i + lo:i + hi].to(device, non_blocking=True)
                                      for i in (0, width, 2 * width)])

                projection = ConvRotInt8Linear(
                    pack(base.qweight), pack(base.scale), pack(base.bias),
                    spec=base.spec, kernel=base.kernel, output_dtype=base.output_dtype,
                )
                shard = FusedQKVAttention(
                    projection, nn.Identity(), num_heads=head_count, head_dim=attention.head_dim,
                    qk_eps=attention.q_norm.eps, device=device, dtype=attention.q_norm.weight.dtype,
                )
                shard.q_norm.weight.copy_(attention.q_norm.weight, non_blocking=True)
                shard.k_norm.weight.copy_(attention.k_norm.weight, non_blocking=True)
                if rank:  # The primary also needs space for the rest of the full DiT block.
                    self.shards[key] = shard
            backend = copy(physical)
            for field in ("topk", "maximum_selected_key_blocks"):
                policy = getattr(physical, field, None)
                if isinstance(policy, tuple):
                    setattr(backend, field, policy[start:stop])
            shard.backend = backend
            inputs.append((shard, hidden.to(device, non_blocking=True),
                           frequencies.to(device, non_blocking=True) if frequencies is not None else None,
                           torch.cuda.current_stream(device)))

        def run_rank(shard, local, freqs, stream):
            with torch.inference_mode(), torch.cuda.device(local.device), torch.cuda.stream(stream):
                chunks = []
                shard.stream_gated_residual_(local, local, freqs, None, segments,
                                             collect=chunks.append, **options)
                return chunks

        pending = [self.pool.submit(copy_context().run, run_rank, *args) for args in inputs[1:]]
        try:
            outputs = [run_rank(*inputs[0]), *(f.result() for f in pending)]
        finally:
            wait(pending)
        for group in zip(*outputs, strict=True):
            start, stop, _ = group[0]
            full = torch.cat([part.to(hidden.device, non_blocking=True) for _, _, part in group], dim=-1)
            _gated_residual_slice_(residual, attention.out_proj(full), gate, segments, start=start, stop=stop)
        self.calls += 1
        return True
