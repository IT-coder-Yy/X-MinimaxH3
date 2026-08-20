# Native H3 engine architecture

## Product boundary

The product is a single-GPU MiniMax H3 service, not a general node graph or a
fork of a general inference framework. Web UI and REST clients submit the same
`GenerationSpec`; a serialized GPU worker calls one `NativeBackendManager`.

```text
browser / REST client
        |
aiohttp API + auth + durable queue
        |
NativeBackendManager        owns job identity and safe output path
        |
NativeH3Engine              owns one request and hot engine mode
        |
NativeH3Pipeline
  |-- conditioning          Qwen3-VL text/image representation
  |-- scheduler             deterministic AV noise and sigma plan
  |-- model                 packed H3 DiT + ConvRot + optional LoRA
  |-- runtime               residency + double-buffer block offload
  |-- video/audio VAE       encode anchors and decode target latents
  `-- mux                   validated atomic MP4 output
```

The queue does not know node IDs, workflow JSON, model ports or checkpoint
keys. The engine does not know API users or persistent job IDs. The model graph
does not know HTTP, output paths or queue state.

## Selective upstream reuse

| Concern | Main source | What is retained | What is rejected |
|---|---|---|---|
| H3 mathematical graph | SGLang | fused-QKV naming/shape, indexed modulation, packed sequence, H3-specific fusion seams | distributed/TP runtime, generic registries, CUDA-13 environment |
| 24 GiB execution | LightX2V | eviction-first model phases, two block buffers, H2D prefetch | full runner/config system and incompatible checkpoint loader |
| Pipeline engineering | FastVideo | typed request/state and single-purpose stages | distributed default and whole framework dependency |
| Quant kernels | Comfy Kitchen / verified local SM89 kernels | ConvRot INT8 contract and production kernel binding | ComfyUI model patcher and graph cache |
| Correctness oracle | existing compatibility backend | tensor/output comparison and validated videos | HTTP/workflow runtime in the final package |

The current ComfyUI/Spectrum code is GPL-3.0 and is treated as a behavioral
oracle, not copied into the native implementation. Reused Apache-2.0 design or
code provenance is recorded in `THIRD_PARTY_NOTICES.md` and component docs.

## Two routes, one core

The high-fidelity and Turbo LoRA products are modes of the same H3 core:

- `original`: 20 simple-scheduler points; explicit actual indices and forecast
  policy selected by the quality slider.
- `lora`: current pruned base plus the pinned Turbo LoRA; 4/5/6/8 distilled
  steps selected by the same public quality slider.

They share input normalization, Qwen representation, packed token layout,
ConvRot kernels, VAE implementations, memory lifecycle and media validation.
This avoids maintaining two mathematical implementations that can drift.

## Migration rule

The compatibility backend is removed only after the native 20/0 route passes:

1. checkpoint header and per-layer shape/dtype audits;
2. block-0 and 50-block tensor comparisons using captured identical inputs;
3. video/audio velocity and decoded-media comparisons;
4. T2VA, first, last and first+last conditions;
5. 480p/720p, 5s/15s within 24 GiB;
6. mandatory multi-frame visual gate before numeric metrics;
7. audible stereo track and Human review; SSIM remains diagnostic only.

Until then, `scripts/preflight_native.py` reports the implemented substrate and
also reports `end_to_end_generator_ready: false`. Passing an import test or a
single block test must never be presented as an independently generated video.
