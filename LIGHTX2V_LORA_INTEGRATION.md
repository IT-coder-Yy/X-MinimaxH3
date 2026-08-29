# LightX2V LoRA integration

The release supports both its original native-name Larry adapter and the
LightX2V Diffusers-format MiniMax H3 adapters below. The runtime scans
`models/loras` recursively, validates the safetensors header, resolves the
task-specific inference profile, and keeps Base/LoRA switching hot inside one
model session.

| LoRA | Task | Calibrated steps | Video/audio shift | Clock |
| --- | --- | ---: | ---: | --- |
| FL2VA Turbo 8-step v1.0 768p | FL2VA/T2VA | 8 | 6/3 | dual |
| FL2VA Turbo 4-step v1.1 768p | FL2VA/T2VA | 4 | 6/3 | dual |
| Ref2VA Turbo 4-step v0.1 | Ref2VA | 4 | 12/3 | dual |

FL2VA and Ref2VA LoRAs are task-specific and must not be interchanged. The Web
console shows only compatible use at engine load time and automatically moves
the sampling-step control to the selected adapter's calibrated value. The
user-facing acceleration control remains available and only changes this
project's layer/attention compute allocation; it does not introduce forecast
steps into the distilled LoRA trajectory.

## Installation

Run:

```bash
bash scripts/install_lightx2v_loras.sh
```

The script stores the 3.87 GiB of weights under Linux-native
`/root/h3-model-store/loras/lightx2v`, verifies fixed SHA-256 digests, and adds
lightweight links under `models/loras/lightx2v`. Set `H3_MODEL_STORE` before
running the script to use another native-Linux model store.

## Runtime conversion

LightX2V stores 312 LoRA pairs with Diffusers names and separate Q, K and V
updates. The native engine retains its fused QKV base projection. It therefore
maps the checkpoint to 208 native runtime modules while applying all 312 source
pairs exactly to their independent Q/K/V output slices. Alpha/rank scaling is
read from checkpoint metadata. The existing Larry-only AdaLN curve is loaded
only for adapters that actually contain AdaLN updates.

## 2026-08-29 functional validation

All tests used the real INT8 24GB engine, real LoRA tensors, generated video and
audio, and acceleration 75. These short 360p runs are functional smoke tests;
they do not claim to establish the 768p visual-quality ranking.

| Profile | Input | Result | End-to-end |
| --- | --- | --- | ---: |
| FL2VA 4-step v1.1 | 360p, 2 s, 4 steps | H.264 + AAC, succeeded | 9.043 s |
| FL2VA 8-step v1.0 | 360p, 2 s, 8 steps | H.264 + AAC, succeeded | 12.652 s |
| Ref2VA 4-step v0.1 | one reference image, 360p, 2 s, 4 steps | H.264 + AAC, succeeded | 11.254 s |

Validation outputs are retained in `runtime/lora_validation_20260829/`.
