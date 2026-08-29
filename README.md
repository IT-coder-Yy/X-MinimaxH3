# X-MinimaxH3

**English** · [简体中文](README.zh-CN.md)

X-MinimaxH3 is a local MiniMax H3 video-generation service optimized for a
single NVIDIA SM89 GPU. It provides one bilingual Web console and REST API for
FL2VA and Ref2VA generation, Base/LoRA hot switching, resource-constrained
execution, checkpoint previews and native H3 second sampling.

> Model weights, user uploads, latent states and generated videos are not
> distributed in this repository.

## Features

- One public control surface: total sampling steps and a continuous `0–100`
  acceleration value.
- Joint Base scheduler for actual DiT evaluations, forecast evaluations and
  per-step/per-layer attention budgets.
- Six isolated launchers: FL2VA and Ref2VA on logical 24GB INT8, 16GB INT8 and
  8GB W4A8 resource profiles.
- Native generation from 360p through 1080p where admitted by the selected
  profile, plus native H3 second sampling up to 1440p on INT8 profiles.
- Text-only, first-frame, last-frame and first+last-frame FL2VA generation.
- Multi-reference Ref2VA with images, videos and independent audio references.
- Larry Turbo and three task-aware LightX2V LoRA profiles.
- Resumable checkpoints with fixed low-cost previews.
- Serial GPU queue, cancellation, task history and one-second hardware
  telemetry.
- Optional ComfyUI HTTP connector that does not load a second H3 model.
- English and Simplified Chinese console and documentation.

## Product video

<p align="center">
  <a href="https://www.youtube.com/watch?v=Bao1nPI0JR8">
    <img src="assets/tutorial/youtube-campaign-en-cover.jpg" width="860" alt="Watch the X-MinimaxH3 product video on YouTube">
  </a>
</p>

<p align="center">
  <strong>▶ X-MinimaxH3 — MiniMax H3 made local</strong><br>
  <sub>English product overview · About 90 seconds</sub><br>
  <a href="https://www.youtube.com/watch?v=Bao1nPI0JR8">Watch on YouTube</a>
</p>

## Measured effect comparison

The following local comparison shows the 720p generation and 1440p native H3
second-sampling result side by side. The divider sweeps across the same source
footage for 5-, 10- and 15-second examples. Measurements were recorded on a
14th Gen Intel Core i9, 128GB RAM and an RTX 4090 24GB using INT8 FL2VA.

<p align="center">
  <a href="https://youtu.be/43gOGaAlwnk">
    <img src="assets/demos/effect-comparison-en-cover.jpg" width="860" alt="Watch the X-MinimaxH3 English effect comparison on YouTube">
  </a><br>
  <strong>▶ Watch the English comparison video on YouTube</strong>
</p>

## Community and feedback

Use [GitHub Discussions](https://github.com/PullMyBoots/X-MinimaxH3/discussions)
for international installation help, hardware compatibility reports,
benchmarks, ComfyUI/API questions and generated-video showcases. For real-time
community chat, join the public
[Telegram group](https://t.me/XMinimaxH3Community). You can also contact the
author directly on WeChat; please include `X-MinimaxH3` in your friend request.

| Join the Telegram community | Contact the author | Join the WeChat group |
|:---:|:---:|:---:|
| <a href="https://t.me/XMinimaxH3Community"><img src="assets/community/telegram-community.png" width="260" alt="X-MinimaxH3 Telegram community QR code"></a> | <img src="assets/community/wechat-contact.jpg" width="260" alt="Author WeChat QR code"> | <img src="assets/community/wechat-group.jpg" width="260" alt="X-MinimaxH3 WeChat group QR code"> |
| [Open the public group](https://t.me/XMinimaxH3Community) | Add `X-MinimaxH3` to the request | An updated QR code will be posted here after the current one expires |

For reproducible bugs, please use
[GitHub Issues](https://github.com/PullMyBoots/X-MinimaxH3/issues) so that the
discussion and resolution remain searchable.

## Validated platform

| Component | Validated configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, SM89 |
| OS | Linux x86-64 / WSL2 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Service build toolkit | CUDA 13.3 |
| Host memory | 64GB effective minimum recommended; more for long/high-resolution jobs |

Other GPU architectures have not been release-validated. The logical 8GB and
16GB routes were tested with hard allocator limits on SM89; a physical card of
the same capacity still requires device-specific validation.

## Quick start

### Fresh installation

This creates the runtime, checks out pinned upstream sources and downloads all
weights declared by `models/manifest.json`:

```bash
git clone https://github.com/PullMyBoots/X-MinimaxH3.git
./setup.sh --download-models --accept-model-license
./run.sh
```

`--accept-model-license` confirms that you have reviewed and accepted the
publishers' model licenses. It does not alter or replace those licenses.

### Reuse an existing installation

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./run.sh
```

The vendor directory must contain `MiniMax-H3/` and `LightX2V/`. The sparse
build argument can be omitted when the compatible extension is in the standard
sibling `extensions/` directory.

Open <http://127.0.0.1:8090>. Stop the service with:

```bash
./stop.sh
```

On WSL2, `./run.sh` automatically mirrors the hot source tree and caches to the
Linux filesystem, avoiding repeated imports and metadata access through
`/mnt/c`.

## Validation

Run a quick installation check, full model/revision preflight and regression
suite with:

```bash
./doctor.sh
./doctor.sh --full
./test.sh
```

The release validation recorded:

- 709 release tests passed, 4 skipped and 0 failed; all 24 ComfyUI connector
  tests also passed;
- exact size and SHA-256 checks passed for all 12 declared model artifacts;
- all six launchers and the SM89 INT8/W4A8 kernel smoke test passed;
- real MP4 generation passed for Base FL2VA, LightX2V FL2VA 4-step and 8-step,
  and LightX2V Ref2VA 4-step.

See [VALIDATION.md](VALIDATION.md) for commands, timings and output hashes.

## Resource profiles

| Profile | Weights | Native first generation | Native H3 second sampling |
|---|---|---|---|
| 24GB | INT8 | up to 1080p × 15s | up to 1440p |
| 16GB | INT8 | experimental up to 1080p × 15s | up to 1440p |
| 8GB | W4A8 | up to 720p × 15s | up to 1080p |

Out-of-envelope jobs are rejected instead of silently switching to another
backend. Resolution, duration and media limits exposed by the active service
are authoritative.

The Settings page exposes a 68–362 frame temporal-context control for native
H3 second sampling. Shorter windows usually reduce per-window DiT latency;
longer windows preserve more motion and identity context. H3 phase alignment,
17-frame overlap, latent crossfade and VRAM-safe shortening remain automatic.

## LoRA profiles

| Profile | Task family | Calibrated steps |
|---|---|---:|
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8, default 6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

FL2VA and Ref2VA LightX2V adapters are task-specific and cannot be
interchanged. The settings page scans compatible files recursively under the
configured model store's `loras/` directory.

## ComfyUI

See the [English ComfyUI guide](integrations/comfyui/README.en.md) or the
[Chinese ComfyUI guide](integrations/comfyui/README.md).

Start X-MinimaxH3 first, select a launcher in its console, and then run:

```bash
./integrations/comfyui/start_comfyui.sh
```

Open <http://127.0.0.1:8188>. Example workflows are provided in
`integrations/comfyui/example_workflows/` in both English and Simplified
Chinese. The connector calls the same 8090
HTTP service and does not allocate another copy of H3 inside ComfyUI.

## Repository layout

```text
h3serve/                 Web/API, queue, scheduler and native H3 runtime
backends/                SM89 kernels and audited narrow binary runtime
static/                  bilingual Web console
integrations/comfyui/    optional connector and example workflows
models/manifest.json     weight provenance, sizes and SHA-256 contract
scripts/                 setup, launch, validation and research utilities
tests/                   unit, contract and runtime regression tests
docs/                    user, deployment and architecture documentation
```

## Documentation

- [English user guide](docs/USER_GUIDE.en.md)
- [English deployment guide](docs/DEPLOYMENT.en.md)
- [中文用户指南](docs/USER_GUIDE.zh-CN.md)
- [中文部署指南](docs/DEPLOYMENT.zh-CN.md)
- [Native engine architecture](docs/NATIVE_ENGINE_ARCHITECTURE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Release validation](VALIDATION.md)

## Security

The default server listens only on `127.0.0.1`. Set a strong
`H3_SERVE_API_KEY` before binding to a non-loopback address. The service does
not provide TLS or multi-tenant isolation; use a trusted reverse proxy for
network deployments. See [SECURITY.md](SECURITY.md).

## Acknowledgements

X-MinimaxH3 builds on important work from the MiniMax H3 community. In
particular, we thank:

- [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
  for the temporal/spatial chunking, overlap and stitching design underlying
  our native second-sampling planner. We adapted this design into a
  ComfyUI-independent runtime, added full-canvas admission, automatic resource
  routing, H3 phase alignment and condition-cache reuse.
- [Comfyui Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  for the learned 3D latent-upscaling architecture and released H3 latent
  upscaler weights used to initialize second sampling.
- [SageAttention](https://github.com/thu-ml/SageAttention) for the quantized
  dense-attention kernels and implementation foundation used by our SM89
  dense Attention path. X-MinimaxH3 adds H3-specific layout, quantization,
  long-sequence stability and scheduler integration around that foundation.

The upstream projects are not affiliated with or responsible for
X-MinimaxH3. Their original licenses and notices remain in force; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This is a **public-source** release, not an open-source license grant. Original
project code is currently all rights reserved. Third-party software and model
artifacts retain their own licenses. Review [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[models/manifest.json](models/manifest.json) before use or redistribution.
