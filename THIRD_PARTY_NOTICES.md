# Third-party notices

X-MinimaxH3 combines or interoperates with third-party software and model
artifacts. This file is a provenance guide, not a substitute for the upstream
license text. Model weights are downloaded separately and are never relicensed
by this repository.

| Component | Purpose | Upstream / license location |
|---|---|---|
| MiniMax H3 | Base model architecture and inference source | `MiniMaxAI/MiniMax-H3`, pinned by `scripts/install.sh`; review the upstream model/source terms |
| LightX2V | H3 runtime integration and LoRA conversion reference | `ModelTC/LightX2V`, Apache-2.0 upstream |
| MiniMax H3 INT8 weights | FL2VA/Ref2VA diffusion, Qwen encoder and VAEs | `Comfy-Org/MiniMax-H3`; exact revisions and hashes in `models/manifest.json` |
| MiniMax H3 W4A8 weights | 8GB FL2VA/Ref2VA profiles | `starsfriday/MiniMax-H3-w4a8`; exact revision and hashes in the manifest |
| Larry Turbo LoRA | Optional accelerated FL2VA profile | `larryvrh/MiniMax-H3-Turbo-Lora`; exact revision and hash in the manifest |
| LightX2V Turbo LoRAs | Optional FL2VA 4/8-step and Ref2VA 4-step profiles | `lightx2v/Minimax-h3-Turbo`; exact revision and hashes in the manifest |
| MMH3 UltimateUpscale | Temporal/spatial second-sampling pieces, overlap and stitching design | [`bbaudio-2025/Comfyui-MMH3-UltimateUpscale`](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale), MIT; pinned evaluation revision `6db8fa5a4e4ca0718d2ea8d08002ea899fe27721`; license copy in `third_party_licenses/MMH3-UltimateUpscale-LICENSE` |
| H3 latent upscaler | Learned 3D latent resize used by native second sampling | [`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler), Apache-2.0; exact weight revision and hash in the manifest |
| SageAttention | Quantized dense-attention acceleration and SM89 implementation foundation | [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention), Apache-2.0; license copy in `third_party_licenses/SageAttention-LICENSE` |
| SpargeAttention | Sparse attention integration | Review the pinned upstream source installed by `scripts/install.sh` |
| Comfy Kitchen 0.2.28 | SM89 INT8/W4A8 kernels; a narrow vendored runtime is included | License and notice in `third_party_licenses/Comfy-Kitchen-*` |
| H3 SiLU/temb grid | 5.3MB deterministic kernel-calibration lookup asset, not a model checkpoint | `backends/turbo/custom_node/h3_silu_temb_grid.safetensors`, SHA-256 recorded in `RELEASE_MANIFEST.json` |
| FastVQA | Optional validation tooling | `third_party_licenses/FasterVQA-LICENSE` |
| ComfyUI connector | Optional HTTP workflow integration | Connector code in `integrations/comfyui`; ComfyUI remains separately licensed upstream |

Additional Python and system dependencies retain their own licenses. Run
`pip-licenses` in the configured environment if a deployment needs a complete
environment-specific software bill of materials.

The MiniMax H3 model license and each weight publisher's repository terms may
restrict commercial use or redistribution. Review them before downloading,
using or redistributing any weight.
