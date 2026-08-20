# Release validation

This document records checks performed against this extracted GitHub release
candidate on 2026-08-20. Research-tree results are not substituted for release
installation evidence.

## Clean installation

The full Windows/WSL installation route was executed with a new runtime root on
the WSL native ext4 filesystem:

```bash
./scripts/windows-wsl.sh install --profile full \
  --skip-models --skip-system-packages
```

It created both isolated environments and passed the post-install doctor:

- main runtime: Python 3.10, Torch 2.8.0+cu126, SM89 CUDA extensions;
- FlashVSR runtime: Python 3.11, Torch 2.6.0+cu124, Block Sparse Attention;
- RTX 4090 / compute capability 8.9 detected;
- packaged SpargeAttention and Comfy-Kitchen binaries imported successfully.

A second identical installation completed in 15.28 seconds, exercising the
idempotent repair/reuse path. PowerShell 5 parsed all four Windows entry scripts,
and the PowerShell-to-WSL dry-run completed successfully.

Windows stores environments and downloaded models in
`~/.local/share/x-minimaxh3` by default. This avoids extracting large Python and
CUDA packages through `/mnt/c`; source, workspaces and outputs remain visible in
the Windows project directory.

## Model contract

`models/manifest.json` defines ten files totaling 65.8 GiB. A full offline
verification against the local production weights passed both exact byte size
and SHA-256 for every file:

- FL2VA and Ref2VA ConvRot INT8 DiT;
- Qwen3-VL NVFP4 text/vision encoder;
- video and audio VAEs;
- Turbo LoRA;
- four FlashVSR files.

All nine ModelScope URLs used for the public H3/FlashVSR files returned HTTP
200 during the audit. The pinned LoRA URL was reachable through HF Mirror and
the official Hugging Face endpoint.

## Extracted runtime cold loads

Using only the release-owned source subset, installed environments and the
manifest-matching model root:

| Model family | Host profile selected | Cold graph build |
|---|---|---:|
| FL2VA | `fullspeed` | 59.898 s |
| Ref2VA | `generation_hot` | 67.974 s |

The same flows were then exercised through the public control API:

- FL2VA entered the ready state with 63.34 s model startup;
- unloading returned to cold state in 3.03 s;
- Ref2VA entered the ready state with 60.43 s model startup;
- changing Base/LoRA inside a loaded family returned immediately and did not
  rebuild the family session.

The de-duplicated offline tokenizer directory loaded `Qwen2Tokenizer` and
`Qwen3VLProcessor`, including image and video processors, with network access
disabled.

## End-to-end generation smoke

An extracted-release FL2VA Base request completed the full path—Qwen encoding,
INT8 DiT, video/audio VAE decoding and MP4 muxing—using 5 total steps and
acceleration 100. The 640×352, 22-frame smoke took 8.996 seconds after warm-up.
The resulting 0.917-second MP4 contained:

- H.264 video at 640×352 and 24 FPS;
- stereo AAC audio at 32 kHz;
- no coloured-block corruption or unformed-noise frame in the inspected sample.

This smoke proves packaging and runtime integration. It is deliberately too
short to support a production quality or long-video throughput claim.

## Automated contracts

Run from the release root after installation:

```bash
runtime/venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=integrations/comfyui \
  runtime/venv/bin/python -m unittest discover -s integrations/comfyui/tests -v
runtime/venv/bin/python scripts/verify_release.py
```

Observed results:

- service/runtime contracts: **117/117 passed**;
- ComfyUI connector contracts: **15/15 passed**;
- shell syntax and Windows PowerShell 5 syntax: passed;
- reference-media geometry contracts cover landscape, portrait and panoramic
  inputs and prove proportional short-edge reduction without crop/stretch/pad;
- generation-limit contracts prove that every resolution/aspect-ratio pair can
  be configured independently and that Web, REST API and ComfyUI share the
  same server-side submission ceiling;
- the Windows wrapper rejects relative WSL state paths and places runtime,
  models and the recoverable Qwen checkpoint cache beneath one absolute
  `-WslStateDir`;
- release verifier: passed with ten model contracts and no generated media,
  Python cache, partial download or file above GitHub's 100 MB hard limit.

Reference-media tests explicitly verify that 720P/480P/360P settings only lower
pixel resolution proportionally. They do not crop, stretch, pad, change aspect
ratio/composition, alter video duration/frame rate or upscale smaller inputs.
The Web studio, REST API and ComfyUI connector use the same server-side policy.

## Remaining publication decision

The project owner must choose the license for original X-MinimaxH3 code before
publishing the repository. Until `LICENSE-DECISION-REQUIRED.md` is replaced by a
real `LICENSE`, this directory is an engineering-complete private release
candidate, not a legally complete public distribution.
