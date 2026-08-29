# X-MinimaxH3 deployment guide

This guide targets Linux x86-64 and WSL2. The calibrated platform is an RTX
4090 (SM89), Python 3.10, PyTorch 2.13.0+cu130 and CUDA Toolkit 13.3. Model
weights are not distributed in the source repository.

## 1. Hardware and storage

- An NVIDIA SM89 GPU; other architectures are not release-validated.
- At least 64GB of available host memory is recommended. Long or
  high-resolution jobs benefit from more than 64GB.
- Allow about 100GB for the complete INT8/W4A8, encoder, VAE, second-sampling
  and LoRA model set.
- On WSL2, keep the hot runtime mirror and build caches in the Linux
  filesystem. The source may remain on a Windows mount: `./run.sh` mirrors it
  to `/root/x-minimaxh3-runtime` automatically.

## 2. Fresh installation

Verify that Linux/WSL can see the GPU, then install:

```bash
nvidia-smi
git clone <your-repository-url> X-MinimaxH3
cd X-MinimaxH3
./setup.sh --download-models --accept-model-license
```

`--accept-model-license` records only that you have reviewed and accepted the
publishers' licenses; it does not alter them. The installer creates an isolated
environment, pins upstream source revisions, downloads the declared manifest
and writes an untracked `.env.local`.

## 3. Reuse an existing installation

Avoid duplicate downloads by reusing a compatible environment and model store:

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
```

The vendor directory must contain `MiniMax-H3/` and `LightX2V/`. A sparse build
in the standard sibling `extensions/` directory is detected automatically. The expected
model layout and hashes are defined by `models/manifest.json`. Run both the
quick structural check and the full release preflight:

```bash
./doctor.sh
./doctor.sh --full
```

## 4. Start and stop

```bash
./run.sh
# Open http://127.0.0.1:8090
./stop.sh
```

The service listens on localhost by default. Before exposing it to a LAN, set
an API key in `.env.local` and then change the bind address:

```bash
export H3_SERVE_API_KEY='replace-with-a-long-random-secret'
export H3_SERVE_HOST=0.0.0.0
```

Never commit `.env.local`, model weights, user inputs or generated media.

## 5. Resource profiles

The console exposes FL2VA and Ref2VA task families. Each has three isolated
profiles:

- 24GB INT8: highest-throughput tier, with the enabled native-generation range
  and second sampling up to 1440p.
- 16GB INT8: constrained-memory execution, including 1080p native generation
  and 1440p second-sampling capability.
- 8GB W4A8: experimental low-memory tier, native generation up to 720p and
  1080p second sampling.

These are logical allocator budgets, not a promise for every GPU of that
capacity. The 8GB and 16GB paths were validated by hard allocator caps on SM89.
An out-of-envelope job is rejected instead of silently switching backends.

## 6. LoRAs and second sampling

The settings page scans `loras/` inside the configured model store. FL2VA and
Ref2VA LoRAs are not interchangeable; registry metadata enforces the task
family. The manifest declares Larry Turbo and three task-aware LightX2V LoRAs.

Native H3 second sampling uses Base weights, SA Solver and fixed scheduler
defaults to denoise the same conditioned task at a higher resolution. Generate
a low-resolution source card first, then start second sampling from task
history. Users control only 1–8 steps, four redraw-strength levels and the
0–100 acceleration value; temporal/spatial tiling is selected automatically.

## 7. ComfyUI

The included ComfyUI node is a small HTTP connector. It does not load another
H3 model in the ComfyUI process. Start the service on port 8090, then run:

```bash
./integrations/comfyui/start_comfyui.sh
```

Open `http://127.0.0.1:8188`. Example workflows live in
`integrations/comfyui/example_workflows/`. A checkpoint job intentionally
pauses for **Continue** or **Discard**. Disable preview mode when no interactive
checkpoint is wanted.

## 8. Troubleshooting

- **Slow first launch:** large weights are mapped and caches are compiled once.
  Keeping the hot runtime and models on the Linux filesystem avoids WSL mount
  metadata overhead.
- **Port 8090 is occupied:** run `./stop.sh`. It terminates only a process owned
  by this release directory.
- **Full VRAM with low power:** this often means weight/activation transfer,
  CPU conditioning or VAE work. Check the real task phase and run `./doctor.sh`
  if it persists.
- **Missing model:** verify `H3_SERVE_MODEL_DIR` in `.env.local`, then run the
  full preflight.
- **Service exit:** inspect `data/service.log` and the task error reference.
  Remove prompts, media and local absolute paths before filing an issue.

## 9. Release validation

```bash
bash -n setup.sh run.sh stop.sh doctor.sh scripts/*.sh
python -m compileall -q h3serve scripts tests
./test.sh
./doctor.sh --full
./scripts/build_release.sh
```

The exact environment and recorded validation results for this checkout are in
`VALIDATION.md`.
