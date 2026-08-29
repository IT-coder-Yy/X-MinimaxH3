# Release validation

This document records reproducible checks for the public source tree. Generated
videos and model weights are deliberately excluded from Git.

## Validated platform

- OS: Linux x86-64 under WSL2
- GPU: NVIDIA GeForce RTX 4090 (SM89)
- Python: 3.10.20
- PyTorch: 2.13.0+cu130 (CUDA runtime 13.0)
- Service launch toolkit: CUDA 13.3; the host's default `nvcc` symlink remains 12.8
- NVIDIA driver: 610.88
- Runtime model store: external to the repository

## Commands

```bash
./setup.sh --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/pinned/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./doctor.sh
bash -n setup.sh run.sh stop.sh doctor.sh scripts/*.sh
python -m compileall -q h3serve scripts tests
./test.sh
PYTHONPATH=integrations/comfyui python -m unittest \
  integrations/comfyui/tests/test_client.py
./doctor.sh --full
./scripts/build_release.sh
```

## Result ledger — 2026-08-29

- Shell syntax, Python compilation, JSON parsing and browser JavaScript syntax:
  passed.
- Unit/contract/runtime regression suite: **709 passed, 4 skipped, 0 failed**
  in 184.253 seconds.
- ComfyUI connector suite: **24 passed, 0 failed**. Both English workflows
  passed JSON parsing, dynamic input/output schema validation and four-file
  installer discovery alongside their Chinese equivalents.
- Full preflight: all 12 declared weight files passed byte-size and SHA-256
  checks; both pinned upstream revisions, all six launcher profiles and the
  SM89 kernel smoke test passed. `end_to_end_runtime_ready` was `true`.
- Web/API: version 1.0.0 served from the Linux hot mirror on port 8091; health,
  model matrix, LoRA registry and no-cache index response passed.
- Real generation smoke tests (640x352, 56 frames, 24fps):

| Runtime | Steps | Generation time | MP4 SHA-256 |
|---|---:|---:|---|
| Base FL2VA INT8 | 5 | 9.524s | `0ef45b18e329eec8b2d8935436a5f9f5bde9c8261c849e91b79d5db3e453b0dd` |
| LightX2V FL2VA 4-step v1.1 | 4 | 8.784s | `d746a7ff4370ea4d717b31b71e84b34ab6f01def7c92b7d0388e12372d62a0c6` |
| LightX2V FL2VA 8-step v1.0 | 8 | 11.784s | `3b152459d5265ad1b263422e835e44e6d599ab84096b250cac5668edd96b1d18` |
| LightX2V Ref2VA 4-step v0.1 | 4 | 11.350s | `43cf39f1d98364a7bb518d7481ac96e3a7e9210bdd33ab45abe36c4769111dce` |

Cold model loading is excluded from the table. Test videos remain in the
ignored local `output/validation/` directory and are not part of the archive.

A valid release must continue to satisfy all of the following:

- no syntax, import or unit-test failures;
- exact model sizes, hashes and pinned upstream revisions pass preflight;
- the bilingual Web console and REST catalog respond from the Linux hot mirror;
- at least one real Base generation and each newly declared LightX2V task
  family can load and produce an output with installed weights;
- the release archive contains no models, outputs, caches, credentials or
  workstation-specific paths.

Status: **passed**.
