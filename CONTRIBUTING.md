# Contributing

This project targets one deliberately narrow production envelope: Linux x86_64
or Windows 11 through WSL2, one RTX 4090 (SM89), and the pinned runtime/model
contract in this repository. Changes that claim another GPU or dependency stack
need reproducible installation, cold-load and generation evidence.

Before opening a pull request:

1. keep downloaded models, workspaces, videos, logs and local secrets out of Git;
2. preserve Web/API/ComfyUI request compatibility and model capabilities;
3. record third-party provenance and license obligations for vendored material;
4. run the service and connector tests described in `VALIDATION.md`;
5. run `python scripts/verify_release.py` from the repository root.

Performance changes should report the exact model variant, canvas, frame count,
sampling steps, acceleration value, seed, hot/cold state, elapsed time and peak
memory. Approximate acceleration also requires human review of motion causality,
audio identity/quality, clarity and visible artifacts; a single automatic score
is not sufficient evidence.
