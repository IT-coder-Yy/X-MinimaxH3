# Third-party notices

X-MinimaxH3 is original service and acceleration code combined with narrowly
vendored source subsets, prebuilt extensions, and separately downloaded model
weights. Component provenance does not change the license of this project's
original code.

| Component | Pinned revision/version | Distribution form | License/provenance |
|---|---:|---|---|
| MiniMax H3 | `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea` | tokenizer and VAE runtime subset; model files downloaded separately | [`third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE`](third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE), root `NOTICE`, and `runtime_sources/PROVENANCE.json` |
| LightX2V | `205d5c872d01557935dc87d67156f4f94069ea65` | audio-VAE loader runtime subset | `runtime_sources/LightX2V/LICENSE` and `runtime_sources/PROVENANCE.json` |
| SageAttention | `2.2.0` | CPython 3.10 Linux SM89 wheel | `third_party_licenses/SageAttention-LICENSE` |
| SpargeAttention | `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` | validated CPython 3.10/Torch 2.8/SM89 extension | `prebuilt/sparge-sm89-py310-torch28-cu126/PROVENANCE.json`; Apache-2.0 |
| Comfy Kitchen | `0.2.26` plus the pinned backend binary | Python package and modified CUDA backend | `third_party_licenses/Comfy-Kitchen-LICENSE`, `third_party_licenses/Comfy-Kitchen-NOTICE` |
| MiniMax H3 Turbo node | derived from `55fee864dd7b2976b1c4ce3c3d5f7968f181409f` | modified loader/runtime code | `third_party_licenses/Turbo-Node-LICENSE` |
| Spectrum MiniMax H3 | `dc6291525112cb4246f864738e5bb4e2b85446da` | provenance for forecasting work | `third_party_licenses/Spectrum-LICENSE` |
| FlashVSR | `b527c6f285fb30df530f5febc8b45764a789c961` | minimal isolated worker source | `third_party/flashvsr/LICENSE`, `third_party/flashvsr/PROVENANCE.md` |
| Block Sparse Attention | `49d6c39e4dc0303442cda3bb758b3925d4399c49` | CPython 3.11 Linux wheel used by FlashVSR | `third_party/block_sparse_attention/LICENSE`, `third_party/block_sparse_attention/PROVENANCE-H3.md` |
| FasterVQA | `8db452e2caa5d5d4da507bcf577c19b8114f2ebd` | optional offline evaluation only | `third_party_licenses/FasterVQA-LICENSE` |

Model weights are not stored in this source distribution. Their repositories,
immutable upstream revisions, expected sizes, install paths, and SHA-256
digests are recorded in `models/manifest.json`. The downloader asks the user to
accept model licenses before transfer. Users remain responsible for reviewing
the publishers' licenses and restrictions before use or redistribution.

The MiniMax H3 agreement contains territorial, acceptable-use, downstream-user,
commercial and attribution requirements. Its complete text is shipped locally;
the short root `NOTICE` is the redistribution notice required by Section III.4.

X-MinimaxH3's original code is released under the root `LICENSE`. Third-party
components remain governed by the licenses listed in this document.
