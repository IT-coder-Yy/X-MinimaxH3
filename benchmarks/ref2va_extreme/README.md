# Ref2VA RTX 4090 optimization history

This directory defines the immutable human-review anchor for Ref2VA work.
Every candidate keeps its own output directory, JSON timing report, source
configuration and generated MP4. Candidates are never overwritten.

The anchor uses three still references and two standalone voice references
from `ref-test`, 864×480, 192 frames at 24 fps, and seed 82416. `v00_dense20`
is the complete 20-evaluation numerical baseline. Later `exact-*` candidates
may change only execution mechanics; approximate candidates must explicitly
record their actual/forecast schedule and attention policy.

## Candidate registry

| Version | Class | Actual/forecast | Change | Status |
| --- | --- | --- | --- | --- |
| `v00_dense20` | baseline | 20/0 | Rebuild immutable reference rows on every denoise step (`--disable-condition-row-cache`) | GPU run pending |
| `v01_exact_condition_rows` | exact mechanical | 20/0 | Cache reference video/audio packed rows inside the request layout | CPU regression complete; isolated RTX 4090 microbench saves 18.46 ms/request; end-to-end A/B pending |
| `v02_exact_reference_latents` | exact online | 20/0 | Retain one content-addressed Video/Audio-VAE reference latent pack across jobs | CPU regression complete; repeated-job GPU A/B pending |
| `v03_equivalent_condition_embeddings` | numerically equivalent | 20/0 | Project immutable reference rows once; project only target rows thereafter | CPU regression and isolated microbench pending; Human review required |

Each GPU run must write to `runtime/outputs/ref2va_extreme/<version>/` and
must retain its report, MP4, phase timings, peak VRAM and command line.  A new
run never replaces a prior version directory.

The v01 microbenchmark is stored at
`runtime/outputs/ref2va_extreme/v01_condition_rows_microbench.json`.  It shows
19.47 ms for rebuilding the condition rows over 20 steps versus 1.01 ms once.
This is a 19.19x local-path speedup but only 18.46 ms saved per request; it must
not be advertised as meaningful end-to-end acceleration.  Long-sequence DiT
remains the expected dominant cost until the full profile proves otherwise.

`ref2va_multiref_8s_enhanced_v1.json` is the Human-requested prompt-rewritten
quality anchor.  It is deliberately separate from the original manifest and
uses the `e00/e03/e10/e11/e12` registry in `candidates_enhanced_v1.json`, so
the original-prompt v00 result and every enhanced-prompt candidate remain
independently reviewable.
