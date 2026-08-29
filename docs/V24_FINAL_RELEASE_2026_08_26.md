# V24 final release (`0.7.0`)

Status: implementation, automated release gates and final Human
continuous-playback review are complete.  C02 Round2 is the default 720p15
knot on the one immutable production surface.  V009, C01 and C03 are retained
only as offline research evidence and cannot be selected by the service.  This
document separates implemented capability, measured execution and the Human
quality decision.

## Product contract

The public inference interface has exactly two numerical controls:

- `sampling_steps`: Base 5–30; LoRA 4–10;
- `acceleration`: 0–100.

For Base FL2VA/Ref2VA, V24 jointly allocates Actual/Forecast trajectory work and
per-step/per-layer Attention fidelity.  For distilled FL2VA/Ref2VA LoRA, all
selected Turbo steps remain Actual and the same acceleration control changes
only Attention allocation.  No prompt word, scene label, seed, character,
dialogue, or generated pixel selects a schedule.

The acceleration semantics are intentionally explicit:

- `0`: full Actual plus Dense Attention;
- `75`: Human-evidence quality knee;
- `75–100`: aggressive extrapolation that may have visible defects;
- `100`: fastest modeled endpoint, not a Human-quality guarantee.

The selector builds one deterministic nested compute–risk path for the real
token geometry.  Moving the slider upward cannot add compute, cannot add an
Actual step that was absent at a lower acceleration, and cannot increase
Attention fidelity on a shared cell.  Every request records curve, strategy,
execution, optimizer-chain, resource-guard, and candidate provenance.

## Unified service matrix

| Public route | Weight family | V24 technique set | Maximum public output request |
|---|---|---|---|
| FL2VA Base | original INT8 | Exact runtime + Actual/Forecast + Attention allocation | 1920x1088x362 (1080p, 15 s) |
| Ref2VA Base | original INT8 | same surface, conditioned by real packed-row pressure | 1920x1088x362 (1080p, 15 s) |
| FL2VA LoRA | Turbo LoRA | all selected Actual + Attention allocation | 1920x1088x362 (1080p, 15 s) |
| Ref2VA LoRA | Turbo LoRA | all selected Actual + Attention allocation | 1920x1088x362 (1080p, 15 s) |

Reference images and audios enter the Base surface only through actual resource
counts.  Static Ref2VA layouts up to 245k packed tokens were exercised in the
service-matrix gate.  Reference video forces every sampler position to Actual;
requests beyond the 250k V24 approximation envelope fall back to Dense rather
than borrowing an unvalidated Forecast path.  This is a safety boundary, not a
quality-routing special case.

## Measured release evidence

All measurements below use one RTX 4090, Torch 2.13, CUDA runtime 13.0, the
pinned SM89 kernels, and unchanged model weights.  The current shell's default
`nvcc` remains CUDA 12.8; CUDA 13.3 is installed separately at
`/usr/local/cuda-13.3` for explicit extension builds.  Hot inference loads the
already hash-locked binaries, so changing the global `nvcc` symlink is neither
required nor part of these timing claims.

### Final control-plane integration

720p5, 20 steps, acceleration 75:

- compiled trajectory: 10 Actual / 10 Forecast;
- end-to-end: 79.527 s; DiT: 63.694 s;
- execution digest: `c48c8b25ea97641100d07cc826ee336ebc8ecc77435ad9f80082aa91aca3abc9`;
- output SHA-256: `1ce3b2fed494db988693bd4cb8ee1a6a922f8717af4880ec311cde9bf0b4886a`;
- output is byte-identical to the Human-accepted V007 anchor.

The older V007 measurements were 78.918 s and 76.950 s, so the final service
wiring reproduces both the physical schedule and expected hot latency.  The raw
benchmark report was created before the reporter learned to replace its
all-Actual pre-tokenisation placeholder; its `execution_profile.joint_acceleration`
and `forecast_profile` are the authoritative executed 10/10 record.  Future
reports emit the executed trajectory at the top level as well.

### 720p15 final knot candidates

Same prompt, seed, 1280x736x362, 20 steps, acceleration 75, one hot session:

| Candidate | Actual/Forecast | End-to-end | DiT | Peak allocated | vs V009 333.98 s |
|---|---:|---:|---:|---:|---:|
| C01 V014b shield | 12/8 | 291.319 s | 251.036 s | 17.368 GiB | 1.146x |
| C02 Round2 trajectory | 12/8 | 289.443 s | 251.364 s | 17.368 GiB | 1.154x |
| C03 Round2 + fidelity shield | 12/8 | 301.508 s | 263.475 s | 17.368 GiB | 1.108x |

All three clear the requested 1.10x speed gate.  Automated peripheral-temporal
screening ranks C03 first, C02 second, and treats C01 as a mechanism control.
WhisperX large-v3 obtains global normalized CER 0.0 on all three.  Neither test
can adjudicate door/key contact, ghost hands, background morphing, lip clarity,
voice naturalness, or continuous motion; those remain one-vote Human hard gates.

Final Human review passed all three candidates and a second formal-H3-prompt
matrix at 720p12.25, 720p13.67 and 720p15.08.  The differences were small.
C03 reduced subtle peripheral flicker, but its key-handoff motion was worse than
C01/C02 and it was 4.2% slower than C02 on the original 720p15 comparator.
Because contact/motion causality is a hard gate while subtle edge flicker is a
secondary artifact, C02 is the final default.  The release does not synthesize
an unreviewed C04 from these observations.

### 1080p15 anchor

The shared acceleration-75 xlong knot is V18's byte-exact helper execution of
the V13-quality physical plan, which Human previously ranked best/equivalent.
Its 1920x1088x362 run completed in 656.546 s.  This is the current stable
1080p15 release anchor; the final C02 720p15 choice does not change it.

### Real route viability

`runtime/release/v24_final_20260826/runs/service_matrix/` preserves real-weight
FL2VA LoRA, Ref2VA Base, and Ref2VA LoRA runs and their reports/job records.
The LoRA jobs explicitly show all eight Actual steps, zero Forecast steps, and
`scheduler_family=h3_lora_v1_no_forecast_round229`.  They prove route execution
and unchanged scheduler provenance; only the final FL2VA Base sample above is a
new V24-control-plane execution.

The final runtime was additionally exercised at the exact maximum public geometry,
1920x1088x362, with acceleration 75 and real conditioning.  This deliberately
used the minimum public step count (Base 5, LoRA 4) as a bounded capability
stress test; the outputs are not quality candidates:

| Route | Actual/Forecast | End-to-end | DiT | Peak memory | Active power mean |
|---|---:|---:|---:|---:|---:|
| FL2VA Base | 3/2 | 398.676 s | 232.701 s | 23,988 MiB | 438.7 W |
| FL2VA LoRA | 4/0 | 531.667 s | 434.740 s | 23,914 MiB | 435.0 W |
| Ref2VA Base, 3 images + 2 audios | 3/2 | 505.157 s | 346.695 s | 23,916 MiB | 462.4 W |
| Ref2VA LoRA, 3 images + 2 audios | 4/0 | 454.474 s | 360.130 s | 23,864 MiB | 467.7 W |

All four completed without OOM or Dense safety fallback and ffprobe verified
1920x1088, 362 video frames, and a stereo AAC audio stream.  During DiT, live
samples reached 100% GPU utilisation and roughly 479–480 W power P90.  FL2VA
and Ref2VA family cold starts were measured separately at 190.554 s and
144.507 s.  The complete machine-readable record and files are in
`runtime/release/v24_final_20260826/runs/max_service_matrix/`.

The LoRA result is intentionally not advertised as universally faster: all
four distilled positions are Actual, whereas this five-step Base stress plan
contains only three Actual evaluations.  The validated product property is a
unified controllable route and a stable maximum envelope, not route-independent
LoRA dominance at every step/shape combination.

### Real HTTP and queue integration

The final `server.py --unified-console` path was then started for a separate
real-weight integration matrix.  A client entered FL2VA, submitted Base and
LoRA, switched to Ref2VA, submitted Base and LoRA with three images plus two
audios, polled the common asynchronous queue, and downloaded each result from
the public video endpoint:

| Route | Resolved engine | Actual/Forecast | Job time | Download bytes |
|---|---|---:|---:|---:|
| FL2VA Base | `original` | 3/2 | 68.976 s | 526,544 |
| FL2VA LoRA | `lora` | 4/0 | 6.446 s | 514,025 |
| Ref2VA Base | `reference` | 3/2 | 74.515 s | 502,119 |
| Ref2VA LoRA | `reference_lora` | 4/0 | 6.142 s | 520,052 |

These bounded 360p one-second jobs are integration smokes, not quality or
maximum-load evidence.  All four downloaded files were byte-identical to their
server outputs and ffprobe verified 640x352, 22 frames, 24 fps and stereo AAC.
FL2VA and Ref2VA were each loaded once; Base-to-LoRA stayed within the hot
family session.  The final model-exit request returned the service to
`active_engine=null` and `warm_state=cold`.  The portable report and outputs
are in `runtime/release/v24_final_20260826/runs/http_service_matrix/`.

## Immutable production choice and whole-policy rollback

The shipped service contains one production surface and requires no candidate
environment variable:

```bash
./scripts/start.sh
```

It always records the selected calibration as:

```bash
v24_final_c02_round2_trajectory_u7p00
```

There is no service environment variable for selecting V009, C01 or C03.
Those snapshots can only be replayed by the explicitly named offline research
compiler.  An operational rollback therefore replaces the whole scheduler:
set `H3_NATIVE_PARETO_V24=0` before process startup to use the previous
release-policy path, or deploy the previous package.  It never combines two
surfaces inside one request.  If the sparse SM89 extension is unavailable, the
UI locks acceleration to 0 and the runtime remains Dense.

## Known limits

- the request-local Forecast-debt signal is telemetry-only; it does not yet
  promote recovery steps because normal accepted long videos can also accrue
  high debt under the current null envelope;
- no claim of global Human perceptual optimality is made—the formal guarantee
  is deterministic marginal optimality on the declared smooth risk proxy;
- 1080p15 is a single-4090 maximum request envelope, not a promise that every
  maximum-reference-media combination has the same latency or approximate path;
- LoRA quality above eight user steps is not calibrated even though 9–10 steps
  are accepted by the execution contract.

## Final Human decision

The completed review record is preserved in
`runtime/release/v24_final_20260826/HUMAN_REVIEW.md`.  All formal-prompt holdout
videos passed.  C02 was selected because it preserves the better reviewed
handoff motion while providing the best overall latency; C03's weaker edge
flicker did not compensate for its worse handoff and higher runtime.
