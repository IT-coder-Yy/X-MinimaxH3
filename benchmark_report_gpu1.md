# X-MinimaxH3 Deployment Benchmark - GPU1

## Test configuration

- Physical GPU: **GPU 1 only**
- Prompt: `A small dog sitting in front of a computer, typing code.`
- Sampling steps: `20`
- Acceleration: `50`
- Aspect ratio: `16:9`
- Model variant: `base`
- Seed: `12345`
- Repeats per case: `3`
- Warm-up: `480p / 5s`, not counted
- Sparse setting: `(unset)`
- Launcher: `fl2va_int8_16gb`
- Engine: `first_last`

Generation time is measured from `POST /api/v1/generations` until the job reaches a terminal state.
Extra VRAM is measured as `peak GPU1 memory.used - memory.used immediately before submission`.

## GPU environment

```text
1, NVIDIA GeForce RTX 4090, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 535.288.01, 24564 MiB, 9412 MiB, 0 %, 19.07 W, 30, P8
```

GPU compute processes captured before the benchmark:

```text
1406204, /usr/local/bin/python3.11, GPU-cf2a1e19-639a-f076-7845-5e962261db3c, 3268 MiB
1718504, ./llama-server, GPU-cf2a1e19-639a-f076-7845-5e962261db3c, 7366 MiB
3073884, /home/ubuntu/data/wangshuitian/prod/funasr-api/.venv/bin/python3, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 1442 MiB
2104764, /home/ubuntu/data/qwen3-vllm/.venv/bin/python3, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 7310 MiB
1788943, /home/ubuntu/data/yqy/minimax-h3/X-MinimaxH3/runtime/venv/bin/python, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 544 MiB
1475605, /home/ubuntu/data/wangshuitian/dev/Mel-Band-Roformer-Vocal-Model/.venv/bin/python3, GPU-4f8edc7c-0436-cfcf-7348-c6a65731624d, 3514 MiB
1718504, ./llama-server, GPU-4f8edc7c-0436-cfcf-7348-c6a65731624d, 6586 MiB
2104472, VLLM::EngineCore, GPU-83ae5fd3-5580-1f13-ea47-f1d9d74f2ce1, 22786 MiB
2548774, VLLM::EngineCore, GPU-ca440f06-7cdd-8cbb-9711-a8bff3fa9ee6, 18610 MiB
966131, VLLM::EngineCore, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 3914 MiB
973002, VLLM::EngineCore, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 4322 MiB
289674, /home/ubuntu/data/wangshuitian/seed-voice/.venv/bin/python3, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 3268 MiB
1487599, VLLM::EngineCore, GPU-8b10676b-73f8-8a20-a0f8-6fd29c91fc71, 18602 MiB
1087919, VLLM::EngineCore, GPU-8b10676b-73f8-8a20-a0f8-6fd29c91fc71, 4910 MiB
```

## Warm-up

- Status: `succeeded`

## Summary

| Resolution | Duration | Successful runs | OOM | Median extra VRAM (GiB) | Median generation time (s) | Job IDs |
|---:|---:|---:|---:|---:|---:|---|
| 480p | 5s | 3/3 | 0 | 7.582 | 42.39 | c15ecde2-47db-48f5-b041-38c0edecf0f5, 738ba3fe-d07f-419b-923f-d5cf58f6c31d, d5bb7bc2-5952-4b6b-88c3-88836950ba15 |
| 480p | 10s | 3/3 | 0 | 9.342 | 90.68 | 9a0ec79c-81a8-48eb-a0a9-30d40a3a3158, 03cbcb9b-f1c2-4fe5-b37b-332d54e3d1a1, a961781a-1410-4a4f-ad87-60c92003cf16 |
| 720p | 5s | 3/3 | 0 | 10.887 | 106.66 | de606c4e-a2ad-455e-8d8e-4ce76c6a0dd3, 0020dbb5-6684-43c4-8a20-7700e1135119, b634c03b-fac8-4858-a94d-16b0adb6dd26 |
| 720p | 10s | 2/3 | 0 | 13.374 | 255.16 | e3fcd3be-9f70-43c7-ab89-daf8cef3756f, e585e875-d169-40fd-b2fd-5068aa8be6e6, aa25b508-a12b-4c6f-a53e-3adc5849f604 |
| 1080p | 5s | 0/3 | 0 | - | - | 43f103fc-3920-4faa-9b9d-e477da572862, b4019668-e436-40ab-83a5-8fd9ed0d26c7, 4488230e-0765-43f2-b2a5-5a72d7739895 |

## Run details

| Case | Run | Status | Extra VRAM (GiB) | Generation time (s) | Job ID | Error |
|---|---:|---|---:|---:|---|---|
| 480p_5s | 1 | succeeded | 7.621 | 42.63 | c15ecde2-47db-48f5-b041-38c0edecf0f5 | - |
| 480p_5s | 2 | succeeded | 7.582 | 42.39 | 738ba3fe-d07f-419b-923f-d5cf58f6c31d | - |
| 480p_5s | 3 | succeeded | 7.508 | 41.36 | d5bb7bc2-5952-4b6b-88c3-88836950ba15 | - |
| 480p_10s | 1 | succeeded | 9.348 | 91.55 | 9a0ec79c-81a8-48eb-a0a9-30d40a3a3158 | - |
| 480p_10s | 2 | succeeded | 9.342 | 90.68 | 03cbcb9b-f1c2-4fe5-b37b-332d54e3d1a1 | - |
| 480p_10s | 3 | succeeded | 9.316 | 90.57 | a961781a-1410-4a4f-ad87-60c92003cf16 | - |
| 720p_5s | 1 | succeeded | 10.887 | 108.86 | de606c4e-a2ad-455e-8d8e-4ce76c6a0dd3 | - |
| 720p_5s | 2 | succeeded | 10.889 | 106.66 | 0020dbb5-6684-43c4-8a20-7700e1135119 | - |
| 720p_5s | 3 | succeeded | 10.596 | 106.55 | b634c03b-fac8-4858-a94d-16b0adb6dd26 | - |
| 720p_10s | 1 | succeeded | 13.459 | 257.20 | e3fcd3be-9f70-43c7-ab89-daf8cef3756f | - |
| 720p_10s | 2 | succeeded | 13.289 | 253.12 | e585e875-d169-40fd-b2fd-5068aa8be6e6 | - |
| 720p_10s | 3 | failed | 11.779 | 248.00 | aa25b508-a12b-4c6f-a53e-3adc5849f604 | generation failed (reference aa25b508) |
| 1080p_5s | 1 | failed | 9.457 | 317.92 | 43f103fc-3920-4faa-9b9d-e477da572862 | generation failed (reference 43f103fc) |
| 1080p_5s | 2 | failed | 8.123 | 315.28 | b4019668-e436-40ab-83a5-8fd9ed0d26c7 | generation failed (reference b4019668) |
| 1080p_5s | 3 | failed | 8.119 | 314.03 | 4488230e-0765-43f2-b2a5-5a72d7739895 | generation failed (reference 4488230e) |

## Notes

- Tests run sequentially; there is no multi-GPU or concurrent generation.
- If one run of a case OOMs, remaining repeats of that case are skipped.
- If OOM kills the H3 service, the report is saved first and the benchmark exits.
- After manually restarting the H3 service, run the same script with `--resume`.
- Completed runs are not repeated when resuming.
- No generated video is downloaded or copied by this benchmark.
- If another process changes its GPU1 memory usage during a run, the extra-VRAM number can be affected.

Last updated: `2026-09-02 10:09:54`

<!-- H3_BENCHMARK_STATE_V1
{"version":1,"config":{"gpu_index":1,"prompt":"A small dog sitting in front of a computer, typing code.","acceleration":50,"sampling_steps":20,"aspect_ratio":"16:9","model_variant":"base","seed":12345,"repeats":3,"cases":[["480p",5],["480p",10],["720p",5],["720p",10],["1080p",5]],"warmup_case":["480p",5]},"metadata":{"launcher":"fl2va_int8_16gb","engine":"first_last","sparse_env":"(unset)","gpu_info":"1, NVIDIA GeForce RTX 4090, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 535.288.01, 24564 MiB, 9412 MiB, 0 %, 19.07 W, 30, P8","gpu_processes":"1406204, /usr/local/bin/python3.11, GPU-cf2a1e19-639a-f076-7845-5e962261db3c, 3268 MiB\n1718504, ./llama-server, GPU-cf2a1e19-639a-f076-7845-5e962261db3c, 7366 MiB\n3073884, /home/ubuntu/data/wangshuitian/prod/funasr-api/.venv/bin/python3, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 1442 MiB\n2104764, /home/ubuntu/data/qwen3-vllm/.venv/bin/python3, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 7310 MiB\n1788943, /home/ubuntu/data/yqy/minimax-h3/X-MinimaxH3/runtime/venv/bin/python, GPU-33ab5898-8be1-a9aa-11a4-6331fa498d9a, 544 MiB\n1475605, /home/ubuntu/data/wangshuitian/dev/Mel-Band-Roformer-Vocal-Model/.venv/bin/python3, GPU-4f8edc7c-0436-cfcf-7348-c6a65731624d, 3514 MiB\n1718504, ./llama-server, GPU-4f8edc7c-0436-cfcf-7348-c6a65731624d, 6586 MiB\n2104472, VLLM::EngineCore, GPU-83ae5fd3-5580-1f13-ea47-f1d9d74f2ce1, 22786 MiB\n2548774, VLLM::EngineCore, GPU-ca440f06-7cdd-8cbb-9711-a8bff3fa9ee6, 18610 MiB\n966131, VLLM::EngineCore, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 3914 MiB\n973002, VLLM::EngineCore, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 4322 MiB\n289674, /home/ubuntu/data/wangshuitian/seed-voice/.venv/bin/python3, GPU-5b9a55bf-3229-3b5b-3590-c59a68c1a4a4, 3268 MiB\n1487599, VLLM::EngineCore, GPU-8b10676b-73f8-8a20-a0f8-6fd29c91fc71, 18602 MiB\n1087919, VLLM::EngineCore, GPU-8b10676b-73f8-8a20-a0f8-6fd29c91fc71, 4910 MiB","base_url":"http://127.0.0.1:21900"},"warmup_done":true,"warmup_status":"succeeded","records":[{"case_id":"480p_5s","resolution":"480p","duration_seconds":5,"run_index":1,"status":"succeeded","job_id":"c15ecde2-47db-48f5-b041-38c0edecf0f5","extra_vram_gib":7.621,"generation_seconds":42.63,"error":"","started_at":"2026-09-02 00:42:37","finished_at":"2026-09-02 00:43:20"},{"case_id":"480p_5s","resolution":"480p","duration_seconds":5,"run_index":2,"status":"succeeded","job_id":"738ba3fe-d07f-419b-923f-d5cf58f6c31d","extra_vram_gib":7.582,"generation_seconds":42.387,"error":"","started_at":"2026-09-02 00:43:23","finished_at":"2026-09-02 00:44:06"},{"case_id":"480p_5s","resolution":"480p","duration_seconds":5,"run_index":3,"status":"succeeded","job_id":"d5bb7bc2-5952-4b6b-88c3-88836950ba15","extra_vram_gib":7.508,"generation_seconds":41.358,"error":"","started_at":"2026-09-02 00:44:09","finished_at":"2026-09-02 00:44:50"},{"case_id":"480p_10s","resolution":"480p","duration_seconds":10,"run_index":1,"status":"succeeded","job_id":"9a0ec79c-81a8-48eb-a0a9-30d40a3a3158","extra_vram_gib":9.348,"generation_seconds":91.549,"error":"","started_at":"2026-09-02 00:44:53","finished_at":"2026-09-02 00:46:25"},{"case_id":"480p_10s","resolution":"480p","duration_seconds":10,"run_index":2,"status":"succeeded","job_id":"03cbcb9b-f1c2-4fe5-b37b-332d54e3d1a1","extra_vram_gib":9.342,"generation_seconds":90.677,"error":"","started_at":"2026-09-02 00:46:28","finished_at":"2026-09-02 00:47:59"},{"case_id":"480p_10s","resolution":"480p","duration_seconds":10,"run_index":3,"status":"succeeded","job_id":"a961781a-1410-4a4f-ad87-60c92003cf16","extra_vram_gib":9.316,"generation_seconds":90.572,"error":"","started_at":"2026-09-02 00:48:02","finished_at":"2026-09-02 00:49:33"},{"case_id":"720p_5s","resolution":"720p","duration_seconds":5,"run_index":1,"status":"succeeded","job_id":"de606c4e-a2ad-455e-8d8e-4ce76c6a0dd3","extra_vram_gib":10.887,"generation_seconds":108.857,"error":"","started_at":"2026-09-02 00:49:36","finished_at":"2026-09-02 00:51:24"},{"case_id":"720p_5s","resolution":"720p","duration_seconds":5,"run_index":2,"status":"succeeded","job_id":"0020dbb5-6684-43c4-8a20-7700e1135119","extra_vram_gib":10.889,"generation_seconds":106.658,"error":"","started_at":"2026-09-02 00:51:28","finished_at":"2026-09-02 00:53:14"},{"case_id":"720p_5s","resolution":"720p","duration_seconds":5,"run_index":3,"status":"succeeded","job_id":"b634c03b-fac8-4858-a94d-16b0adb6dd26","extra_vram_gib":10.596,"generation_seconds":106.552,"error":"","started_at":"2026-09-02 00:53:17","finished_at":"2026-09-02 00:55:04"},{"case_id":"720p_10s","resolution":"720p","duration_seconds":10,"run_index":1,"status":"succeeded","job_id":"e3fcd3be-9f70-43c7-ab89-daf8cef3756f","extra_vram_gib":13.459,"generation_seconds":257.203,"error":"","started_at":"2026-09-02 00:55:07","finished_at":"2026-09-02 00:59:25"},{"case_id":"720p_10s","resolution":"720p","duration_seconds":10,"run_index":2,"status":"succeeded","job_id":"e585e875-d169-40fd-b2fd-5068aa8be6e6","extra_vram_gib":13.289,"generation_seconds":253.118,"error":"","started_at":"2026-09-02 00:59:28","finished_at":"2026-09-02 01:03:41"},{"case_id":"720p_10s","resolution":"720p","duration_seconds":10,"run_index":3,"status":"failed","job_id":"aa25b508-a12b-4c6f-a53e-3adc5849f604","extra_vram_gib":11.779,"generation_seconds":247.997,"error":"generation failed (reference aa25b508)","started_at":"2026-09-02 01:03:44","finished_at":"2026-09-02 01:07:52"},{"case_id":"1080p_5s","resolution":"1080p","duration_seconds":5,"run_index":1,"status":"failed","job_id":"43f103fc-3920-4faa-9b9d-e477da572862","extra_vram_gib":9.457,"generation_seconds":317.924,"error":"generation failed (reference 43f103fc)","started_at":"2026-09-02 01:07:55","finished_at":"2026-09-02 01:13:14"},{"case_id":"1080p_5s","resolution":"1080p","duration_seconds":5,"run_index":2,"status":"failed","job_id":"b4019668-e436-40ab-83a5-8fd9ed0d26c7","extra_vram_gib":8.123,"generation_seconds":315.284,"error":"generation failed (reference b4019668)","started_at":"2026-09-02 01:13:17","finished_at":"2026-09-02 01:18:32"},{"case_id":"1080p_5s","resolution":"1080p","duration_seconds":5,"run_index":3,"status":"failed","job_id":"4488230e-0765-43f2-b2a5-5a72d7739895","extra_vram_gib":8.119,"generation_seconds":314.03,"error":"generation failed (reference 4488230e)","started_at":"2026-09-02 01:18:35","finished_at":"2026-09-02 01:23:49"}],"case_oom":{},"created_at":"2026-09-02 00:41:43","updated_at":"2026-09-02 10:09:54"}
H3_BENCHMARK_STATE_V1 -->
