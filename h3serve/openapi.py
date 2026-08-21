"""Public OpenAPI contract for H3 Serve clients.

The Web studio, command-line clients and ComfyUI connector all use this same
asynchronous job API.  Runtime-only implementation fields intentionally stay
out of this document.
"""

from __future__ import annotations

from typing import Any

from .contract import MAX_NATIVE_PIXEL_FRAMES


def document(version: str) -> dict[str, Any]:
    request_properties = {
        "service_family": {"type": "string", "enum": ["first_last", "reference"]},
        "model_variant": {"type": "string", "enum": ["base", "lora"], "default": "base"},
        "mode": {"type": "string", "enum": ["preset", "advanced"], "default": "preset"},
        "prompt": {
            "type": "string", "minLength": 1, "maxLength": 20000,
            "description": (
                "Final H3 model-facing text. The generation API forwards this string "
                "without invoking MiMo, compiling a storyboard, or appending sound/BGM fields."
            ),
        },
        "quality": {"type": "string", "enum": ["fast", "balanced", "quality", "ultra"]},
        "resolution": {"type": "string", "enum": ["360p", "480p", "720p", "1080p"]},
        "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"]},
        "duration_seconds": {
            "type": "number", "minimum": 1, "maximum": 15,
            "description": (
                "Requested storyboard duration. The resolved native canvas and H3 frame grid "
                f"must satisfy width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES}. Query /api/v1/options "
                "duration.max_by_preset for preset-specific ceilings."
            ),
        },
        "seed": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "enum": ["random"]}]},
        "width": {"type": "integer", "minimum": 192, "maximum": 1920, "multipleOf": 32},
        "height": {"type": "integer", "minimum": 192, "maximum": 1920, "multipleOf": 32},
        "frames": {"type": "integer", "minimum": 5, "maximum": 362},
        "sampling_steps": {
            "type": "integer",
            "minimum": 4,
            "maximum": 30,
            "description": "用户指定的总采样轨迹步数；INT8支持5–30，LoRA支持4–10。LoRA超过8步未经质量校准。",
        },
        "acceleration": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "default": 0,
            "description": (
                "连续加速强度。0为全真实步Dense端点；100为当前内部质量保护边界内的最快调度。"
                "Base自动联合分配真实/预测步和逐步逐层注意力预算；"
                "LoRA保留用户指定的全部真实Turbo步，只自适应分配逐步逐层注意力预算。"
            ),
        },
        "actual_steps": {"type": "integer", "minimum": 5, "maximum": 20, "deprecated": True},
        "lora_steps": {"type": "integer", "minimum": 4, "maximum": 8, "deprecated": True},
        "attention_keep_ratio": {"type": "number", "minimum": 0.5, "maximum": 1.0, "deprecated": True},
        "sparse_scope": {"type": "string", "enum": ["middle_only", "guarded", "full"], "deprecated": True},
        "upscale_enabled": {"type": "boolean"},
        "upscale_mode": {"type": "string", "enum": ["basic", "advanced"]},
        "upscale_resolution": {"type": "string", "enum": ["720p", "1080p", "2k"]},
        "upscale_target_width": {"type": "integer"},
        "upscale_target_height": {"type": "integer"},
        "preview_mode": {"type": "string", "enum": ["off", "auto", "pause"], "default": "off"},
        "preview_step_index": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "enum": ["auto"]}]},
        "preview_branch_steps": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
        "preview_fast_finish": {"type": "boolean", "default": False},
        "execution_mode": {"type": "string", "enum": ["complete", "checkpoint"], "default": "complete"},
        "checkpoint_step": {"type": "integer", "minimum": 1},
        "checkpoint_retain": {"type": "boolean", "default": True},
        "checkpoint_preview": {"type": "boolean", "default": False},
        "checkpoint_preview_steps": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        "checkpoint_preview_resolution": {"type": "string", "enum": ["source", "360p", "480p", "720p"], "default": "source"},
        "reference_image_resolution": {
            "type": "string",
            "enum": ["original", "360p", "480p", "720p"],
            "default": "720p",
            "description": (
                "参考图片短边分辨率上限；只按比例缩小，不放大、不裁切、不拉伸。"
                "original跳过额外压缩，模型所需的最小32像素对齐填充仍会执行。"
            ),
        },
        "reference_video_resolution": {
            "type": "string",
            "enum": ["original", "360p", "480p", "720p"],
            "default": "360p",
            "description": (
                "参考视频短边分辨率上限；只按比例缩小每帧，不改变画幅、构图或时长。"
                "original跳过额外压缩，模型所需的最小32像素对齐填充仍会执行。"
            ),
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "X-MinimaxH3 API",
            "version": version,
            "description": "Asynchronous MiniMax H3 generation API for the Web studio, scripts and ComfyUI.",
        },
        "servers": [{"url": "/"}],
        "components": {
            "securitySchemes": {
                "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
            "schemas": {
                "GenerationRequest": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": request_properties,
                    "x-h3-native-spatiotemporal-budget": {
                        "inequality": f"width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES}",
                        "reference_envelope": {
                            "width": 1920, "height": 1088, "frames": 192,
                        },
                        "frame_grid": "17*n+5",
                        "absolute_max_frames": 362,
                    },
                },
                "Job": {
                    "type": "object",
                    "required": ["id", "status", "request", "progress"],
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "status": {"type": "string", "enum": ["queued", "starting_backend", "running", "checkpointed", "awaiting_preview", "succeeded", "failed", "cancelled"]},
                        "request": {"type": "object"},
                        "progress": {"type": "object"},
                        "video_url": {"type": "string"},
                        "preview": {"type": "object"},
                        "checkpoint": {"type": "object"},
                        "inference_plan": {
                            "type": "object",
                            "description": (
                                "只读调度回执：Dense回退原因，或V19候选、"
                                "execution/envelope/certificate digest与实际/预测步数。"
                            ),
                        },
                        "error": {"type": ["string", "null"]},
                    },
                },
            },
        },
        "security": [{"ApiKey": []}],
        "paths": {
            "/healthz": {"get": {"security": [], "summary": "Liveness", "responses": {"200": {"description": "Alive"}}}},
            "/readyz": {"get": {"security": [], "summary": "Active engine readiness", "responses": {"200": {"description": "Ready"}, "503": {"description": "Not ready"}}}},
            "/api/v1/options": {"get": {"summary": "Capabilities and defaults", "responses": {"200": {"description": "Options"}}}},
            "/api/v1/workspace/browse": {
                "get": {
                    "summary": "Browse server-local workspace folders",
                    "parameters": [{"name": "path", "in": "query", "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Current folder and child directories"}},
                },
            },
            "/api/v1/workspace": {
                "put": {
                    "summary": "Select the idle unified console workspace",
                    "description": "Requires no loaded engine and no active or queued jobs.",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string", "description": "Absolute server-local directory"}},
                    }}}},
                    "responses": {"200": {"description": "Workspace selected"}, "409": {"description": "Engine or queue is active"}},
                },
            },
            "/api/v1/engine": {
                "put": {
                    "summary": "Load or switch the active service family",
                    "description": (
                        "original/lora select FL2VA; reference/reference_lora select Ref2VA. "
                        "The suffix only selects the initial task variant; later jobs can hot-switch "
                        "base and LoRA inside the loaded family. Requires an idle queue when changing family."
                    ),
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["engine"],
                        "properties": {"engine": {
                            "type": "string",
                            "enum": ["original", "lora", "reference", "reference_lora"],
                        }},
                    }}}},
                    "responses": {
                        "200": {"description": "Service family loaded"},
                        "409": {"description": "Queue is busy or family cannot be switched"},
                    },
                },
                "delete": {
                    "summary": "Unload the active service family",
                    "description": "Requires no active or queued jobs.",
                    "responses": {
                        "200": {"description": "Service family unloaded"},
                        "409": {"description": "Queue is busy"},
                    },
                },
            },
            "/api/v1/generations": {
                "post": {
                    "summary": "Submit a generation job",
                    "description": (
                        "Use JSON for text-only jobs or multipart/form-data for first/last/reference media. "
                        "Ref2VA accepts reference_image_1..9, reference_video_1..3 and "
                        "reference_audio_1..3. Each reference video is 2–15 seconds; the total "
                        "video duration is at most 15 seconds and embedded video audio is ignored. "
                        "reference_image_resolution/reference_video_resolution select proportional "
                        "downscale caps shared by Web, API and ComfyUI; they preserve aspect ratio "
                        "and never crop, stretch or pad the user media canvas."
                    ),
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/GenerationRequest"}},
                        "multipart/form-data": {"schema": {"type": "object", "properties": {
                            **request_properties,
                            "first_frame": {"type": "string", "format": "binary"},
                            "last_frame": {"type": "string", "format": "binary"},
                            **{f"reference_image_{index}": {"type": "string", "format": "binary"} for index in range(1, 10)},
                            **{f"reference_video_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                            **{f"reference_audio_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                        }}},
                    }},
                    "responses": {"202": {"description": "Accepted", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}}, "400": {"description": "Invalid request"}},
                }
            },
            "/api/v1/jobs": {
                "get": {
                    "summary": "List jobs in the active workspace",
                    "responses": {"200": {"description": "Ordered job list"}},
                },
            },
            "/api/v1/jobs/{job_id}/preview": {
                "get": {"summary": "Download a ready fork preview", "responses": {"200": {"description": "Preview MP4"}, "409": {"description": "Not ready"}}},
            },
            "/api/v1/jobs/{job_id}/preview/{decision}": {
                "post": {
                    "summary": "Continue the exact main trajectory or discard the card",
                    "parameters": [
                        {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "decision", "in": "path", "required": True, "schema": {"type": "string", "enum": ["continue", "discard"]}},
                    ],
                    "responses": {"200": {"description": "Decision accepted"}, "409": {"description": "Job is not waiting"}},
                },
            },
            "/api/v1/jobs/{job_id}/resume": {
                "post": {
                    "summary": "Queue continuation from a retained formal checkpoint",
                    "responses": {"202": {"description": "Resume queued"}, "409": {"description": "Checkpoint unavailable"}},
                },
            },
            "/api/v1/settings/mimo-key": {
                "get": {
                    "summary": "Check whether the in-memory MiMo key is configured",
                    "responses": {"200": {"description": "Configuration status only; never returns the key"}},
                },
                "put": {
                    "summary": "Set or clear the process-memory MiMo key",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["api_key"],
                        "properties": {"api_key": {"type": "string", "maxLength": 4096}},
                    }}}},
                    "responses": {"200": {"description": "Configuration status only"}},
                },
            },
            "/api/v1/settings/reference-media": {
                "get": {
                    "summary": "Read the shared reference-media preprocessing defaults",
                    "description": (
                        "These server defaults apply to Web, API and ComfyUI requests "
                        "that do not provide a per-request override."
                    ),
                    "responses": {"200": {"description": "Current proportional downscale policy"}},
                },
                "put": {
                    "summary": "Update the shared reference-media preprocessing defaults",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "image_resolution": {"type": "string", "enum": ["original", "360p", "480p", "720p"]},
                            "video_resolution": {"type": "string", "enum": ["original", "360p", "480p", "720p"]},
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved policy"},
                        "400": {"description": "Invalid resolution policy"},
                    },
                },
            },
            "/api/v1/settings/generation-limits": {
                "get": {
                    "summary": "Read per-resolution and per-ratio generation ceilings",
                    "description": (
                        "The returned preset matrix is authoritative "
                        "for Web, API and ComfyUI submissions."
                    ),
                    "responses": {"200": {"description": "Current limit policy"}},
                },
                "put": {
                    "summary": "Set every preset's maximum submission duration",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "required": ["preset_limits"],
                        "properties": {
                            "preset_limits": {
                                "type": "object",
                                "description": (
                                    "Complete resolution -> aspect ratio -> seconds matrix; "
                                    "each value is 1..15 in 0.5-second increments."
                                ),
                            },
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved policy and effective budget"},
                        "400": {"description": "Invalid limit policy"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}": {
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "get": {"summary": "Get job state", "responses": {"200": {"description": "Job", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}}}},
                "delete": {"summary": "Cancel a job", "responses": {"200": {"description": "Cancelled or cancelling"}}},
            },
            "/api/v1/jobs/{job_id}/video": {
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "get": {"summary": "Download completed MP4", "responses": {"200": {"description": "Video", "content": {"video/mp4": {}}}, "409": {"description": "Not ready"}}},
            },
        },
    }
