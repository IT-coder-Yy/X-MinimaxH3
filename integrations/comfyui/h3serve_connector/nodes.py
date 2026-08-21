from __future__ import annotations

import io
import json
import math
import os
import wave
from pathlib import Path
from typing import Any

from .client import H3ServeClient, H3ServeError


CATEGORY = "H3 Serve"
CONNECTION = "H3_SERVE_CONNECTION"
REFERENCES = "H3_SERVE_REFERENCES"
QUALITY = {
    "极速": "fast",
    "均衡": "balanced",
    "高质量": "quality",
    "超高质量": "ultra",
}
MODEL_VARIANTS = {"原始权重": "base", "LoRA 极速": "lora"}
PREVIEW_MODES = {
    "关闭": "off",
    "生成预览并自动继续": "auto",
    "在Web控制台抽卡": "pause",
}
REFERENCE_RESOLUTIONS: dict[str, str | None] = {
    "使用服务端设置": None,
    "保持原样": "original",
    "360P": "360p",
    "480P": "480p",
    "720P": "720p",
}
PRESET_SHORT_EDGES = {"360p": 360, "480p": 480, "720p": 720, "1080p": 1080}
PRESET_ASPECT_RATIOS = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4),
    "16:9": (16, 9), "9:16": (9, 16),
}
MAX_NATIVE_PIXEL_FRAMES = 1920 * 1088 * 192


def _acceleration_value(value: Any) -> float:
    """Normalize corrupt values from older positionally serialized workflows."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _nearest_32(value: float) -> int:
    return max(32, int(value / 32 + 0.5) * 32)


def _preset_geometry(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    short_edge = PRESET_SHORT_EDGES[resolution]
    rw, rh = PRESET_ASPECT_RATIOS[aspect_ratio]
    if rw >= rh:
        return _nearest_32(short_edge * rw / rh), _nearest_32(short_edge)
    return _nearest_32(short_edge), _nearest_32(short_edge * rh / rw)


def _max_native_duration(width: int, height: int) -> float:
    raw_frames = min(362, MAX_NATIVE_PIXEL_FRAMES // (int(width) * int(height)))
    legal_frames = 5 + 17 * max(0, (raw_frames - 5) // 17)
    return min(15.0, legal_frames / 24.0)


def _validate_native_duration(width: int, height: int, duration_seconds: float) -> None:
    maximum = _max_native_duration(width, height)
    if float(duration_seconds) > maximum:
        raise H3ServeError(
            f"{width}×{height} 在当前时空预算下最长支持 {maximum:.3f} 秒"
        )


def _add_reference_resolution_fields(fields: dict[str, Any], values: dict[str, Any]) -> None:
    """Map Ref2VA UI controls to the shared HTTP generation contract."""

    image_value = values.get("参考图片分辨率")
    video_value = values.get("参考视频分辨率")
    if image_value is not None and REFERENCE_RESOLUTIONS[image_value] is not None:
        fields["reference_image_resolution"] = REFERENCE_RESOLUTIONS[image_value]
    if video_value is not None and REFERENCE_RESOLUTIONS[video_value] is not None:
        fields["reference_video_resolution"] = REFERENCE_RESOLUTIONS[video_value]


def _add_reference_resolution_inputs(schema: dict[str, Any]) -> dict[str, Any]:
    schema["required"].update({
        "参考图片分辨率": (list(REFERENCE_RESOLUTIONS), {"default": "使用服务端设置"}),
        "参考视频分辨率": (list(REFERENCE_RESOLUTIONS), {"default": "使用服务端设置"}),
    })
    return schema


def _image_png(image: Any) -> bytes:
    import numpy as np
    from PIL import Image

    array = image[0].detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array[..., :3]), mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _audio_wav(audio: dict[str, Any]) -> bytes:
    import numpy as np

    waveform = audio["waveform"][0].detach().float().clamp(-1, 1).cpu().numpy()
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    pcm = (np.ascontiguousarray(waveform.T) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(pcm.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(int(audio["sample_rate"]))
        writer.writeframes(pcm.tobytes())
    return output.getvalue()


def _video_bytes(video: Any) -> bytes:
    """Read a ComfyUI VIDEO without re-encoding it.

    Core ``Load Video`` returns ``VideoFromFile``.  Its public
    ``get_stream_source`` method preserves the original container, which is
    precisely what the service-side validator and Ref2VA decoder expect.
    """

    source_getter = getattr(video, "get_stream_source", None)
    if not callable(source_getter):
        raise ValueError("参考视频必须连接 ComfyUI 的 VIDEO 输出（例如 Load Video）")
    source = source_getter()
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source.read()
    raise ValueError("不支持的 ComfyUI VIDEO 来源")


def _client(connection: dict[str, Any]) -> H3ServeClient:
    return H3ServeClient(connection["server_url"], connection.get("api_key", ""))


def _require_ready_engine(client: H3ServeClient) -> tuple[str, dict[str, Any]]:
    """Fail early unless the operator has entered a fully ready engine."""

    options = client.get("/api/v1/options")
    control = options.get("engine_control") or {}
    if control.get("switching"):
        raise H3ServeError("H3引擎正在加载或切换，请等待8090控制台显示“服务已就绪”")
    engine = options.get("current_engine")
    if not engine:
        raise H3ServeError("H3服务已启动，但尚未选择引擎；请先在8090控制台选择并加载模型")
    try:
        readiness = client.get("/readyz")
    except H3ServeError as error:
        raise H3ServeError(
            f"当前H3引擎尚未就绪；请等待8090控制台完成模型加载（{error}）"
        ) from error
    if readiness.get("status") != "ready":
        raise H3ServeError("当前H3引擎尚未就绪；请等待8090控制台完成模型加载")
    return str(engine), options


def _prompt_with_music_policy(prompt: str, background_music: str) -> str:
    if background_music == "关闭" and "non_diegetic_music:" not in prompt:
        return prompt.rstrip() + "\n\nnon_diegetic_music: N/A"
    return prompt


def _reference_files(references: dict[str, Any] | None) -> dict[str, tuple[str, bytes]]:
    files: dict[str, tuple[str, bytes]] = {}
    if not references:
        return files
    for index, content in enumerate(references.get("images", ()), 1):
        files[f"reference_image_{index}"] = (f"reference_{index}.png", content)
    for index, content in enumerate(references.get("videos", ()), 1):
        files[f"reference_video_{index}"] = (f"reference_{index}.mp4", content)
    for index, content in enumerate(references.get("audios", ()), 1):
        files[f"reference_audio_{index}"] = (f"reference_{index}.wav", content)
    return files


def _direct_reference_inputs() -> dict[str, tuple[str]]:
    """Expose the model's numbered references directly on generation nodes."""

    return {
        **{f"Picture {index}": ("IMAGE",) for index in range(1, 10)},
        **{f"Video {index}": ("VIDEO",) for index in range(1, 4)},
        **{f"Audio {index}": ("AUDIO",) for index in range(1, 4)},
    }


def _direct_references(values: dict[str, Any]) -> dict[str, tuple[bytes, ...]] | None:
    images = tuple(
        _image_png(values[f"Picture {index}"])
        for index in range(1, 10)
        if values.get(f"Picture {index}") is not None
    )
    videos = tuple(
        _video_bytes(values[f"Video {index}"])
        for index in range(1, 4)
        if values.get(f"Video {index}") is not None
    )
    audios = tuple(
        _audio_wav(values[f"Audio {index}"])
        for index in range(1, 4)
        if values.get(f"Audio {index}") is not None
    )
    return {"images": images, "videos": videos, "audios": audios} if images or videos or audios else None


def _run(connection: dict[str, Any], fields: dict[str, Any], *,
         first_frame=None, last_frame=None, references=None,
         include_metadata: bool = False,
         expose_preview_output: bool = False):
    import comfy.model_management
    import comfy.utils
    import folder_paths
    from comfy_api.latest import InputImpl, io as comfy_io, ui as comfy_ui

    client = _client(connection)
    _require_ready_engine(client)
    files = _reference_files(references)
    if first_frame is not None:
        files["first_frame"] = ("first_frame.png", _image_png(first_frame))
    if last_frame is not None:
        files["last_frame"] = ("last_frame.png", _image_png(last_frame))
    job = client.submit(fields, files)
    job_id = str(job["id"])
    pbar = comfy.utils.ProgressBar(100)

    def cancel_check():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def progress(document):
        value = float(document.get("progress", {}).get("percent") or 0)
        pbar.update_absolute(round(value), 100)

    try:
        completed = client.wait(job_id, poll_seconds=float(connection["poll_seconds"]),
                                progress=progress, cancel_check=cancel_check)
    except BaseException:
        client.cancel(job_id)
        raise
    output = Path(folder_paths.get_output_directory()) / "h3_serve" / f"{job_id}.mp4"
    client.download(job_id, output)
    pbar.update_absolute(100, 100)
    values = [InputImpl.VideoFromFile(str(output))]
    preview_output = None
    preview_result = None
    if expose_preview_output and fields.get("preview_mode") != "off":
        preview_output = (
            Path(folder_paths.get_output_directory())
            / "h3_serve"
            / f"{job_id}-preview.mp4"
        )
        client.download_preview(job_id, preview_output)
        values.append(InputImpl.VideoFromFile(str(preview_output)))
        preview_result = comfy_ui.SavedResult(
            preview_output.name, "h3_serve", comfy_io.FolderType.output,
        )
    elif expose_preview_output:
        # VIDEO outputs may be intentionally empty when the user disables the
        # optional preview branch. Downstream nodes should connect this socket
        # only when preview is enabled.
        values.append(None)
    if include_metadata:
        values.extend((
            str(output),
            job_id,
            json.dumps(completed, ensure_ascii=False, indent=2),
        ))
    ui_results = [
        comfy_ui.SavedResult(output.name, "h3_serve", comfy_io.FolderType.output),
    ]
    if preview_result is not None:
        # Put the cheap branch first in the node's embedded preview area; the
        # named output sockets still remain 最终视频 followed by 预览视频.
        ui_results.insert(0, preview_result)
    return comfy_io.NodeOutput(*values, ui=comfy_ui.PreviewVideo(ui_results))


def _wait_for_checkpoint(connection, fields, *, first_frame=None, last_frame=None,
                         references=None):
    """Submit a breakpoint job without pretending it already has a final VIDEO."""

    import comfy.model_management
    import comfy.utils

    client = _client(connection)
    _require_ready_engine(client)
    files = _reference_files(references)
    if first_frame is not None:
        files["first_frame"] = ("first_frame.png", _image_png(first_frame))
    if last_frame is not None:
        files["last_frame"] = ("last_frame.png", _image_png(last_frame))
    job = client.submit(fields, files)
    job_id = str(job["id"])
    pbar = comfy.utils.ProgressBar(100)

    def cancel_check():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def progress(document):
        value = float(document.get("progress", {}).get("percent") or 0)
        pbar.update_absolute(round(value), 100)

    try:
        stopped = client.wait_until_stopped(
            job_id, poll_seconds=float(connection["poll_seconds"]),
            progress=progress, cancel_check=cancel_check,
        )
    except BaseException:
        client.cancel(job_id)
        raise
    return job_id, json.dumps(stopped, ensure_ascii=False, indent=2)


def _download_job_video(connection, job_id: str, *, preview: bool):
    import folder_paths
    from comfy_api.latest import InputImpl, io as comfy_io, ui as comfy_ui

    client = _client(connection)
    suffix = "preview" if preview else "final"
    output = Path(folder_paths.get_output_directory()) / "h3_serve" / f"{job_id}-{suffix}.mp4"
    if preview:
        client.download_preview(job_id, output)
    else:
        client.download(job_id, output)
    return comfy_io.NodeOutput(
        InputImpl.VideoFromFile(str(output)),
        ui=comfy_ui.PreviewVideo([
            comfy_ui.SavedResult(output.name, "h3_serve", comfy_io.FolderType.output),
        ]),
    )


def _interactive_checkpoint_output(connection, job_id: str, document: dict[str, Any], *,
                                   resumed: bool = False):
    """Expose a stopped formal checkpoint as the creator node's preview output."""

    import folder_paths
    from comfy_api.latest import InputImpl, io as comfy_io, ui as comfy_ui

    status = str(document.get("status") or "")
    output_dir = Path(folder_paths.get_output_directory()) / "h3_serve"
    output_dir.mkdir(parents=True, exist_ok=True)
    ui_results = []
    final_video = None
    preview_video = None
    client = _client(connection)

    if status == "succeeded":
        final_path = output_dir / f"{job_id}-final.mp4"
        client.download(job_id, final_path)
        final_video = InputImpl.VideoFromFile(str(final_path))
        ui_results.append(
            comfy_ui.SavedResult(final_path.name, "h3_serve", comfy_io.FolderType.output)
        )
        if resumed:
            try:
                preview_path = output_dir / f"{job_id}-preview.mp4"
                client.download_preview(job_id, preview_path)
                preview_video = InputImpl.VideoFromFile(str(preview_path))
            except H3ServeError:
                preview_video = None
    elif status == "checkpointed":
        preview_path = output_dir / f"{job_id}-preview.mp4"
        client.download_preview(job_id, preview_path)
        preview_video = InputImpl.VideoFromFile(str(preview_path))
        ui_results.append(
            comfy_ui.SavedResult(preview_path.name, "h3_serve", comfy_io.FolderType.output)
        )
    else:
        raise H3ServeError(f"断点任务意外停止在状态：{status or 'unknown'}")

    ui_payload = comfy_ui.PreviewVideo(ui_results).as_dict() if ui_results else {}
    ui_payload["h3_checkpoint"] = [{
        "job_id": job_id,
        "status": status,
        "resume_available": bool(
            (document.get("checkpoint") or {}).get("resume_available")
        ),
    }]
    return comfy_io.NodeOutput(final_video, preview_video, ui=ui_payload)


def _run_interactive_checkpoint(connection, fields, *, first_frame=None,
                                last_frame=None, references=None):
    job_id, details = _wait_for_checkpoint(
        connection, fields, first_frame=first_frame, last_frame=last_frame,
        references=references,
    )
    return _interactive_checkpoint_output(
        connection, job_id, json.loads(details), resumed=False,
    )


def _resume_interactive_checkpoint(connection, job_id: str):
    import comfy.model_management
    import comfy.utils

    client = _client(connection)
    client.resume(job_id)
    pbar = comfy.utils.ProgressBar(100)
    completed = client.wait(
        job_id,
        poll_seconds=float(connection["poll_seconds"]),
        progress=lambda document: pbar.update_absolute(
            round(float(document.get("progress", {}).get("percent") or 0)), 100
        ),
        cancel_check=comfy.model_management.throw_exception_if_processing_interrupted,
    )
    pbar.update_absolute(100, 100)
    return _interactive_checkpoint_output(
        connection, job_id, completed, resumed=True,
    )


def _discard_interactive_checkpoint(connection, job_id: str):
    from comfy_api.latest import io as comfy_io

    _client(connection).delete_record(job_id)
    return comfy_io.NodeOutput(
        None, None,
        ui={"h3_checkpoint": [{
            "job_id": "", "status": "reset", "resume_available": False,
        }]},
    )


class _H3ServeCheckpointSubmitBase:
    RAW_PROMPT = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "连接": (CONNECTION,),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 192, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 480, "min": 192, "max": 1920, "step": 32}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "model_variant": (list(MODEL_VARIANTS), {"default": "原始权重"}),
                "sampling_steps": ("INT", {"default": 8, "min": 4, "max": 20, "step": 1}),
                "acceleration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "checkpoint_step": ("INT", {"default": 3, "min": 1, "max": 19, "step": 1}),
                "retain_checkpoint": (["保留", "预览后删除"], {"default": "保留"}),
                "output_preview": (["关闭", "开启"], {"default": "关闭"}),
                "preview_resolution": (["原分辨率", "360p", "480p", "720p"], {"default": "原分辨率"}),
                "preview_lora_steps": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 4404, "min": 0, "max": 0xffffffffffffffff}),
                "background_music": (["关闭", "开启"], {"default": "关闭"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("任务ID", "断点任务详情")
    FUNCTION = "submit_checkpoint"
    CATEGORY = f"{CATEGORY}/断点任务"
    OUTPUT_NODE = True

    def submit_checkpoint(
        self, 连接, prompt, width, height, duration_seconds, model_variant,
        sampling_steps, acceleration, checkpoint_step, retain_checkpoint,
        output_preview, preview_resolution, preview_lora_steps, seed, **kwargs,
    ):
        background_music = kwargs.pop("background_music", "原样")
        variant = MODEL_VARIANTS[model_variant]
        lower, upper = (4, 8) if variant == "lora" else (8, 20)
        if not lower <= int(sampling_steps) <= upper:
            raise H3ServeError(
                f"{model_variant} 的总采样步数必须在 {lower}–{upper} 之间"
            )
        if not 1 <= int(checkpoint_step) < int(sampling_steps):
            raise H3ServeError(
                "断点位置必须大于0且小于总采样步数"
            )
        fields = {
            "mode": "advanced",
            "prompt": (
                prompt if self.RAW_PROMPT
                else _prompt_with_music_policy(prompt, background_music)
            ),
            "width": int(width),
            "height": int(height),
            "duration_seconds": duration_seconds,
            "model_variant": variant,
            "sampling_steps": int(sampling_steps),
            "acceleration": _acceleration_value(acceleration),
            "seed": seed,
            "execution_mode": "checkpoint",
            "checkpoint_step": checkpoint_step,
            "checkpoint_retain": retain_checkpoint == "保留",
            "checkpoint_preview": output_preview == "开启",
            "checkpoint_preview_steps": preview_lora_steps,
            "checkpoint_preview_resolution": (
                "source" if preview_resolution == "原分辨率" else preview_resolution
            ),
        }
        _add_reference_resolution_fields(fields, kwargs)
        return _wait_for_checkpoint(
            连接, fields,
            first_frame=kwargs.get("first_frame"),
            last_frame=kwargs.get("last_frame"),
            references=kwargs.get("参考素材") or _direct_references(kwargs),
        )


class H3ServeFL2VACheckpointSubmit(_H3ServeCheckpointSubmitBase):
    RAW_PROMPT = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"].pop("background_music", None)
        schema["optional"] = {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)}
        return schema


class H3ServeRef2VACheckpointSubmit(_H3ServeCheckpointSubmitBase):
    RAW_PROMPT = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = _add_reference_resolution_inputs(super().INPUT_TYPES())
        schema["required"].pop("background_music", None)
        schema["optional"] = _direct_reference_inputs()
        return schema


class H3ServeCheckpointPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"连接": (CONNECTION,), "任务ID": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("断点预览",)
    FUNCTION = "download"
    CATEGORY = f"{CATEGORY}/断点任务"
    OUTPUT_NODE = True

    def download(self, 连接, 任务ID):
        return _download_job_video(连接, 任务ID.strip(), preview=True)


class H3ServeCheckpointResume:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"连接": (CONNECTION,), "任务ID": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("最终视频",)
    FUNCTION = "resume"
    CATEGORY = f"{CATEGORY}/断点任务"
    OUTPUT_NODE = True

    def resume(self, 连接, 任务ID):
        import comfy.model_management
        import comfy.utils

        job_id = 任务ID.strip()
        client = _client(连接)
        client.resume(job_id)
        pbar = comfy.utils.ProgressBar(100)
        completed = client.wait(
            job_id,
            poll_seconds=float(连接["poll_seconds"]),
            progress=lambda document: pbar.update_absolute(
                round(float(document.get("progress", {}).get("percent") or 0)), 100
            ),
            cancel_check=comfy.model_management.throw_exception_if_processing_interrupted,
        )
        if completed.get("status") != "succeeded":
            raise H3ServeError("断点恢复没有生成最终视频")
        pbar.update_absolute(100, 100)
        return _download_job_video(连接, job_id, preview=False)


class H3ServeConnection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "server_url": ("STRING", {"default": "http://127.0.0.1:8090"}),
            "api_key": ("STRING", {"default": ""}),
            "poll_seconds": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 10.0, "step": 0.25}),
        }}

    RETURN_TYPES = (CONNECTION,)
    RETURN_NAMES = ("连接",)
    FUNCTION = "connect"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Engine selection/readiness is external state. Never let ComfyUI cache
        # yesterday's connection result after the operator exits or switches.
        return float("nan")

    def connect(self, server_url, api_key, poll_seconds):
        connection = {"server_url": server_url.rstrip("/"), "api_key": api_key, "poll_seconds": poll_seconds}
        client = _client(connection)
        _require_ready_engine(client)
        return (connection,)


class H3ServeReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {"已有参考": (REFERENCES,)}}

    RETURN_TYPES = (REFERENCES,)
    RETURN_NAMES = ("参考素材",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/参考素材"

    def append(self, image, **kwargs):
        current = kwargs.get("已有参考") or {"images": (), "audios": ()}
        images = tuple(current.get("images", ())) + (_image_png(image),)
        if len(images) > 9:
            raise ValueError("Ref2VA 最多支持9张参考图片")
        return ({"images": images, "audios": tuple(current.get("audios", ()))},)


class H3ServeReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}, "optional": {"已有参考": (REFERENCES,)}}

    RETURN_TYPES = (REFERENCES,)
    RETURN_NAMES = ("参考素材",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/参考素材"

    def append(self, audio, **kwargs):
        current = kwargs.get("已有参考") or {"images": (), "audios": ()}
        audios = tuple(current.get("audios", ())) + (_audio_wav(audio),)
        if len(audios) > 3:
            raise ValueError("Ref2VA 最多支持3段参考音频")
        return ({"images": tuple(current.get("images", ())), "audios": audios},)


class H3ServePresetGenerate:
    EXPOSE_PREVIEW_OUTPUT = False
    RAW_PROMPT = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "连接": (CONNECTION,),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "resolution": (["360p", "480p", "720p", "1080p"], {"default": "480p"}),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "sampling_steps": ("INT", {"default": 8, "min": 4, "max": 20, "step": 1}),
                "acceleration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "model_variant": (list(MODEL_VARIANTS), {"default": "原始权重"}),
                "preview_mode": (list(PREVIEW_MODES), {"default": "关闭"}),
                "seed": ("INT", {"default": 4404, "min": 0, "max": 0xffffffffffffffff}),
                "background_music": (["关闭", "开启"], {"default": "关闭"}),
                "upscale": (["关闭", "720p", "1080p", "2K"], {"default": "关闭"}),
            },
            "optional": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",), "参考素材": (REFERENCES,)},
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    # The service has already produced and downloaded the final MP4.  Mark the
    # generation node itself as an output node so ComfyUI previews that file
    # directly; routing it through core SaveVideo would encode a second copy and
    # is incompatible with DynamicCombo serialization across ComfyUI releases.
    OUTPUT_NODE = True

    def generate(self, 连接, prompt, resolution, aspect_ratio, duration_seconds,
                 sampling_steps, acceleration, model_variant, preview_mode, seed,
                 upscale, first_frame=None, last_frame=None, **kwargs):
        background_music = kwargs.pop("background_music", "原样")
        variant = MODEL_VARIANTS[model_variant]
        lower, upper = (4, 8) if variant == "lora" else (8, 20)
        if not lower <= int(sampling_steps) <= upper:
            raise H3ServeError(
                f"{model_variant} 的总采样步数必须在 {lower}–{upper} 之间"
            )
        checkpoint_job_id = str(kwargs.get("断点任务ID") or "").strip()
        checkpoint_action = str(kwargs.get("断点动作") or "新建任务")
        if preview_mode == "开启" and checkpoint_job_id:
            if checkpoint_action == "继续生成":
                return _resume_interactive_checkpoint(连接, checkpoint_job_id)
            if checkpoint_action == "放弃生成":
                return _discard_interactive_checkpoint(连接, checkpoint_job_id)
            raise H3ServeError("断点任务正在等待选择：请点击继续生成或放弃生成")
        fields = {
            "mode": "preset", "prompt": (
                prompt if self.RAW_PROMPT
                else _prompt_with_music_policy(prompt, background_music)
            ),
            "resolution": resolution, "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds, "seed": seed,
            "upscale_enabled": upscale != "关闭",
            "sampling_steps": int(sampling_steps),
            "acceleration": _acceleration_value(acceleration),
            "model_variant": variant,
            "preview_mode": "off" if preview_mode == "开启" else PREVIEW_MODES[preview_mode],
        }
        if preview_mode == "开启":
            preview_step = int(kwargs.get("预览位置", 6))
            if not 1 <= preview_step < int(sampling_steps):
                raise H3ServeError("预览位置必须大于0且小于总采样步数")
            fields.update({
                # Stop the formal trajectory and release the worker. This is
                # deliberately not the old auto-continue preview branch.
                "execution_mode": "checkpoint",
                "checkpoint_step": preview_step,
                "checkpoint_retain": True,
                "checkpoint_preview": True,
                "checkpoint_preview_resolution": {
                    "原分辨率": "source", "360p": "360p",
                    "480p": "480p", "720p": "720p",
                }[kwargs.get("预览分辨率", "原分辨率")],
                "checkpoint_preview_steps": int(
                    kwargs.get("LoRA预览步数", 4)
                ),
            })
        if upscale != "关闭":
            fields.update({
                "upscale_mode": "basic",
                "upscale_resolution": "2k" if upscale == "2K" else upscale,
            })
        _add_reference_resolution_fields(fields, kwargs)
        references = kwargs.get("参考素材") or _direct_references(kwargs)
        if preview_mode == "开启":
            return _run_interactive_checkpoint(
                连接, fields, first_frame=first_frame, last_frame=last_frame,
                references=references,
            )
        return _run(
            连接, fields, first_frame=first_frame, last_frame=last_frame,
            references=references, expose_preview_output=self.EXPOSE_PREVIEW_OUTPUT,
        )


class H3ServeAdvancedGenerate:
    RAW_PROMPT = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "连接": (CONNECTION,),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 192, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 480, "min": 192, "max": 1920, "step": 32}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "sampling_steps": ("INT", {"default": 8, "min": 4, "max": 20, "step": 1}),
                "acceleration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "model_variant": (list(MODEL_VARIANTS), {"default": "原始权重"}),
                "preview_mode": (list(PREVIEW_MODES), {"default": "关闭"}),
                "preview_branch_steps": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "seed": ("INT", {"default": 4404, "min": 0, "max": 0xffffffffffffffff}),
                "background_music": (["关闭", "开启"], {"default": "关闭"}),
                "upscale": (["关闭", "720p", "1080p", "2K"], {"default": "关闭"}),
            },
            "optional": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",), "参考素材": (REFERENCES,)},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "本地路径", "任务ID", "任务详情")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def generate(self, 连接, prompt, width, height, duration_seconds,
                 sampling_steps, acceleration, model_variant, preview_mode,
                 preview_branch_steps, seed, upscale,
                 first_frame=None, last_frame=None, **kwargs):
        background_music = kwargs.pop("background_music", "原样")
        engine, _ = _require_ready_engine(_client(连接))
        variant = MODEL_VARIANTS[model_variant]
        lower, upper = (4, 8) if variant == "lora" else (8, 20)
        if not lower <= int(sampling_steps) <= upper:
            raise H3ServeError(
                f"{model_variant} 的总采样步数必须在 {lower}–{upper} 之间"
            )
        fields = {
            "mode": "advanced", "prompt": (
                prompt if self.RAW_PROMPT
                else _prompt_with_music_policy(prompt, background_music)
            ),
            "width": width, "height": height, "duration_seconds": duration_seconds,
            "sampling_steps": int(sampling_steps),
            "acceleration": _acceleration_value(acceleration),
            "seed": seed, "upscale_enabled": upscale != "关闭",
            "model_variant": variant,
            "preview_mode": PREVIEW_MODES[preview_mode],
            "preview_branch_steps": preview_branch_steps,
        }
        if upscale != "关闭":
            fields.update({
                "upscale_mode": "basic",
                "upscale_resolution": "2k" if upscale == "2K" else upscale,
            })
        _add_reference_resolution_fields(fields, kwargs)
        return _run(连接, fields, first_frame=first_frame, last_frame=last_frame,
                    references=kwargs.get("参考素材") or _direct_references(kwargs),
                    include_metadata=True)


def _interactive_preset_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Give both creator workflows the same prompt and preview contract."""

    ordered = {}
    for name, field in schema["required"].items():
        if name == "prompt":
            # A standard Text (Multiline) node can feed and share the prompt.
            ordered[name] = ("STRING", {"forceInput": True})
        elif name == "preview_mode":
            ordered[name] = (["关闭", "开启"], {"default": "关闭"})
            ordered["预览位置"] = (
                "INT", {"default": 6, "min": 1, "max": 19, "step": 1},
            )
            ordered["预览分辨率"] = (
                ["原分辨率", "360p", "480p", "720p"],
                {"default": "原分辨率"},
            )
            ordered["LoRA预览步数"] = (
                "INT", {"default": 4, "min": 1, "max": 8, "step": 1},
            )
        else:
            ordered[name] = field
    # Append internal state after every existing widget so older saved
    # workflows keep their positional widget values intact.
    ordered["断点任务ID"] = ("STRING", {"default": ""})
    ordered["断点动作"] = (
        ["新建任务", "等待选择", "继续生成", "放弃生成"],
        {"default": "新建任务"},
    )
    schema["required"] = ordered
    return schema


class H3ServeFL2VAPresetGenerate(H3ServePresetGenerate):
    """Preset FL2VA surface: text and optional first/last anchors only."""

    EXPOSE_PREVIEW_OUTPUT = True
    RAW_PROMPT = True
    RETURN_TYPES = ("VIDEO", "VIDEO")
    RETURN_NAMES = ("最终视频", "预览视频")

    @classmethod
    def INPUT_TYPES(cls):
        schema = _interactive_preset_schema(super().INPUT_TYPES())
        schema["required"].pop("background_music", None)
        schema["optional"] = {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }
        return schema


class H3ServeRef2VAPresetGenerate(H3ServePresetGenerate):
    """Preset Ref2VA surface with direct, numbered image/video/audio inputs."""

    EXPOSE_PREVIEW_OUTPUT = True
    RAW_PROMPT = True
    RETURN_TYPES = ("VIDEO", "VIDEO")
    RETURN_NAMES = ("最终视频", "预览视频")

    @classmethod
    def INPUT_TYPES(cls):
        # Reference downscaling is a service-wide policy configured on 8090;
        # the creator node should not duplicate those global controls.
        schema = _interactive_preset_schema(super().INPUT_TYPES())
        schema["required"].pop("background_music", None)
        schema["optional"] = _direct_reference_inputs()
        return schema


class H3ServeFL2VAAdvancedGenerate(H3ServeAdvancedGenerate):
    """Advanced FL2VA surface: text and optional first/last anchors only."""

    RAW_PROMPT = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"].pop("background_music", None)
        schema["optional"] = {
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }
        return schema


class H3ServeRef2VAAdvancedGenerate(H3ServeAdvancedGenerate):
    """Advanced Ref2VA surface with direct, numbered image/video/audio inputs."""

    RAW_PROMPT = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = _add_reference_resolution_inputs(super().INPUT_TYPES())
        schema["required"].pop("background_music", None)
        schema["optional"] = _direct_reference_inputs()
        return schema


NODE_CLASS_MAPPINGS = {
    "H3ServeConnection": H3ServeConnection,
    "H3ServeFL2VAPresetGenerate": H3ServeFL2VAPresetGenerate,
    "H3ServeRef2VAPresetGenerate": H3ServeRef2VAPresetGenerate,
    "H3ServeFL2VAAdvancedGenerate": H3ServeFL2VAAdvancedGenerate,
    "H3ServeRef2VAAdvancedGenerate": H3ServeRef2VAAdvancedGenerate,
    "H3ServeFL2VACheckpointSubmit": H3ServeFL2VACheckpointSubmit,
    "H3ServeRef2VACheckpointSubmit": H3ServeRef2VACheckpointSubmit,
    "H3ServeCheckpointPreview": H3ServeCheckpointPreview,
    "H3ServeCheckpointResume": H3ServeCheckpointResume,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ServeConnection": "H3 Serve · 连接服务",
    "H3ServeFL2VAPresetGenerate": "H3 Serve · FL2VA简单生成",
    "H3ServeRef2VAPresetGenerate": "H3 Serve · Ref2VA简单生成",
    "H3ServeFL2VAAdvancedGenerate": "H3 Serve · FL2VA高级生成",
    "H3ServeRef2VAAdvancedGenerate": "H3 Serve · Ref2VA高级生成",
    "H3ServeFL2VACheckpointSubmit": "H3 Serve · FL2VA断点任务",
    "H3ServeRef2VACheckpointSubmit": "H3 Serve · Ref2VA断点任务",
    "H3ServeCheckpointPreview": "H3 Serve · 查看断点预览",
    "H3ServeCheckpointResume": "H3 Serve · 恢复断点任务",
}
