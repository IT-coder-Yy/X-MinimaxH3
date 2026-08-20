#!/usr/bin/env python3
"""Download with resume support and verify the X-MinimaxH3 model contract.

The default source order is designed for Mainland China: ModelScope, the
community Hugging Face mirror, then the official Hugging Face endpoint.
Every file is verified by byte count and SHA-256 before it becomes visible at
its final runtime path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"
CHUNK_SIZE = 16 * 1024 * 1024

PROFILE_ROLES = {
    "full": None,
    "core": {
        "diffusion_model", "reference_diffusion_model", "text_encoder",
        "video_vae", "audio_vae", "turbo_lora",
    },
    "fl2va": {
        "diffusion_model", "text_encoder", "video_vae", "audio_vae",
        "turbo_lora",
    },
    "ref2va": {
        "reference_diffusion_model", "text_encoder", "video_vae",
        "audio_vae", "turbo_lora",
    },
    "upscaler": {
        "flashvsr_diffusion", "flashvsr_lq_projection",
        "flashvsr_temporal_decoder", "flashvsr_prompt_embedding",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def verify(path: Path, artifact: dict, *, quiet: bool = False) -> bool:
    if not path.is_file():
        if not quiet:
            print(f"缺少：{path}", flush=True)
        return False
    expected_size = int(artifact["bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        if not quiet:
            print(f"大小不匹配：{path} ({actual_size} != {expected_size})", flush=True)
        return False
    actual_hash = sha256(path)
    if actual_hash != artifact["sha256"]:
        if not quiet:
            print(f"SHA-256 不匹配：{path}\n  {actual_hash}", flush=True)
        return False
    if not quiet:
        print(f"验证通过：{path.name}", flush=True)
    return True


def source_urls(artifact: dict, source: str) -> list[tuple[str, str]]:
    filename = quote(str(artifact["filename"]), safe="/")
    repo = quote(str(artifact["repo"]), safe="/")
    revision = quote(str(artifact["revision"]), safe="")
    candidates: list[tuple[str, str]] = []
    if source in {"auto", "modelscope"} and artifact.get("modelscope_repo"):
        ms_repo = quote(str(artifact["modelscope_repo"]), safe="/")
        ms_revision = quote(str(artifact.get("modelscope_revision", "master")), safe="")
        ms_filename = quote(
            str(artifact.get("modelscope_filename", artifact["filename"])), safe="/"
        )
        candidates.append((
            "ModelScope",
            f"https://www.modelscope.cn/models/{ms_repo}/resolve/"
            f"{ms_revision}/{ms_filename}",
        ))
    if source in {"auto", "hf-mirror"}:
        endpoint = os.environ.get("HF_MIRROR_ENDPOINT", "https://hf-mirror.com").rstrip("/")
        candidates.append(("HF Mirror", f"{endpoint}/{repo}/resolve/{revision}/{filename}"))
    if source in {"auto", "huggingface"}:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        candidates.append(("Hugging Face", f"{endpoint}/{repo}/resolve/{revision}/{filename}"))
    return candidates


def request_headers(offset: int, url: str) -> dict[str, str]:
    headers = {
        "User-Agent": "X-MinimaxH3/0.7 model-downloader",
        "Accept-Encoding": "identity",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "huggingface" in url:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_one(url: str, target: Path, expected_size: int) -> None:
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.is_file() else 0
    if offset > expected_size:
        raise RuntimeError(f"临时文件大于清单大小，请删除后重试：{part}")
    if offset == expected_size:
        return
    request = Request(url, headers=request_headers(offset, url))
    with urlopen(request, timeout=90) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset > 0 and not append:
            offset = 0
        mode = "ab" if append else "wb"
        downloaded = offset
        started = last_update = time.monotonic()
        with part.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_update >= 1.0:
                    elapsed = max(now - started, 0.001)
                    speed = (downloaded - offset) / elapsed
                    percent = min(100.0, downloaded * 100 / expected_size)
                    print(
                        f"\r  {percent:6.2f}%  {format_bytes(downloaded)} / "
                        f"{format_bytes(expected_size)}  {format_bytes(speed)}/s",
                        end="", flush=True,
                    )
                    last_update = now
    print(flush=True)
    actual = part.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"下载未完成：{part} ({actual} != {expected_size})；下次会断点续传"
        )


def selected_artifacts(manifest: dict, profile: str) -> Iterable[dict]:
    roles = PROFILE_ROLES[profile]
    for artifact in manifest["artifacts"]:
        if roles is None or artifact["role"] in roles:
            yield artifact


def materialize_qwen_cache(path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from h3serve.native_engine.local_checkpoint_cache import (
            materialize_local_checkpoint,
            should_localize_checkpoint,
        )
    except Exception as error:
        print(f"跳过 Qwen Linux 高速副本：{error}", flush=True)
        return
    if should_localize_checkpoint(path):
        print("Qwen 位于 WSL 跨盘路径，正在准备 Linux 原生高速副本…", flush=True)
        selected = materialize_local_checkpoint(path)
        if selected.resolve() == path.resolve():
            print("警告：高速副本未创建；服务可运行，但新提示词编码较慢。", flush=True)
        else:
            print(f"Qwen 高速副本：{selected}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path, nargs="?", default=ROOT / "models")
    parser.add_argument("--profile", choices=tuple(PROFILE_ROLES), default="full")
    parser.add_argument(
        "--source", choices=("auto", "modelscope", "hf-mirror", "huggingface"),
        default="auto",
    )
    parser.add_argument("--accept-model-license", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--repair", action="store_true",
        help="删除清单路径中校验失败的文件并重新下载",
    )
    parser.add_argument("--skip-local-qwen-cache", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and not args.accept_model_license:
        parser.error("下载前必须传入 --accept-model-license")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise SystemExit("不支持的模型清单版本")
    root = args.model_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts = list(selected_artifacts(manifest, args.profile))
    total = sum(int(item["bytes"]) for item in artifacts)
    print(
        f"模型配置：{args.profile}，{len(artifacts)} 个文件，"
        f"总计 {format_bytes(total)}", flush=True,
    )

    failed: list[str] = []
    qwen_target: Path | None = None
    for artifact in artifacts:
        target = root / artifact["install_path"]
        if artifact["role"] == "text_encoder":
            qwen_target = target
        if verify(target, artifact, quiet=True):
            print(f"已有且校验通过：{target.name}", flush=True)
            continue
        if args.verify_only:
            verify(target, artifact)
            failed.append(artifact["install_path"])
            continue
        if target.exists():
            if not args.repair:
                raise SystemExit(
                    f"目标文件存在但校验失败：{target}\n"
                    "确认可以覆盖后增加 --repair。"
                )
            print(f"移除损坏文件：{target}", flush=True)
            target.unlink()

        errors: list[str] = []
        downloaded = False
        for label, url in source_urls(artifact, args.source):
            print(f"下载 {artifact['role']}（{label}）", flush=True)
            for attempt in range(1, 4):
                try:
                    download_one(url, target, int(artifact["bytes"]))
                    part = target.with_name(target.name + ".part")
                    if not verify(part, artifact, quiet=True):
                        raise RuntimeError("字节数或 SHA-256 校验失败")
                    part.replace(target)
                    verify(target, artifact)
                    downloaded = True
                    break
                except (HTTPError, URLError, OSError, RuntimeError) as error:
                    errors.append(f"{label} 第{attempt}次：{error}")
                    print(f"  失败：{error}", flush=True)
                    if attempt < 3:
                        time.sleep(min(2 ** attempt, 8))
            if downloaded:
                break
        if not downloaded:
            raise SystemExit(
                f"所有下载源均失败：{artifact['install_path']}\n" + "\n".join(errors)
            )

    if failed:
        raise SystemExit("模型校验失败：\n  " + "\n  ".join(failed))
    if qwen_target is not None and qwen_target.is_file() and not args.skip_local_qwen_cache:
        materialize_qwen_cache(qwen_target)
    print("模型合同全部就绪。", flush=True)


if __name__ == "__main__":
    main()
