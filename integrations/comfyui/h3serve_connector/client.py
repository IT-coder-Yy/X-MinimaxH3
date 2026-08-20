from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable


class H3ServeError(RuntimeError):
    pass


class H3ServeClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        parsed = urllib.parse.urlparse(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise H3ServeError("服务地址必须是完整的 http:// 或 https:// 地址")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Accept": "application/json", **({"X-API-Key": self.api_key} if self.api_key else {})}

    def _request(self, method: str, path: str, *, body: bytes | None = None,
                 content_type: str | None = None, accept_json: bool = True,
                 extra_headers: dict[str, str] | None = None) -> Any:
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read()
                if accept_json:
                    return json.loads(content.decode("utf-8"))
                return content
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise H3ServeError(f"H3 Serve HTTP {error.code}: {detail or error.reason}") from error
        except urllib.error.URLError as error:
            raise H3ServeError(f"无法连接 H3 Serve：{error.reason}") from error

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def submit(self, fields: dict[str, Any], files: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
        if not files:
            body = json.dumps(fields, ensure_ascii=False).encode("utf-8")
            return self._request("POST", "/api/v1/generations", body=body, content_type="application/json")
        boundary = "----H3Serve" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend([
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                str(value).encode("utf-8"), b"\r\n",
            ])
        for name, (filename, content) in files.items():
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            safe_name = Path(filename).name.replace('"', "")
            chunks.extend([
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{safe_name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
                content, b"\r\n",
            ])
        chunks.append(f"--{boundary}--\r\n".encode())
        return self._request(
            "POST", "/api/v1/generations", body=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def cancel(self, job_id: str) -> None:
        try:
            self._request("DELETE", f"/api/v1/jobs/{job_id}")
        except H3ServeError:
            pass

    def decide_preview(self, job_id: str, decision: str) -> dict[str, Any]:
        if decision not in {"continue", "discard"}:
            raise H3ServeError("预览决定必须是 continue 或 discard")
        return self._request(
            "POST", f"/api/v1/jobs/{job_id}/preview/{decision}",
            body=b"", content_type="application/json",
        )

    def resume(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/api/v1/jobs/{job_id}/resume",
            body=b"", content_type="application/json",
        )

    def download_preview(self, job_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        data = self._request(
            "GET", f"/api/v1/jobs/{job_id}/preview", accept_json=False
        )
        temporary.write_bytes(data)
        temporary.replace(destination)
        return destination

    def wait(self, job_id: str, *, poll_seconds: float,
             progress: Callable[[dict[str, Any]], None],
             cancel_check: Callable[[], None], max_wait_seconds: float = 21600) -> dict[str, Any]:
        deadline = time.monotonic() + max_wait_seconds
        while True:
            cancel_check()
            job = self.get(f"/api/v1/jobs/{job_id}")
            progress(job)
            status = job.get("status")
            if status == "succeeded":
                return job
            if status in {"failed", "cancelled"}:
                raise H3ServeError(job.get("error") or f"任务状态：{status}")
            if time.monotonic() >= deadline:
                raise H3ServeError("等待生成结果超时；服务器任务仍可在任务管理页查看")
            time.sleep(max(0.25, poll_seconds))

    def wait_until_stopped(
        self, job_id: str, *, poll_seconds: float,
        progress: Callable[[dict[str, Any]], None],
        cancel_check: Callable[[], None], max_wait_seconds: float = 21600,
    ) -> dict[str, Any]:
        """Wait for either a final video or a persisted formal checkpoint."""

        deadline = time.monotonic() + max_wait_seconds
        while True:
            cancel_check()
            job = self.get(f"/api/v1/jobs/{job_id}")
            progress(job)
            status = job.get("status")
            if status in {"checkpointed", "succeeded"}:
                return job
            if status in {"failed", "cancelled"}:
                raise H3ServeError(job.get("error") or f"任务状态：{status}")
            if time.monotonic() >= deadline:
                raise H3ServeError("等待任务停止超时；服务器任务仍可在任务管理页查看")
            time.sleep(max(0.25, poll_seconds))

    def download(self, job_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        data = self._request("GET", f"/api/v1/jobs/{job_id}/video", accept_json=False)
        temporary.write_bytes(data)
        temporary.replace(destination)
        return destination
