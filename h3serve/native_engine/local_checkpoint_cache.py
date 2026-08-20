"""Recoverable WSL-native cache for large repeatedly streamed checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


QWEN_SHA256 = "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
QWEN_LAYER_CACHE_SCHEMA = 1


def drop_file_page_cache(path: Path) -> None:
    """Best-effort release of reclaimable streaming pages on compact hosts."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            advice = getattr(os, "POSIX_FADV_DONTNEED", None)
            if advice is not None and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(descriptor, 0, 0, advice)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def default_cache_root() -> Path:
    """Return a per-user Linux cache directory suitable for published builds."""

    configured = os.environ.get("H3_SERVE_LOCAL_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "h3serve" / "checkpoints"


def _mount_fstype(path: Path) -> str | None:
    """Return the longest matching mount's filesystem type."""

    resolved = path.resolve()
    best: tuple[int, str] | None = None
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields = left.split()
            mount = Path(fields[4].replace("\\040", " "))
            try:
                resolved.relative_to(mount)
            except ValueError:
                continue
            candidate = (len(str(mount)), right.split()[0])
            if best is None or candidate[0] > best[0]:
                best = candidate
    except (OSError, ValueError, IndexError):
        return None
    return None if best is None else best[1]


def should_localize_checkpoint(source: Path) -> bool:
    """DrvFS/9p random tensor reads are much slower than native ext4."""

    override = os.environ.get("H3_SERVE_LOCALIZE_QWEN")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    return _mount_fstype(source) in {"9p", "drvfs", "fuseblk"}


def materialize_local_checkpoint(
    source: Path,
    *,
    cache_root: Path | None = None,
    reserve_bytes: int = 20 * 1024**3,
    expected_sha256: str = QWEN_SHA256,
) -> Path:
    """Return a byte-identical native-filesystem copy, or the source on failure.

    Validation uses source identity plus final size. The initial copy is atomic;
    interrupted temporary files are never selected. The cache is an optional
    performance artifact and all errors deliberately fall back to the source.
    """

    source = source.resolve()
    if not should_localize_checkpoint(source):
        return source
    root = Path(cache_root).expanduser() if cache_root else default_cache_root()
    target = root / source.name
    metadata = target.with_suffix(target.suffix + ".source.json")
    lock_path = target.with_suffix(target.suffix + ".lock")
    try:
        stat = source.stat()
        root.mkdir(parents=True, exist_ok=True)
        # A user override that points back to /mnt/c cannot improve streaming.
        if _mount_fstype(root) in {"9p", "drvfs", "fuseblk"}:
            return source
        expected = {
            "schema_version": 1,
            "source": str(source),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "sha256": expected_sha256,
        }
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.is_file() and target.stat().st_size == stat.st_size:
                try:
                    if json.loads(metadata.read_text()) == expected:
                        return target
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            free = shutil.disk_usage(root).free
            if free < stat.st_size + reserve_bytes:
                return source
            temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
            try:
                digest = hashlib.sha256()
                with source.open("rb", buffering=8 * 1024**2) as reader, temporary.open(
                    "wb", buffering=8 * 1024**2
                ) as writer:
                    while chunk := reader.read(8 * 1024**2):
                        digest.update(chunk)
                        writer.write(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
                if temporary.stat().st_size != stat.st_size:
                    raise OSError("localized checkpoint size mismatch")
                if expected_sha256 and digest.hexdigest() != expected_sha256:
                    raise OSError("localized checkpoint SHA-256 mismatch")
                temporary.replace(target)
                meta_tmp = metadata.with_suffix(metadata.suffix + f".tmp-{os.getpid()}")
                meta_tmp.write_text(json.dumps(expected, sort_keys=True) + "\n")
                meta_tmp.replace(metadata)
                drop_file_page_cache(source)
                drop_file_page_cache(target)
                return target
            finally:
                temporary.unlink(missing_ok=True)
    except OSError:
        return source


def materialize_qwen_layer_cache(
    source: Path,
    *,
    cache_root: Path | None = None,
    layers: int = 50,
    reserve_bytes: int = 12 * 1024**3,
) -> Path | None:
    """Build an execution-ordered Qwen decoder cache for compact hosts.

    The published Qwen safetensors file groups equal tensor names across all
    decoder layers.  H3 evaluates one complete decoder layer at a time, so the
    compact path otherwise jumps through a 14+ GiB file for every projection.
    These byte-identical per-layer safetensors make each layer a short
    sequential read.  They are a recoverable disk cache, never model weights
    and never resident host memory.

    Creation is atomic and one layer is materialized at a time.  Any failure
    returns ``None`` so inference can retain the original safe streaming path.
    """

    override = os.environ.get("H3_SERVE_QWEN_LAYER_CACHE")
    if override is not None and override.strip().lower() in {
        "0", "false", "no", "off",
    }:
        return None
    source = source.resolve()
    root = Path(cache_root).expanduser() if cache_root else default_cache_root()
    cache_key = f"{source.stem}.layers-v{QWEN_LAYER_CACHE_SCHEMA}-{QWEN_SHA256[:12]}"
    target = root / cache_key
    manifest_path = target / "manifest.json"
    lock_path = root / f"{cache_key}.lock"
    try:
        stat = source.stat()
        root.mkdir(parents=True, exist_ok=True)
        if _mount_fstype(root) in {"9p", "drvfs", "fuseblk"}:
            return None
        expected = {
            "schema_version": QWEN_LAYER_CACHE_SCHEMA,
            "source_size": stat.st_size,
            "source_sha256": QWEN_SHA256,
            "layers": int(layers),
        }

        def ready() -> bool:
            try:
                manifest = json.loads(manifest_path.read_text())
                return manifest == expected and all(
                    (target / f"layer-{index:02d}.safetensors").is_file()
                    for index in range(layers)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                return False

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if ready():
                return target
            # Decoder layers account for roughly 13 GiB.  Use the source size
            # as a conservative bound without first touching every data page.
            if shutil.disk_usage(root).free < stat.st_size + reserve_bytes:
                return None
            from safetensors import safe_open
            from safetensors.torch import save_file

            temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}-", dir=root))
            try:
                with safe_open(source, framework="pt", device="cpu") as checkpoint:
                    all_keys = tuple(checkpoint.keys())
                    for index in range(layers):
                        prefix = f"model.layers.{index}."
                        keys = tuple(key for key in all_keys if key.startswith(prefix))
                        if not keys:
                            raise OSError(f"Qwen layer {index} is absent")
                        tensors = {
                            key: checkpoint.get_tensor(key).contiguous()
                            for key in keys
                        }
                        save_file(
                            tensors,
                            temporary / f"layer-{index:02d}.safetensors",
                        )
                        del tensors
                (temporary / "manifest.json").write_text(
                    json.dumps(expected, sort_keys=True) + "\n"
                )
                if target.exists():
                    shutil.rmtree(target)
                temporary.replace(target)
                drop_file_page_cache(source)
                return target
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
    except (OSError, RuntimeError, ValueError):
        return None


__all__ = [
    "QWEN_SHA256", "default_cache_root", "drop_file_page_cache",
    "materialize_local_checkpoint", "materialize_qwen_layer_cache",
    "should_localize_checkpoint",
]
