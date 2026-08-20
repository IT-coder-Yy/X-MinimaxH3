"""Per-project storage roots for X-MinimaxH3.

Model weights and reusable compiler caches belong to the installation.  User
assets, job records, generated videos and resumable checkpoints belong to a
workspace.  Keeping that boundary explicit lets one service installation work
with multiple creative projects without mixing their history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    root: Path
    data_dir: Path
    output_dir: Path

    @classmethod
    def at(cls, root: Path) -> "WorkspaceLayout":
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            data_dir=resolved / ".x-minimax-h3",
            output_dir=resolved / "outputs",
        )

    def prepare(self) -> "WorkspaceLayout":
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("workspace path points to a file")
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jobs", "uploads", "logs", "checkpoints", "private"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        return self

    def public(self, *, is_default: bool) -> dict[str, object]:
        return {
            "path": str(self.root),
            "name": self.root.name,
            "is_default": is_default,
            "output_path": str(self.output_dir),
        }


class WorkspaceController:
    """Resolve and persist the workspace selected in the idle engine lobby."""

    def __init__(self, release_root: Path) -> None:
        self.release_root = release_root.resolve()
        self.default_root = (self.release_root / "workspace" / "default").resolve()
        self._selection_file = self.release_root / "runtime" / "control" / "workspace.json"
        self.current = WorkspaceLayout.at(self._load_selected_root()).prepare()

    def _load_selected_root(self) -> Path:
        try:
            document = json.loads(self._selection_file.read_text(encoding="utf-8"))
            value = str(document.get("path", "")).strip()
            if value:
                candidate = Path(value).expanduser()
                if candidate.is_absolute() and (not candidate.exists() or candidate.is_dir()):
                    return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return self.default_root

    def resolve(self, value: str) -> WorkspaceLayout:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("workspace path is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ValueError("workspace path must be absolute")
        return WorkspaceLayout.at(candidate).prepare()

    def activate(self, layout: WorkspaceLayout) -> None:
        self.current = layout
        self._selection_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._selection_file.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"path": str(layout.root)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self._selection_file)

    def public(self, *, switchable: bool) -> dict[str, object]:
        return {
            "current": self.current.public(is_default=self.current.root == self.default_root),
            "default_path": str(self.default_root),
            "switchable": switchable,
        }

    def browse(self, value: str | None = None) -> dict[str, object]:
        if value:
            current = Path(value).expanduser()
            if not current.is_absolute():
                raise ValueError("browse path must be absolute")
            current = current.resolve()
        else:
            current = self.current.root
        if not current.is_dir():
            raise ValueError("browse path is not a directory")
        directories: list[dict[str, str]] = []
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name.casefold())
        except OSError as error:
            raise ValueError(f"cannot browse directory: {error}") from error
        for entry in entries:
            try:
                if entry.is_dir() and not entry.name.startswith("."):
                    directories.append({"name": entry.name, "path": str(entry.resolve())})
            except OSError:
                continue
        parent = current.parent if current.parent != current else None
        return {
            "path": str(current),
            "parent": str(parent) if parent is not None else None,
            "directories": directories,
        }


__all__ = ["WorkspaceController", "WorkspaceLayout"]
