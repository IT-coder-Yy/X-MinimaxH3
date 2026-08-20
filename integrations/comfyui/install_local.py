from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the H3 Serve connector into ComfyUI")
    parser.add_argument("comfy_root", type=Path)
    args = parser.parse_args()
    root = args.comfy_root.resolve()
    if not (root / "nodes.py").is_file() or not (root / "custom_nodes").is_dir():
        parser.error(f"不是有效的 ComfyUI 根目录：{root}")
    source = Path(__file__).resolve().parent
    target = root / "custom_nodes" / "ComfyUI-H3-Serve-Connector"
    if target.is_symlink() and target.resolve() == source:
        print(f"已安装：{target}")
        return
    if target.exists() or target.is_symlink():
        parser.error(f"目标已存在，请先移除或改名：{target}")
    target.symlink_to(source, target_is_directory=True)
    print(f"已安装：{target} -> {source}")
    print("重启 ComfyUI 后搜索 H3 Serve。")


if __name__ == "__main__":
    main()

