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
    elif target.exists() or target.is_symlink():
        parser.error(f"目标已存在，请先移除或改名：{target}")
    else:
        target.symlink_to(source, target_is_directory=True)
        print(f"已安装：{target} -> {source}")

    workflow_target = root / "user" / "default" / "workflows"
    workflow_target.mkdir(parents=True, exist_ok=True)
    for workflow in sorted((source / "example_workflows").glob("H3_Serve_*.json")):
        installed_workflow = workflow_target / workflow.name
        if (
            installed_workflow.is_symlink()
            and installed_workflow.resolve() == workflow.resolve()
        ):
            print(f"工作流已安装：{installed_workflow}")
        elif installed_workflow.exists() or installed_workflow.is_symlink():
            print(f"保留现有工作流，未覆盖：{installed_workflow}")
        else:
            installed_workflow.symlink_to(workflow)
            print(f"工作流已安装：{installed_workflow} -> {workflow}")
    print("重启 ComfyUI 后搜索 H3 Serve。")


if __name__ == "__main__":
    main()
