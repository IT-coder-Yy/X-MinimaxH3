# 64GB Windows / WSL2 部署

项目支持的是“64GB 物理内存级别”的工作站，不是 WSL 默认只看到约 32GB 的状态。
发行服务会读取 Linux `MemTotal`、`MemAvailable` 和 cgroup 上限；只有有效上限不少于
约 58GiB、启动时可用内存不少于约 47.4GiB，才会启用经过完整任务验证的 compact
策略。这样会在任务开始前失败，而不是运行很久后 OOM 或大量使用 Swap。

如果 Windows 主机安装了 64GB，而 WSL 中 `free -h` 只显示约 32GB，请在 Windows
用户目录（`%UserProfile%`）创建或修改 `.wslconfig`：

```ini
[wsl2]
memory=58GB
swap=8GB
```

然后在 Windows PowerShell 执行：

```powershell
wsl --shutdown
```

重新进入 WSL 后检查并预检：

```bash
free -h
runtime/venv/bin/python scripts/preflight.py
```

预检输出中的 `host_memory.selected_profile` 应为 `compact`（或内存更大的档位）。
运行其他大型程序会降低 `MemAvailable`；如果物理内存虽为 64GB 但启动时不足，服务
会拒绝装配，关闭占用内存的程序后重试即可。不建议通过增加 Swap 绕过检查：Swap
只能避免部分崩溃，无法维持该档位的实测延迟。

64GB 档还需要约 45GB 的 Linux 原生磁盘空间保存 Qwen高速副本与按执行层排列的
流式缓存。运行时仅滚动保留约两个层，不会把这部分磁盘缓存变成常驻RAM。项目本体和输出
可以放在 `/mnt/c`，但高速副本默认位于 `~/.cache/h3serve/checkpoints`；不要把
`H3_SERVE_LOCAL_MODEL_CACHE` 指回 `/mnt/c`。
