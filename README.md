# X-MinimaxH3

面向单张 RTX 4090 的 MiniMax H3 本地音视频生成工具。

> 当前发布的是 **RTX 4090 尝鲜版**。面向其他显卡型号的通用版本正在开发中。

## 交流与反馈

欢迎加入交流群讨论安装、使用和生成效果，也可以添加作者微信直接反馈问题。

| 添加作者 | 加入微信群 |
|:---:|:---:|
| <img src="assets/community/wechat-contact.jpg" width="260" alt="作者微信二维码"> | <img src="assets/community/wechat-group.jpg" width="260" alt="X-MinimaxH3 微信交流群二维码"> |
| 请备注 `X-MinimaxH3` | 群二维码过期后会在这里更新 |

支持：

- FL2VA 首尾帧生成视频
- Ref2VA 图片、音频和视频参考生成
- INT8 与 Turbo LoRA 动态切换
- 稀疏注意力与预测步加速
- 任务队列、断点保存与 LoRA 快速预览
- Web UI、REST API 和 ComfyUI
- 可选 FlashVSR 超分

## 运行要求

- NVIDIA RTX 4090 24GB
- 64GB 内存起步，推荐 96GB 以上
- 至少 120GB 可用磁盘
- Linux x86_64，或 Windows 11 + WSL2

当前预编译加速扩展只验证了 RTX 4090（SM89）。Windows 原生 CUDA 暂不支持。

## Windows 安装

在 PowerShell 中进入项目目录：

```powershell
.\setup-and-start-windows.ps1 -AcceptModelLicense
```

也可以双击 `setup-and-start-windows.cmd`。安装完成后打开：

<http://127.0.0.1:8090>

以后启动和停止：

```powershell
.\start-windows.ps1
.\stop-windows.ps1
```

模型和运行环境默认保存在 WSL 的
`~/.local/share/x-minimaxh3`，生成结果保存在当前工作空间。

## Linux 安装

```bash
chmod +x *.sh scripts/*.sh
./setup-and-start.sh --accept-model-license
```

只安装、不启动：

```bash
./install.sh --accept-model-license
```

启动后打开 <http://127.0.0.1:8090>。

## 按需安装

| 配置 | 内容 | 约需下载 |
|---|---|---:|
| `full` | FL2VA、Ref2VA、LoRA、FlashVSR | 65.8GiB |
| `core` | FL2VA、Ref2VA、LoRA | 59.8GiB |
| `fl2va` | FL2VA 与 LoRA | 40.3GiB |
| `ref2va` | Ref2VA 与 LoRA | 40.3GiB |
| `upscaler` | 仅 FlashVSR | 6.0GiB |

例如只安装 FL2VA：

```bash
./install.sh --profile fl2va --accept-model-license
```

中国大陆默认优先从 ModelScope 下载公开权重，不需要 Token。下载支持断点续传和
SHA-256 校验。

## 使用

启动服务后，在首页选择 FL2VA 或 Ref2VA。INT8 与 LoRA 由每个任务自行选择，
不需要启动多个后端进程。

- Web UI：<http://127.0.0.1:8090>
- API 合同：<http://127.0.0.1:8090/openapi.json>
- ComfyUI 安装：[integrations/comfyui/README.md](integrations/comfyui/README.md)

工作空间保存上传素材、任务记录、断点和生成视频。默认位置为
`workspace/default`，也可以在 Web UI 中选择其他目录。

## 设置

Web UI 设置页可以统一配置：

- MiMo API Key
- 参考图片和参考视频的压缩上限
- 每种分辨率与画面比例允许生成的最长时长
- 主机内存策略

这些限制同时作用于 Web UI、REST API 和 ComfyUI。参考媒体只会等比例降低
分辨率，不会裁切、拉伸或改变时长。

## 自检

```bash
./scripts/doctor.py --profile full
```

完整校验所有模型文件：

```bash
./scripts/doctor.py --profile full --full-hash
```

## 注意事项

- 加速力度越高，生成速度越快，动作连续性、音色和画面稳定性的风险也越高。
- Ref2VA 的参考视频编辑能力存在模型边界，复杂指令不保证完全遵循。
- 1080P 长视频可能因显存不足进入低功耗卸载状态；请在设置中按本机显存调整时长上限。

## 许可证

项目原创代码使用 [The Unlicense](LICENSE)，允许自由使用、修改和分发。

MiniMax H3、模型权重和第三方组件仍受各自许可证约束。下载或分发前请阅读：

- [MiniMax H3 Community License](third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE)
- [第三方组件与来源](THIRD_PARTY_NOTICES.md)
