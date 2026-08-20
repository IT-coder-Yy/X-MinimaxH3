# X-MinimaxH3

面向单张 NVIDIA RTX 4090（24GB）的 MiniMax H3 本地音视频生成服务。项目包含
Web 创作台、任务队列、HTTP API、ComfyUI 连接器、FL2VA/Ref2VA 的 INT8 与
LoRA 热切换、冻结 Round229 联合稀疏调度、断点预览，以及可选 FlashVSR 超分。
代码同时保留证书化 V19 选择器；没有经过完整输入能力验收的 V19 证书包时，发布版
不会冒充已启用 V19，而是使用已经冻结并实测过的 Round229 路由。

> 当前 GPU 运行时是经过验证的 **Linux x86_64 + RTX 4090 (SM89)** 二进制栈。
> Windows 版本通过 NVIDIA 支持的 WSL2 运行，不宣称原生 Win32/CUDA 兼容。

## 硬件与系统

- NVIDIA RTX 4090 24GB；当前预编译扩展不支持其他 GPU。
- 主机内存最低 64GB，推荐 96GB 或 128GB。
- Linux：推荐 Ubuntu 22.04；其他发行版需自行准备 Python 3.10、可选的
  Python 3.11，以及 ffmpeg。
- Windows：Windows 11、最新 NVIDIA 驱动、WSL2 与 Ubuntu 22.04。
- 磁盘：完整模型约65.8GiB，两个运行环境实测约12GiB，Qwen的WSL高速副本约
  15.7GiB；考虑安装缓存、断点下载和工作空间，建议至少预留120GiB。

## Windows 一键安装

在 PowerShell 中进入仓库目录：

```powershell
.\setup-and-start-windows.ps1 -AcceptModelLicense
```

也可以双击 `setup-and-start-windows.cmd`，随后在交互提示中确认模型许可证。该脚本
会完成安装、权重下载、校验并启动服务。若只需安装而暂不启动，可改用
`install-windows.ps1`或`install-windows.cmd`。脚本会：

1. 自动选择已有 Ubuntu WSL2，若不存在则安装经过验证的Ubuntu 22.04（Windows首次启用WSL时重启后需再运行一次）；
2. 在隔离环境中安装锁定的主服务与 FlashVSR 依赖；
3. 优先从 ModelScope 下载国内可用的权重，失败后依次尝试 HF Mirror 和官方 HF；
4. 对每个权重执行大小与 SHA-256 校验。

为避免WSL在`/mnt/c`解包数GB Python/CUDA文件时极慢，Windows安装器默认把
运行环境、模型和Qwen执行缓存保存到WSL原生目录`~/.local/share/x-minimaxh3`；
源码、工作空间和生成视频仍保留在当前Windows项目目录，资源管理器可以直接访问。
需要放到其他WSL磁盘时可增加例如`-WslStateDir /data/x-minimaxh3`（必须是绝对的
Linux路径），以后启动和停止时使用同一参数。指定目录会统一包含`runtime/`、
`models/`和`cache/checkpoints/`，不会再把Qwen缓存悄悄写回系统盘。

启动与停止：

```powershell
.\start-windows.ps1
.\stop-windows.ps1
```

浏览器打开 <http://127.0.0.1:8090>。

## Linux 一键安装

```bash
chmod +x *.sh scripts/*.sh
./setup-and-start.sh --accept-model-license
```

该命令安装、下载、校验后在前台启动服务；只安装时运行`./install.sh`。

Ubuntu 会自动补齐系统包。非 Ubuntu 系统请先安装 Python 3.10、Python 3.11
和 ffmpeg，或使用 `--without-upscaler` 省略 Python 3.11/FlashVSR。

## 安装配置

| profile | 内容 | 下载量 |
|---|---|---:|
| `full` | FL2VA + Ref2VA + LoRA + FlashVSR（默认） | 65.8GiB |
| `core` | FL2VA + Ref2VA + LoRA，不含超分 | 59.8GiB |
| `fl2va` | FL2VA + 共享组件 + LoRA | 40.3GiB |
| `ref2va` | Ref2VA + 共享组件 + LoRA | 40.3GiB |
| `upscaler` | 仅 FlashVSR | 6.0GiB |

例如只安装 FL2VA：

```bash
./install.sh --profile fl2va --accept-model-license
```

下载器支持断点续传。只校验已有模型或修复损坏文件：

```bash
runtime/venv/bin/python scripts/download_models.py --profile full --verify-only
runtime/venv/bin/python scripts/download_models.py --profile full \
  --source auto --accept-model-license --repair
```

公开 ModelScope 模型不要求 Token。LoRA 当前会由 `hf-mirror.com` 或 Hugging
Face 下载；若访问私有仓库，可通过 `HF_TOKEN` 环境变量提供凭据。

## 自检

```bash
./scripts/doctor.py --profile full
# 发布或疑难排查时，重新哈希全部约 66GiB 权重：
./scripts/doctor.py --profile full --full-hash
```

自检会明确报告 Python/Torch/CUDA、RTX 4090 SM89、运行时源码、预编译稀疏
扩展、模型和 FlashVSR 的状态。启动器也会实际导入扩展；ABI 不兼容不会被
悄悄当成可用加速。

## 使用方式

启动后先在控制台选择 FL2VA 或 Ref2VA 模型族。INT8/LoRA 在同一模型族会话
内按任务动态切换，不需要启动四个独立服务。

- Web：<http://127.0.0.1:8090>
- OpenAPI/HTTP API：服务启动后访问 <http://127.0.0.1:8090/openapi.json>；
  该机器可读描述是接口的权威合同。
- ComfyUI：连接器在 `integrations/comfyui`；安装说明见
  [`integrations/comfyui/README.md`](integrations/comfyui/README.md)。

工作空间、上传文件、断点 latent 和生成视频都在用户选择的 workspace 中，
不会写进 Git。默认工作空间为 `workspace/default`。

### 参考素材自动降分辨率

控制台设置中可统一指定参考图片和参考视频的分辨率上限，初始值分别为720P和
360P；也可选择360P、480P、720P或保持原分辨率。这里的档位是**短边像素上限**，
不是新的画幅：处理只会等比例降低像素分辨率，不裁切、不拉伸、不补边、不改变
构图、宽高比、视频时长或帧率，也不会放大本来较小的素材。例如640×480的4:3
视频选择360P后得到480×360，而不是640×360或480×384。

该控制台设置是Web、REST API与ComfyUI共同继承的服务端默认值。API可以通过
`reference_image_resolution`和`reference_video_resolution`为单次任务覆盖；
ComfyUI节点选择“使用服务端设置”时不发送覆盖字段。

### 分辨率与时长上限

控制台同一个设置窗口提供“生成任务上限”。部署者可以为每一个
“分辨率 × 画面比例”独立设置1–15秒的最长提交时长。例如可以将
`720p/16:9`设为15秒、`1080p/16:9`设为8秒；显存更大的机器也可以自行提高
后者。设置页显示检测到的显存作为参考，但不会用固定公式替用户决定上限。

保存后，Web工作台会立即按对应组合调整时长滑块；REST API和ComfyUI提交也由
同一服务端矩阵校验。该设置只改变接单边界，不会裁切、拉伸或者降低输出分辨率。
当前矩阵可从`GET /api/v1/options`的`duration.max_by_preset`读取，也可以通过
`PUT /api/v1/settings/generation-limits`更新。

## 下载来源与许可证

`models/manifest.json` 是唯一模型合同。ModelScope 只是传输镜像，最终内容仍
必须匹配锁定的上游文件 SHA-256。详细第三方来源见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

MiniMax H3 不是宽松开源许可证。随包提供的
[`MiniMax H3 Community License`](third_party_licenses/MiniMax-H3-COMMUNITY-LICENSE)
规定了适用地域、可接受使用、下游用户约束、安全措施、商业授权和再分发义务；根目录
`NOTICE` 是其要求的再分发声明。尤其需要注意，协议定义的适用地域排除了欧盟、英国、
韩国和美国，向这些地区公开提供仓库或服务前应先取得 MiniMax 的单独许可。把本项目
部署为他人可访问的服务时，运营者还必须自行落实协议要求的用户条款与安全措施。

**发布阻断项：** 项目所有者尚未为原创代码选择许可证。公开推送 GitHub 前
必须用正式 `LICENSE` 替换 `LICENSE-DECISION-REQUIRED.md`。这不会阻止本地
工程验证，但会阻止我们把当前目录宣称为法律意义上的公开发行版。

## 已知边界

- 只验证 RTX 4090；其他 Ada、Ampere、Blackwell 卡不会自动复用 SM89 二进制。
- Windows 原生 Python 不受支持，必须使用 WSL2。
- Round229联合稀疏调度属于近似推理；加速强度越高，运动因果、音色和清晰度风险越高。
- 证书化V19选择器代码已随包提供，但默认没有启用未经完整能力验收的V19证书包。
- Ref2VA 的参考视频能力由模型本身决定；压缩参考媒体只能降低条件编码压力，
  不能保证复杂视频编辑指令必然遵循。
