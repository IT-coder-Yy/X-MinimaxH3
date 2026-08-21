# X-MinimaxH3

MiniMax H3 本地音视频生成与推理加速框架。

本项目在 NVIDIA RTX 4090 24GB 上完成开发、测试和针对性调优。

## 效果对比

下面是两组完整样片的轻量动态预览，不展示提示词；点击预览可以播放带原声的完整
MP4。测试平台数据均为 RTX 4090 24GB 上测得的端到端时间，包含模型加载完成后的
推理、解码与成片封装；加速指数越高，框架采用的加速策略越积极。官方未公开的数据
不作推测。

<table>
  <tr>
    <td width="33.33%" align="center">
      <strong>T2V · 官方</strong><br><br>
      <a href="assets/comparison/fl2va-official.mp4"><img src="assets/comparison/fl2va-official.webp" width="100%" alt="T2V 官方生成效果"></a><br>
      <sub>1344×768 · 成片 8 秒<br>推理步数 / 生成耗时：官方未公开</sub>
    </td>
    <td width="33.33%" align="center">
      <strong>T2V · 测试平台 · INT8 加速</strong><br><br>
      <a href="assets/comparison/fl2va-base.mp4"><img src="assets/comparison/fl2va-base.webp" width="100%" alt="T2V X-MinimaxH3 INT8 生成效果"></a><br>
      <sub>1920×1088 · 20 步<br>加速指数 75 · 端到端 499 秒</sub>
    </td>
    <td width="33.33%" align="center">
      <strong>T2V · 测试平台 · LoRA 加速</strong><br><br>
      <a href="assets/comparison/fl2va-lora.mp4"><img src="assets/comparison/fl2va-lora.webp" width="100%" alt="T2V X-MinimaxH3 LoRA 生成效果"></a><br>
      <sub>1920×1088 · 7 步<br>加速指数 40 · 端到端 350 秒</sub>
    </td>
  </tr>
  <tr>
    <td width="33.33%" align="center">
      <strong>Ref2VA · 官方</strong><br><br>
      <a href="assets/comparison/ref-official.mp4"><img src="assets/comparison/ref-official.webp" width="100%" alt="Ref2VA 官方生成效果"></a><br>
      <sub>1024×768 · 成片 10 秒<br>推理步数 / 生成耗时：官方未公开</sub>
    </td>
    <td width="33.33%" align="center">
      <strong>Ref2VA · 测试平台 · INT8 加速</strong><br><br>
      <a href="assets/comparison/ref-base.mp4"><img src="assets/comparison/ref-base.webp" width="100%" alt="Ref2VA X-MinimaxH3 INT8 生成效果"></a><br>
      <sub>1440×1088 · 20 步<br>加速指数 75 · 端到端 544 秒</sub>
    </td>
    <td width="33.33%" align="center">
      <strong>Ref2VA · 测试平台 · LoRA 加速</strong><br><br>
      <a href="assets/comparison/ref-lora.mp4"><img src="assets/comparison/ref-lora.webp" width="100%" alt="Ref2VA X-MinimaxH3 LoRA 生成效果"></a><br>
      <sub>1440×1088 · 7 步<br>加速指数 40 · 端到端 373 秒</sub>
    </td>
  </tr>
</table>

## 视频教程

<p align="center">
  <a href="https://www.bilibili.com/video/BV1Fn8q6JEhX/">
    <img src="assets/tutorial/bilibili-quick-guide.jpg" width="860" alt="让你的 MiniMax H3 快如闪电——X-MinimaxH3 简易教程">
  </a>
</p>

<p align="center">
  <strong>▶ 让你的 MiniMax H3 快如闪电</strong><br>
  <sub>项目部署与使用简易教程 · 约 20 分钟 · BV1Fn8q6JEhX</sub><br>
  <a href="https://www.bilibili.com/video/BV1Fn8q6JEhX/">前往哔哩哔哩观看完整视频</a>
</p>

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

## 用户文档

第一次使用请从 [《X-MinimaxH3 用户手册》](docs/USER_GUIDE.md) 开始。手册按实际
操作顺序说明 Windows/Linux 安装、启动控制台、选择服务、Web 创作、ComfyUI 接入、
REST API 提交与下载，以及常见错误排查。

## 运行要求

- NVIDIA CUDA 显卡；当前开发、测试和调优平台为 RTX 4090 24GB
- 64GB 内存起步，推荐 96GB 以上
- 至少 120GB 可用磁盘
- Linux x86_64，或 Windows 11 + WSL2

当前预编译加速扩展在 RTX 4090（SM89）上完成验证，其他显卡型号尚未完成兼容性
验证。Windows 原生 CUDA 暂不支持。

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
- 完整用户手册：[docs/USER_GUIDE.md](docs/USER_GUIDE.md)

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
