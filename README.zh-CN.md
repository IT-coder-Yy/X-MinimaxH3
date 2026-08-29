# X-MinimaxH3

[English](README.md) · **简体中文**

X-MinimaxH3 是面向单张 NVIDIA SM89 GPU 优化的 MiniMax H3 本地视频生成服务。
项目通过统一的中英文 Web 控制台和 REST API 提供 FL2VA、Ref2VA、Base/LoRA
热切换、分档显存执行、断点预览以及 H3 原生二次采样。

> 本源码仓库不分发模型权重、用户上传文件、中间 latent 或生成视频。

## 主要能力

- 统一用户控制面：只需要设定总采样步数和连续的 `0–100` 加速力度。
- Base 联合调度器统一安排真实 DiT 计算、预测计算以及逐步逐层 Attention 预算。
- 六个相互隔离的启动器：FL2VA 与 Ref2VA 分别对应逻辑 24GB INT8、16GB
  INT8 和 8GB W4A8 档位。
- 在相应档位允许范围内提供 360p–1080p 原生生成；INT8 档位可进行最高
  1440p 的 H3 原生二次采样。
- FL2VA 支持纯文本、首帧、尾帧和首尾帧约束。
- Ref2VA 支持参考图片、参考视频和独立参考音频。
- 内置 Larry Turbo 与三套任务型 LightX2V LoRA 配置。
- 可恢复断点任务和固定低成本预览。
- 串行 GPU 队列、任务取消、历史记录和每秒硬件监控。
- 可选 ComfyUI HTTP 连接器，不会在 ComfyUI 中重复加载一套 H3。
- 控制台和主要用户文档均支持英文与简体中文。

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

## 效果实测对比

以下本机实测对比展示了同一素材的 720P 视频直出与 1440P H3 原生二次采样效果。
分割线会在 5 秒、10 秒和 15 秒样例中横向滑动。测试配置为 Intel Core i9（14代）、
128GB 内存、RTX 4090 24GB，使用 INT8 FL2VA。

<p align="center">
  <a href="https://www.bilibili.com/video/BV1Sm4C6VEhf/">▶ 前往哔哩哔哩观看中文效果对比视频</a>
</p>

## 交流与反馈

欢迎加入交流群讨论安装、使用和生成效果，也可以添加作者微信直接反馈问题。

| 添加作者 | 加入微信群 |
|:---:|:---:|
| <img src="assets/community/wechat-contact.jpg" width="260" alt="作者微信二维码"> | <img src="assets/community/wechat-group.jpg" width="260" alt="X-MinimaxH3 微信交流群二维码"> |
| 请备注 `X-MinimaxH3` | 群二维码过期后会在这里更新 |

对于能够复现的 Bug 和功能建议，请优先提交到
[GitHub Issues](https://github.com/PullMyBoots/X-MinimaxH3/issues)，方便长期检索问题和解决方案。

## 已验证平台

| 组件 | 已验证配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，SM89 |
| 系统 | Linux x86-64 / WSL2 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA Runtime | 13.0 |
| 服务编译工具链 | CUDA 13.3 |
| 主机内存 | 建议至少 64GB 有效内存；长视频和高分辨率任务建议更多 |

其他 GPU 架构尚未作为发布平台验收。逻辑 8GB/16GB 路线是在 SM89 上通过
显存硬上限测试的；同容量物理显卡仍需要单独进行设备级验证。

## 快速部署

### 全新安装

下面的命令会创建运行环境、检出固定版本的上游源码，并下载
`models/manifest.json` 声明的全部权重：

```bash
git clone <你的 GitHub 仓库地址> X-MinimaxH3
cd X-MinimaxH3
./setup.sh --download-models --accept-model-license
./run.sh
```

`--accept-model-license` 仅表示你确认已经阅读并接受各权重发布者的许可证，
不会修改或替代模型本身的许可条款。

### 复用现有环境与权重

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./run.sh
```

`vendor` 目录必须包含 `MiniMax-H3/` 和 `LightX2V/`。如果兼容的稀疏算子
编译产物已经位于 `vendor` 同级的标准 `extensions/` 目录，可以省略
`--sparse-build-dir`。

浏览器打开 <http://127.0.0.1:8090>。停止服务：

```bash
./stop.sh
```

在 WSL2 中，`./run.sh` 会自动把热运行源码与缓存同步到 Linux 文件系统，避免
服务反复通过 `/mnt/c` 导入大量 Python 文件和访问元数据。

## 安装与发布验收

快速检查、完整权重/源码版本检查和回归测试入口分别是：

```bash
./doctor.sh
./doctor.sh --full
./test.sh
```

当前发布版的验收结果：

- 709 项发布回归测试通过，4 项跳过，0 项失败；另有24项ComfyUI连接器测试全部通过；
- 清单声明的 12 个模型工件全部通过精确大小和 SHA-256 检查；
- 六个启动器以及 SM89 INT8/W4A8 内核 smoke test 全部通过；
- Base FL2VA、LightX2V FL2VA 4步/8步、LightX2V Ref2VA 4步均完成真实
  MP4 生成。

详细命令、耗时和结果哈希见 [VALIDATION.md](VALIDATION.md)。

## 资源档位

| 档位 | 权重 | 原生首遍生成 | H3 原生二次采样 |
|---|---|---|---|
| 24GB | INT8 | 最高 1080p × 15秒 | 最高 1440p |
| 16GB | INT8 | 实验性最高 1080p × 15秒 | 最高 1440p |
| 8GB | W4A8 | 最高 720p × 15秒 | 最高 1080p |

超出当前后端能力边界的任务会被明确拒绝，不会静默切换到其他后端。运行服务
返回的分辨率、时长和参考媒体限制才是当前档位的最终有效边界。

设置页可把H3二次采样的时间上下文设为68–362帧。短窗口通常降低单个DiT窗口的
延迟，长窗口保留更多动作与身份连续性；H3时间相位对齐、17帧Overlap、latent
交叉融合以及显存不足时的安全缩窗仍由后端自动处理。

## LoRA 配置

| LoRA | 任务族 | 标定步数 |
|---|---|---:|
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8，默认6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

LightX2V 的 FL2VA 与 Ref2VA LoRA 属于不同任务族，不能互换。设置页会递归扫描
当前模型仓库的 `loras/` 目录，并只开放通过兼容性检查的权重。

## ComfyUI

完整说明见[中文 ComfyUI 指南](integrations/comfyui/README.md)或
[English ComfyUI guide](integrations/comfyui/README.en.md)。

先启动 X-MinimaxH3 并在控制台选择启动器，然后执行：

```bash
./integrations/comfyui/start_comfyui.sh
```

打开 <http://127.0.0.1:8188>。示例工作流位于
`integrations/comfyui/example_workflows/`，同时提供中文和英文版本。连接器只调用
同一个 8090 HTTP 服务，
不会在 ComfyUI 进程中额外占用一套 H3 模型显存。

## 目录结构

```text
h3serve/                 Web/API、队列、调度器与 H3 原生运行时
backends/                SM89 算子和经过审计的窄化二进制运行时
static/                  中英文 Web 控制台
integrations/comfyui/    可选连接器与示例工作流
models/manifest.json     权重来源、大小和 SHA-256 契约
scripts/                 安装、启动、验收和研究工具
tests/                   单元、契约与运行时回归测试
docs/                    用户、部署与架构文档
```

## 详细文档

- [中文用户指南](docs/USER_GUIDE.zh-CN.md)
- [中文部署指南](docs/DEPLOYMENT.zh-CN.md)
- [English user guide](docs/USER_GUIDE.en.md)
- [English deployment guide](docs/DEPLOYMENT.en.md)
- [原生引擎架构](docs/NATIVE_ENGINE_ARCHITECTURE.md)
- [第三方组件声明](THIRD_PARTY_NOTICES.md)
- [发布验收记录](VALIDATION.md)

## 安全说明

服务默认只监听 `127.0.0.1`。绑定到非回环地址前必须设置足够强的
`H3_SERVE_API_KEY`。服务自身不提供 TLS 或多租户隔离；网络部署应使用可信的
反向代理。详见 [SECURITY.md](SECURITY.md)。

## 致谢

X-MinimaxH3 建立在 MiniMax H3 社区多项重要工作的基础上，特别感谢：

- [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)：
  本项目的原生二次采样规划器改造了其时间/空间分块、Overlap 与融合拼接设计，
  并在此基础上实现了脱离 ComfyUI 的运行时、整幅画布准入、自动资源路由、H3
  时间相位对齐和条件缓存复用。
- [Comfyui Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：
  提供学习式 3D latent 放大网络的架构和公开权重，用于构造 H3 二次采样的高分辨率
  初始 latent。
- [SageAttention](https://github.com/thu-ml/SageAttention)：提供量化稠密
  Attention 算子及其实现基础。本项目围绕该基础进一步完成了 H3 专用布局、量化、
  长序列稳定性与统一调度器集成。

以上上游项目不隶属于 X-MinimaxH3，也不对本项目负责；其原始许可证和声明继续
有效，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

当前版本属于**公开可查看源码**，并不自动构成开源许可证授权。项目原创代码目前
保留所有权利；第三方软件和模型工件继续遵循各自许可证。使用或分发前请阅读
[LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和
[models/manifest.json](models/manifest.json)。
