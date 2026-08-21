# X-MinimaxH3 用户手册

这份手册面向第一次安装和使用 X-MinimaxH3 的用户，依次说明安装、启动、
Web 创作、ComfyUI 接入和 REST API 调用。

## 1. 使用前准备

本项目在 NVIDIA RTX 4090 24GB 上完成开发、测试和针对性调优。当前运行要求：

- NVIDIA CUDA 显卡；当前预编译加速扩展仅在 RTX 4090（SM89）上完成验证；
- 64GB 系统内存起步，推荐 96GB 以上；
- 至少 120GB 可用磁盘；
- Linux x86_64，或者 Windows 11 + WSL2。

模型权重不包含在 Git 仓库中。安装器默认优先从 ModelScope 下载，并对每个
文件执行大小和 SHA-256 校验。下载模型前需要阅读并接受模型许可证。

## 2. 安装

### Windows 11

在 PowerShell 中进入项目根目录，然后执行：

```powershell
.\setup-and-start-windows.ps1 -AcceptModelLicense
```

也可以双击项目根目录中的 `setup-and-start-windows.cmd`。脚本会检查 WSL2，
在 Ubuntu WSL 中安装运行环境和模型，并在安装完成后启动 8090 控制台。

如果 Windows 尚未安装 WSL，请先以管理员身份打开 PowerShell：

```powershell
wsl --install
```

系统要求重启时，重启后再次运行 X-MinimaxH3 安装命令。

### Linux

在项目根目录执行：

```bash
chmod +x *.sh scripts/*.sh integrations/comfyui/*.sh
./setup-and-start.sh --accept-model-license
```

如果只想安装而暂时不启动：

```bash
./install.sh --accept-model-license
```

第一次安装数据量较大。可按需要减少下载内容：

```bash
./install.sh --profile fl2va --accept-model-license
./install.sh --profile ref2va --accept-model-license
./install.sh --profile core --accept-model-license
```

`fl2va` 和 `ref2va` 只安装对应服务族与 LoRA；`core` 安装两个服务族但不安装
FlashVSR；`full` 是默认的完整安装。

## 3. 启动与关闭

### Windows

以后启动和关闭不需要重新安装：

```powershell
.\start-windows.ps1
.\stop-windows.ps1
```

也可以双击 `start-windows.cmd` 和 `stop-windows.cmd`。

### Linux 或 WSL

在项目根目录执行：

```bash
./start.sh
```

关闭服务：

```bash
./stop.sh
```

启动成功后访问：

- 控制台：<http://127.0.0.1:8090>
- 存活检查：<http://127.0.0.1:8090/healthz>
- 当前引擎就绪检查：<http://127.0.0.1:8090/readyz>

`start.sh` 只启动统一控制台，不会立即加载模型。这样可以先选择工作空间和
服务族，避免同时加载互斥的 FL2VA 与 Ref2VA 权重。

## 4. 进入服务

打开 8090 控制台后按以下顺序操作：

1. 首次使用时打开设置，选择工作空间。上传素材、任务记录、断点和成片都会
   保存在这个工作空间；未选择时使用项目内置的默认工作空间。
2. 在首页选择服务族：
   - **FL2VA**：文本、首帧、尾帧或首尾帧生成视频；
   - **Ref2VA**：使用图片、独立音频或参考视频生成。
3. 等待页面显示“服务已就绪”。加载期间不要从 Web、API 或 ComfyUI 提交任务。
4. 模型进入热态后，可以在任务级别选择 INT8 原始权重或 Turbo LoRA，不需要
   退出并重新加载服务族。
5. 要切换 FL2VA 与 Ref2VA，先等待当前队列结束，再点击顶部的“切换模型”。

设置页还可以配置 MiMo API Key、主机内存策略、参考媒体压缩上限，以及不同
分辨率和画幅的最长生成时长。这些服务端设置同时约束 Web、API 和 ComfyUI。

## 5. 使用 Web 创作台

模型就绪后，顶部可以在“任务管理”和“创作台”之间切换。

### FL2VA

1. 上传首帧、尾帧，或者同时上传两者；也可以提交纯文本任务。
2. 选择模块化编辑或自由文本：
   - 模块化编辑可以在 Web 内使用 MiMo 辅助润色；
   - 自由文本会把整段 `prompt` 原样发送给 H3。
3. 设置时长、分辨率、画幅、INT8/LoRA、采样步数和加速力度。
4. 如有需要，展开断点预览、超分和背景音乐设置。
5. 点击发送按钮，任务进入统一队列。

### Ref2VA

1. 添加最多 9 张图片、3 段参考视频和 3 段独立参考音频。
2. 在提示词中使用 `<Picture 1>`、`<Video 1>` 和 `<Audio 1>` 等编号引用素材。
3. 参考音频只用于音色；参考视频中的内嵌音轨不会作为音色参考。
4. 选择推理与生成参数后提交任务。

任务管理页会显示排队、加载、运行、断点、成功或失败状态。任务完成后可以直接
预览和下载，成片也保存在当前工作空间的 `outputs/` 目录。

## 6. 接入 ComfyUI

ComfyUI 节点只是 X-MinimaxH3 HTTP 客户端，不在 ComfyUI 进程里加载 H3 权重。
因此必须先启动 8090 控制台、选择服务族，并等待模型就绪。

### 6.1 安装节点

在 X-MinimaxH3 项目根目录执行：

```bash
python3 integrations/comfyui/install_local.py /绝对路径/ComfyUI
```

本项目中的 ComfyUI 示例：

```bash
python3 integrations/comfyui/install_local.py \
  /mnt/c/Users/descfly/Desktop/AI-help-me/minimax-H3-speedup/subprojects-main/main/ComfyUI
```

安装完成后重启 ComfyUI，搜索 `H3 Serve`。

### 6.2 启动 ComfyUI

推荐使用随包启动器，让 ComfyUI 保持 CPU 模式，避免占用 4090 显存：

```bash
./integrations/comfyui/start_comfyui.sh /绝对路径/ComfyUI
```

本项目的完整命令是：

```bash
./integrations/comfyui/start_comfyui.sh \
  /mnt/c/Users/descfly/Desktop/AI-help-me/minimax-H3-speedup/subprojects-main/main/ComfyUI
```

注意命令开头是 `./integrations`，不是 `/integrations`。前者表示项目当前目录，
后者会错误地从 Linux 系统根目录查找文件。

启动后访问 <http://127.0.0.1:8188>。

### 6.3 使用工作流

示例工作流位于：

- `integrations/comfyui/example_workflows/H3_Serve_FL2VA_First_Last.json`
- `integrations/comfyui/example_workflows/H3_Serve_Ref2VA_Multi_Reference.json`

使用步骤：

1. 在 ComfyUI 中载入与当前 8090 服务族对应的 JSON 工作流；
2. “连接服务”节点保持 `http://127.0.0.1:8090`；
3. 如果服务端配置了 API Key，在连接节点填写相同密钥；
4. FL2VA 连接首帧和尾帧，Ref2VA 将图片、视频和音频连接到对应编号端口；
5. 输入最终 `prompt`，设置采样步数和加速力度并运行工作流；
6. 成片下载到 `ComfyUI/output/h3_serve/`，并显示在节点预览区。

ComfyUI 和公共 API 不调用 MiMo，也不会包装、润色或追加提示词字段。它们会把
一个完整 `prompt` 原样交给服务端。

## 7. 使用 REST API

API 是异步任务接口：先提交，获得任务 ID，然后轮询状态，最后下载视频。

完整机器可读合同：<http://127.0.0.1:8090/openapi.json>

如果在服务设置中启用了 API Key，下面所有 `/api/v1` 请求都需要添加：

```bash
-H "X-API-Key: 你的密钥"
```

没有配置服务API Key时可以省略该请求头。MiMo API Key只用于Web提示词辅助，
不是本地服务的调用密钥。

### 7.1 选择服务族

FL2VA：

```bash
curl -X PUT http://127.0.0.1:8090/api/v1/engine \
  -H "Content-Type: application/json" \
  -d '{"engine":"original"}'
```

Ref2VA：

```bash
curl -X PUT http://127.0.0.1:8090/api/v1/engine \
  -H "Content-Type: application/json" \
  -d '{"engine":"reference"}'
```

等待就绪：

```bash
curl http://127.0.0.1:8090/readyz
```

通过API切换服务族前，也必须保证没有正在运行或排队的任务。使用Web首页完成这一步
通常更直观。

### 7.2 提交一个文本任务

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "明亮室内，一个女孩拿起桌上的书包，镜头缓慢拉远。",
    "model_variant": "base",
    "resolution": "480p",
    "aspect_ratio": "16:9",
    "duration_seconds": 5,
    "sampling_steps": 15,
    "acceleration": 50,
    "seed": 4404
  }'
```

`model_variant` 可选 `base` 或 `lora`。INT8采样步数支持5–30，LoRA支持4–10；
`acceleration` 是0–100，数值越高速度越快，近似误差风险也越高。

### 7.3 提交FL2VA首尾帧任务

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generations \
  -F 'prompt=人物从首帧姿态自然运动到尾帧姿态，镜头保持连续。' \
  -F 'first_frame=@/路径/first.png' \
  -F 'last_frame=@/路径/last.png' \
  -F 'resolution=720p' \
  -F 'aspect_ratio=16:9' \
  -F 'duration_seconds=5' \
  -F 'sampling_steps=15' \
  -F 'acceleration=50'
```

### 7.4 提交Ref2VA参考任务

先确认当前服务族是 Ref2VA，然后执行：

```bash
curl -X POST http://127.0.0.1:8090/api/v1/generations \
  -F 'prompt=保持<Picture 1>的人物身份，让她自然转身并向镜头挥手。' \
  -F 'reference_image_1=@/路径/person.png' \
  -F 'reference_audio_1=@/路径/voice.wav' \
  -F 'resolution=480p' \
  -F 'aspect_ratio=16:9' \
  -F 'duration_seconds=5' \
  -F 'sampling_steps=15' \
  -F 'acceleration=50'
```

素材字段按编号扩展为 `reference_image_1..9`、`reference_video_1..3` 和
`reference_audio_1..3`。

### 7.5 查询与下载

提交成功返回的JSON中包含任务 `id`。用它查询状态：

```bash
curl http://127.0.0.1:8090/api/v1/jobs/任务ID
```

常见状态包括：

- `queued`：排队中；
- `starting_backend`：模型正在准备；
- `running`：生成中；
- `checkpointed`：正式断点已保存并释放GPU；
- `awaiting_preview`：等待预览决定；
- `succeeded`：生成成功；
- `failed`：失败，查看返回的 `error`；
- `cancelled`：已取消。

成功后下载：

```bash
curl -L http://127.0.0.1:8090/api/v1/jobs/任务ID/video \
  -o x-minimaxh3-output.mp4
```

取消正在运行或排队的任务：

```bash
curl -X DELETE http://127.0.0.1:8090/api/v1/jobs/任务ID
```

正式断点、预览和恢复接口也记录在 `openapi.json` 中。

## 8. 常见问题

### 8090端口已占用

先在项目根目录执行：

```bash
./stop.sh
```

如果占用端口的是项目以外的进程，先使用 `ss -ltnp 'sport = :8090'` 查看进程，
确认目标后再关闭，或者临时使用其他端口启动。

### ComfyUI提示尚未选择引擎

打开 <http://127.0.0.1:8090>，选择与工作流一致的 FL2VA 或 Ref2VA，并等待
“服务已就绪”。ComfyUI不会替你切换服务族。

### 生成失败

在任务详情中查看完整错误，而不是只看卡片上的短引用编号。常见原因包括模型文件
缺失、输入素材超过数量或时长限制、目标分辨率超过本机配置，以及显存不足。

运行环境自检：

```bash
./scripts/doctor.py --profile full
```

完整校验权重：

```bash
./scripts/doctor.py --profile full --full-hash
```
