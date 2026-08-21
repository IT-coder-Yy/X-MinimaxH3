# H3 Serve Connector for ComfyUI

这是 H3 Video Service 正式发行包内的可选 ComfyUI 客户端集成，源码位于
`integrations/comfyui/`。它不会在 ComfyUI 内加载 H3 权重，而是调用
独立运行的 H3 Serve HTTP API，因此服务端模型热态、任务队列、4090优化和超分策略均会保留。

## 本地安装

```bash
python install_local.py /path/to/ComfyUI
```

正式启动顺序固定为：

1. 在H3 Serve目录执行`./scripts/start.sh`，只启动8090控制台。
2. 打开8090，选择FL2VA或Ref2VA服务族，等待模型加载并显示“服务已就绪”。
3. 再打开ComfyUI并运行工作流。

ComfyUI不负责切换FL2VA/Ref2VA服务族；但每个生成节点可以选择原始权重或LoRA，
该选择只是常驻会话内的任务级热开关。未选择服务族、正在加载或未就绪时，连接节点
会明确拒绝提交任务。

重启 ComfyUI 后搜索 `H3 Serve`。最简单的工作流是：

1. `H3 Serve · 连接服务`：默认地址 `http://127.0.0.1:8090`；如服务设置了密钥则填写。
2. `H3 Serve · 简单生成`：连接提示词与可选首/尾帧，选择分辨率、时长和质量档。
3. 运行生成节点；它会直接保存并预览服务端成片。`视频` 输出仍可连接其他后处理节点。

示例按输入协议拆成两套，避免把互斥端口放在同一个初学者模板里：

- `example_workflows/H3_Serve_FL2VA_First_Last.json`：文本、首帧、尾帧或首尾帧生成。
- `example_workflows/H3_Serve_Ref2VA_Multi_Reference.json`：多图、参考视频与参考音频生成。

生成节点会把服务端成片下载到 ComfyUI 的 `output/h3_serve/` 并直接显示预览，
无需再连接 ComfyUI 的 `SaveVideo` 节点。视频输出仍可继续连接到其他后处理节点。

原始权重与LoRA共用对应服务族的工作流和热会话，在生成节点的`推理路线`中选择。
FL2VA生成节点只暴露`first_frame`/`last_frame`；Ref2VA生成节点直接暴露编号明确的
`Picture 1`～`Picture 9`、`Video 1`～`Video 3`和`Audio 1`～`Audio 3`，两套互斥输入不会出现在同一个节点上。将 ComfyUI 的 `Load Video` 输出直接接到 `Video N`；单段视频为2–15秒、最多三段且合计不超过15秒。其内嵌音轨不参与参考，音色请连接独立的 `Load Audio` 到 `Audio N`。

如果这个ComfyUI只用于调用H3服务，推荐使用随包启动器，使ComfyUI保持CPU模式、不占用
4090显存：

在 X-MinimaxH3 项目根目录执行：

```bash
./integrations/comfyui/start_comfyui.sh /path/to/ComfyUI
```

注意开头是 `./integrations`，不是从系统根目录开始的 `/integrations`。

Ref2VA不再使用素材集合串联节点。将ComfyUI的`Load Image`输出直接接到生成节点的
`Picture N`，将`Load Audio`输出直接接到`Audio N`。编号同时决定提示词中对应的
`<Picture N>`、`<Video N>`与`<Audio N>`。编号决定提示词中对应的引用；例如“保持<Video 1>的骑行轨迹和镜头运动，只将人物服装改为 OL 工装”。

Ref2VA创建工作流不再重复提供`参考图片分辨率`与`参考视频分辨率`；它直接使用控制台保存的
服务端全局设置（初始分别为720P和360P）。
该设置由H3 Serve统一执行：只等比降低像素分辨率，
不裁切、不拉伸、不改变完整构图或视频时长，小素材也不会被放大。Web、REST API和
ComfyUI走的是同一个预处理实现。

FL2VA与Ref2VA都不在ComfyUI里调用MiMo，也不做分镜编译、模板包装或BGM字段追加。
两类生成节点的`prompt`都是普通`STRING`输入端口：示例工作流用内置
`Text (Multiline)`提供一整段最终提示词，连接器逐字原样交给H3 Serve。若提示词需要
H3结构、声音字段或引用标签，由用户全部写在这一个字符串中。MiMo润色只属于Web创作台
的模块化编辑器，不属于ComfyUI或公共生成API。

高级节点只暴露两个推理控制量：`sampling_steps` 是总采样轨迹长度（原始权重8–20，
LoRA 4–8），`acceleration` 是0–100连续加速档位。0表示全部真实步和Dense
Attention；越高表示越小的计算预算。服务自动安排真实/预测位置和逐步逐层Attention，
构图、因果交互、声音与末端细节保护不能在ComfyUI里关闭。LoRA当前只自动调度
Attention，不插入未经人工验收的预测步。连接器轮询服务进度并同步到ComfyUI进度条；中断ComfyUI任务时，
它也会请求服务端取消对应任务。成片下载到 `ComfyUI/output/h3_serve/`。

FL2VA与Ref2VA生成节点的预览都只有`关闭`/`开启`两个状态。开启后才显示`预览位置`、
`预览分辨率`与`LoRA预览步数`三个参数。正式轨迹运行到预览位置后会保存断点并停止，
不会继续计算剩余正式步；此时任务释放GPU执行权，快速预览显示在生成节点下方和
`预览视频`输出端口。节点随后显示两个按钮：

- `继续生成`：把同一正式轨迹重新排队，从保留的断点继续到最终视频；
- `放弃生成`：删除服务端任务、预览和断点，并将节点复位为可提交新任务的状态。

关闭预览时不创建断点，节点直接运行到最终视频。原来的独立“断点任务/查看断点预览/
恢复断点任务”节点仍保留，用于需要自行编排任务ID的高级工作流；普通工作流不再需要它们。

断点节点中的数字是正式σ采样位置。LoRA正式任务的全部指定步均为真实Turbo步，
加速档位只调整逐步逐层Attention配额，不插入预测步。`保留中间状态`决定能否恢复，`输出预览`只建立
可丢弃LoRA分支；两者互不替代。断点文件保存在H3 Serve的数据目录，不复制到ComfyUI。

## API兼容性

连接器使用：

- `GET /healthz`
- `GET /api/v1/options`
- `POST /api/v1/generations`
- `GET/DELETE /api/v1/jobs/{id}`
- `DELETE /api/v1/jobs/{id}/record`
- `GET /api/v1/jobs/{id}/preview`
- `POST /api/v1/jobs/{id}/preview/{continue|discard}`
- `POST /api/v1/jobs/{id}/resume`
- `GET /api/v1/jobs/{id}/video`

H3 Serve 的机器可读契约位于 `http://服务地址/openapi.json`。
