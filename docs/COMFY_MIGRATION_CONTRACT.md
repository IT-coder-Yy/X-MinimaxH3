# ComfyUI 脱离迁移契约

状态：设计与回归契约。目标运行入口为
`h3serve.native_engine.engine.NativeH3Engine`，服务适配器为
`h3serve.backend.NativeBackendManager`（兼容公开名称 `BackendManager`）。

本文规定的是“替换运行基座时不能丢失什么”，不是要求把 ComfyUI
重新包装一遍。最终生成热路径必须是进程内的 H3 专用引擎，不得启动
ComfyUI、提交 workflow JSON 或通过本机 HTTP 调用另一个推理服务。

## 1. 迁移边界

Web/API、鉴权、串行队列、任务持久化和静态控制台继续由 `h3serve`
拥有。它们只依赖下面这个中性接口：

```python
result = await backend.generate(
    spec, job_id, first_frame, last_frame, cancel_event
)
```

返回值固定包含：

```text
runtime_key: str
elapsed_seconds: float
output_path: pathlib.Path
```

`JobService` 不得了解权重加载、采样器、节点、workflow、后端端口或
模型驻留细节。`NativeBackendManager` 负责把 service spec、图片路径和 job id
规范化成窄的 engine plan，并拥有同一张 GPU 上的热模型生命周期、模式切换和
安全输出路径；`NativeH3Engine` 只接收生成所需的 plan/conditioning、显式输出
目标和取消 token，不应知道 API job id、队列位置或任务持久化格式。

最终发布热路径明确禁止：

- `import comfy` 或从 ComfyUI 目录动态导入；
- `h3serve.workflows`、`class_type` 节点图和数字 node id；
- `/prompt`、`/history`、`/upload/image`、`/system_stats` 等 ComfyUI HTTP 协议；
- 启动或管理 ComfyUI 子进程；
- 从原研究仓库、历史优化目录或用户的 ComfyUI 安装目录猜测代码和权重。

## 2. 必须保持的产品输入契约

### 2.1 四种 FL2VA 任务

| `condition_mode` | 输入 | 必须保持的语义 |
|---|---|---|
| `text` | prompt | T2VA；无视觉条件行 |
| `first` | prompt + 首帧 | 首帧锚定到输出像素帧 0 |
| `last` | prompt + 尾帧 | 尾帧锚定到输出最后一帧 |
| `first_last` | prompt + 首尾帧 | 两个锚点都进入 Qwen 表示和 DiT 条件行，顺序为首帧、尾帧 |

首帧预处理保持“直接缩放到目标画布”；尾帧保持“等比 cover 后居中裁剪到
目标画布”。二者都只使用上传图片的第一张，转为 RGB、像素范围 `[0, 1]`，
进入视频 VAE 前变换为 `[-1, 1]`。不要为了代码统一而把两种 resize 变成
同一种，否则会改变已经验证的首尾帧行为。

视觉条件经视频 VAE 编码后每个采样步重新注入，不能进入目标 latent 的去噪
更新。条件噪声增广保持 `visual_cond_noise_aug=0.999`；每个视觉条件使用同一
请求 seed 重新开始 RNG 流。只有首帧和尾帧锚点受支持，其他索引必须明确报错。

### 2.2 画布和时间网格

- FPS 固定为 24。
- 支持短边档位 360、480、720，以及 1:1、4:3、3:4、16:9、9:16。
- 宽高分别四舍五入到最近的 32 倍数，最小为 32；公开返回真实宽高。
- 因此 480p 16:9 是 `864x480`；720p 16:9 是 `1280x736`，不是
  `1280x720`。
- 请求时长范围当前为 1--15 秒。目标帧数对齐到最近的 `17k+5` 网格并限制
  到最多 362 帧；公开同时返回请求时长和 `frames / 24` 的实际时长。
- 视频 latent 时间长度为：帧数不超过 5 时为 2；否则
  `((frames - 5) // 17) * 5 + 2`。
- 音频 latent 是 `[1, 32, 2, round(actual_duration * 40)]`，40 Hz。
- Batch size 第一版固定为 1。

这些规则以 `GenerationSpec` 的序列化值为唯一服务契约。引擎不得再次根据
原始秒数或分辨率标签作另一轮独立取整。

### 2.3 请求确定性

相同的权重校验和、引擎档位、prompt、已规范化图片、真实几何、帧数、seed、
采样配置和 kernel 精度策略构成一次可复现请求。seed 是无符号 64 位整数。
随机 seed 必须在接收请求时生成并持久化，不能在 worker 开始时重新生成。

## 3. 必须重写或移植的算法组件

下列内容不是 ComfyUI 编排，删除节点系统后仍必须存在。

### 3.1 权重加载与格式适配

第一版精确支持 manifest 中的五类 artifact：

1. FL2VA Pruned INT8 ConvRot DiT；
2. Qwen3-VL-32B NVFP4/AWQ 文本编码器；
3. FP16 视频 VAE；
4. FP32 音频 VAE；
5. 极速路线可选的 Turbo LoRA。

加载器必须校验 role、路径、文件大小、SHA-256、关键 tensor、shape、dtype 和
量化 metadata，发现不支持的格式要在生成前明确失败，不得静默回退到错误的
反量化或不同数学路径。模型文件只从配置的 model root 解析；所有解析结果必须
保持在该 root 内。

### 3.2 文本与图像表示

必须保持 MiniMax H3 的 Qwen3-VL 表示，而不是普通 Qwen 文本输出：

- 取 H3 所需的隐藏层表示与 5120 维上下文；
- 纯文本至少产生 pad token；
- 图像以 `<Picture i>:`、vision start/end 和视觉 embedding 的既定顺序加入；
- 保存 text token modality tag：普通文本为 1，视觉 block（含两侧 token）为 0；
- 文本经 `condition_proj` 和两层 token refiner 后进入 5376 维 packed stream。

这部分可以复用/改造 Apache-2.0 框架的 H3 tokenizer 和加载代码，但输出张量、
token tag 和层选择必须用当前 Comfy 正确流程作为 oracle 做对照。

### 3.3 H3 packed DiT

数学实现至少包括：

- 视频 latent `[B,24,T,H,W]` 以 `1x2x2` patchify；音频 latent
  `[B,32,2,T]` 按 stereo channel-major pack；
- packed 顺序为 `text | keyframe cond | target audio | target video`；
- 三轴 area-normalized position grid、时间跨度 `(1,4,4,4,4) * 5/3`；
- fused QKV、每 head RMSNorm、96 维 partial split-half RoPE；
- 50 个 DiT block、两层 token refiner、SwiGLU MLP、分段 AdaLN 与 gate；
- 视频 shift 12、音频 shift 3 的闭式 sigma 映射和音频速度 slope 修正；
- 视频、音频输出头保持 FP32 island，之后恢复各自 latent dtype；
- 输入无法整除 patch size 时先 pad，只裁剪目标视频输出。

推荐以 SGLang Diffusion 的 fused H3 图作为 Apache-2.0 主体来源，以 LightX2V
的单卡分层 offload/预取为显存策略，以 FastVideo 的 stage 和测试组织为工程
参考。不要把三个完整框架同时变成运行依赖。

### 3.4 Pruned AdaLN curve

当前 pruned checkpoint 没有完整 time embedder，而是保存共享
`adaln_t_table[grid,k]`。每次 forward：

1. 从视频 sigma 得到 `t_video`，由 shift 映射得到 `t_audio`；
2. 条件行还可能产生 `max(t, 0.999)`；
3. 对不同时间值排序去重；
4. 将 `t` clamp 到 `[0,1]`，映射到 grid；
5. 在相邻两行间线性插值；`t=1` 使用最后一个区间而不越界；
6. curve 模式的 AdaLN projection 不再额外 SiLU，并保持 FP32 权重/插值精度。

这是正确性硬要求，不能把 `adaln_t_table` 当作普通 embedding lookup。

Turbo LoRA 的 AdaLN 更新位于原 2688 维 SiLU time-embedding 空间。pruned base
需要用发布包内固定 E-grid 恢复该低秩更新后再加到每个 AdaLN projection；
不能直接合并到 8 维 curve 权重。E-grid 本身必须进入 artifact manifest 和校验。

### 3.5 ConvRot INT8 与高性能算子

当前 DiT linear metadata 是 tensor-wise INT8，`convrot=true`，默认 group size
256。运行时必须在激活量化前施加同一正交旋转，并保持权重 scale、转置约定、
bias 与输出 dtype。错误地把权重当普通 INT8 虽然可能“跑通”，但会产生彩色色斑、
形变或整体失真，必须视为硬失败。

可直接复用 Apache-2.0 的 Comfy Kitchen kernel（保留许可证与 NOTICE），也可
保留已经验证的 SM89 NVRTC/Triton kernel。所有融合都必须先有逐算子 shape/dtype
对照，再通过真实视频视觉门控。4090 上保留 SageAttention 后端探测，但提供
标准 SDPA 正确性 fallback；fallback 不得改变 Q/K/V layout 和缩放。

### 3.6 两条生成路线

高保真路线：

- 20 个 `simple` sigma 调度点与 `res_multistep` 采样行为；
- `ultra` 为 20 actual / 0 forecast 的引擎正确性基线；
- 其他档位使用显式 actual step index，不得只按数量均匀生成；
- 当前默认 9/11 schedule 是 `[0,1,2,3,4,8,12,16,19]`；
- quality 12/8 是 `[0,1,2,3,4,6,8,11,14,17,18,19]`；
- fast 8/12 是 `[0,1,2,3,4,8,13,19]`。

forecast 不是“空跑”：需要保留 depth-3 anchor、两个 actual tail、音频全局与视频
局部方向置信度、gamma `[1,1.35]`、不一致区域回落和 pinned-host history。原先
Spectrum 节点只提供事务/patch 编排，最终引擎应在 DiT block loop 内直接调用
我们的 forecast controller。

极速路线：

- 加载同一 pruned base 和 Turbo LoRA，strength 当前固定 1.0；
- 对外档位为 4/5/6/8 步，六步是稳定质量基线；
- 使用 Turbo sampler 的音视频双时钟语义；
- LoRA 默认在 activation space bypass，避免低秩 delta 被 INT8/BF16 合并舍入；
- INT8 fused `mlp.fc2` 不经过 module forward，必须使用已验证的特殊合并/权重
  路径，否则会静默丢失 50 个 fc2 LoRA 更新；
- low-VRAM merge 可以作为显式高级选项，但不能冒充同质量默认路径。

### 3.7 显存与组件生命周期

单张 4090 24GB、batch 1 是硬约束。建议生命周期：

```text
text encoder -> 释放/卸载 -> DiT 热采样 -> 释放 forecast history
-> video VAE tiled decode -> audio VAE decode -> mux
```

模型级和 block 级 offload、双 CUDA buffer、copy/compute stream、下一 block 异步
预取借鉴 LightX2V，但调度器必须显式拥有驻留状态。不要移植 ComfyUI 的通用
ModelPatcher、节点 cache、任意设备图或 workflow cache。

### 3.8 视频、音频解码与 MP4

- 视频 VAE：24 channel normalized latent，时空解码与内部 spatial/temporal tiling
  保持当前 checkpoint 语义；最终输出 `[frames,H,W,3]`、float `[0,1]`。
- 音频 VAE：FP32，32 kHz，stereo，800 samples/latent；输出 clamp 到 `[-1,1]`。
- 保持当前音频后处理：对 batch 的 channel/time 标准差乘 5，低于 1 的 divisor
  设为 1，再做除法。这是现有可听响度的一部分，不能漏掉。
- 输出 MP4 为 24 FPS、8-bit H.264-compatible video、stereo 32 kHz audio；音视频
  时间戳必须从同一实际时长产生。编码器选择和 codec 参数写入运行 metadata。
- MP4 必须先写入同一 output root 下的临时文件，完成 fsync/close 和媒体探测后
  原子重命名；不得把任意后端返回路径直接暴露给 API。

媒体探测至少确认：容器可读、视频帧数/宽高/FPS、存在双声道音轨、采样率和
时长在约定容差内。没有音轨或视频流时生成任务失败。

## 4. 可以删除的 ComfyUI 编排

以下内容没有算法价值，原生基座完成后直接删除：

- `workflows.py` 和所有节点 id/class type；
- ComfyUI `/prompt`、history 轮询、upload/image 和 output record 解析；
- 两个 ComfyUI server、SQLite prompt DB、端口健康检查；
- `folder_paths`、custom node 注册、workflow queue 和 generic node cache；
- import hook、`runpy(COMFY/main.py)` 和对 Comfy 模块的 monkey patch；
- `H3_SERVE_*_COMFY_DIR`、`--comfy-dir`、额外模型路径 YAML；
- 通用训练、任意模型、任意设备、任意 batch 和多 GPU 兼容层。

保留的是这些编排背后的数学与产品行为，而不是它们的节点表现形式。

## 5. 服务状态与错误契约

状态机保持：

```text
queued -> starting_backend -> running -> succeeded
                                  |-> failed
                                  |-> cancelled
```

`starting_backend` 在原生实现中表示“确保所需模型模式已加载/驻留”，不是启动
HTTP server。服务重启时未完成任务标为 failed；完成任务只有在输出文件实际
存在且媒体探测通过后才持久化 succeeded。

客户端错误保持简短稳定：`generation failed (reference XXXXXXXX)`；完整 traceback
只写服务器日志。内部错误要使用可分类 exception/code（至少 model artifact、
unsupported request、OOM、kernel、cancel、encode、decode、mux、timeout），方便
API 和日志定位，但不得把本机绝对路径或 traceback 暴露给远端用户。

取消必须在 text encode、每个 actual/forecast step、VAE tile 和 mux 前检查。
取消后清理临时输出和只属于该请求的 pinned history；模型热驻留状态可以保留。

## 6. 不依赖 GPU 的回归门禁

`tests/native_engine/` 提供三类 CPU-only 检查：

1. 服务边界：四种 conditioning mode 原样穿过单一 `generate()`；成功、取消、
   失败遮蔽和持久化状态正确；`JobService` 不再拼 workflow。
2. 纯契约：几何、时间网格、latent shape、seed、两个路线的 preset 和 artifact
   manifest 可序列化且稳定。
3. 原生发布静态门：`backend.py` 与 `native_engine/` 不导入 Comfy、workflow 或
   HTTP client，不出现 Comfy endpoint/子进程；旧 `workflows.py` 已删除。

普通开发测试：

```bash
python -m unittest discover -s tests -v
```

最终原生发布门：

```bash
H3_NATIVE_RELEASE_GATE=1 python -m unittest discover \
  -s tests/native_engine -t . -v
```

严格门在 `NativeH3Engine` 尚未落地时会失败；普通测试不会假装 GPU 可用，也
不会加载 42GB 权重。解除迁移阻断的精准条件见第 8 节。

## 7. 许可证与代码来源边界

- 当前 ComfyUI 源码和 Spectrum runtime 是 GPL-3.0。若项目所有者不选择
  GPL-compatible 的发布许可证，不应逐文件复制或修改后嵌入它们；应把它们
  作为行为 oracle，基于公开模型结构作独立实现。此处不是法律意见，公开发布前
  仍需正式许可证审阅。
- SGLang Diffusion、LightX2V、FastVideo、Comfy Kitchen 和现有 Turbo node
  许可副本为 Apache-2.0，可在保留版权、LICENSE/NOTICE 和修改说明的前提下
  选择性移植。
- 不要从多个项目复制同一功能。每个移植文件在文件头和
  `THIRD_PARTY_NOTICES.md` 记录上游仓库、commit、原路径和本项目修改。
- 模型权重不随源码默认分发，继续使用 manifest 固定来源、revision、size 和
  SHA-256，并要求用户明确接受权重许可证。

## 8. 阻断项与上线门槛

当前阻断项：

1. `h3serve/backend.py` 仍是兼容实现，含 aiohttp、workflow 和 Comfy 子进程；
2. `h3serve/native_engine/engine.py` 尚未完成可运行 `NativeH3Engine`；
3. ConvRot/pruned AdaLN/Qwen NVFP4 的独立加载链尚未完成逐算子对照；
4. 高保真 forecast 仍借 Spectrum runtime 事务，需直接并入 block loop；
5. Turbo pruned-AdaLN E-grid 尚未进入统一 artifact manifest；
6. 独立 VAE/mux 与媒体探测尚未实现；
7. 项目自身发布许可证尚未决定。

上线必须同时满足：

- 严格 CPU-only 原生发布门全部通过且无 skip/expected failure；
- 干净环境安装后进程模块与发行包中都没有 ComfyUI/Spectrum runtime；
- 同 seed/config 的 20/0 native baseline 与现有正确流程完成逐算子、latent 和
  媒体结构对照；
- 两条路线覆盖 T2VA、首帧、尾帧、首尾帧；
- 480p/720p × 5s/15s，常见比例均在 24GB 内完成；
- 每个真实 MP4 **先**通过多帧视觉门控：无彩色色斑、严重散光重影、异物闪现、
  明显形体崩坏或远景烟雾扭曲；不通过则不进入数值评价；
- 音轨存在、可解码、双声道 32 kHz，台词/环境声由 Human 验收；
- 通过视觉门控后才记录热/冷时间、峰值 VRAM、媒体信息和相似度诊断；
- SSIM 只作同 seed 数值差异诊断，**不得**作为质量硬门或替代 Human 判断；
- 取消、OOM、坏权重、坏图片、磁盘满、mux 失败和服务重启都有稳定回归；
- LICENSE、第三方 NOTICE、权重许可与 provenance 审阅完成。

这套 native 20/0 路径跑通后才成为新的内部 baseline。V7 高保真调度、ConvRot
融合、SageAttention、VAE tiling、V8 Turbo LoRA 和既有 profile 方法都以单项
消融的方式逐步接回，避免在迁移阶段把错误与优化混在一起。
