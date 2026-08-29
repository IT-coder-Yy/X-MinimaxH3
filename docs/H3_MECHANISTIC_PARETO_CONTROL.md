# H3 机制驱动的帕累托推理控制系统

## 1. 对外契约

创作者只设置两个量：

- 总去噪步数 `N`；
- 加速力度 `α ∈ [0, 100]`。

`α` 不是版本号、候选配置编号或某张历史步表的插值系数。它表示：在当前工作负载经过 Human 证据准入的计算节省区间内，用户希望取得多大比例的计算节省。系统把它编译成一次完整的 Actual/Forecast 轨迹、每个 Actual 步的 50 层 Attention 动作，以及有限的运行时纠偏预算。

## 2. 被联合求解的物理决策

令 `x_k` 表示第 `k` 个 sigma 时刻的音视频联合 latent。一次计划同时决定：

- `z_k ∈ {Actual, Forecast}`：是否执行完整的 50-block DiT；
- `a_{k,l}`：Actual 步中第 `l` 层使用 Dense 或某个已测量的稀疏 Attention 实现；
- `u_k`：运行时是否把原计划的 Forecast 提升为 Actual 纠偏。

这些量不是三个相互独立的开关。Forecast 会改变后续 Actual 所看到的状态误差，Attention 近似会在这个带误差的状态上继续作用，因此风险模型包含一阶交互项。优化器在同一个状态转移中联合选择 `z_k` 与全部 `a_{k,l}`；运行时 `u_k` 只能消费离线计划预留的真实毫秒预算。

## 3. 机制风险模型

### 3.1 局部误差

Dense Attention 的局部近似误差定义为 0。每个稀疏 Attention 动作的局部误差来自同一 H3 层、同一归一化 sigma 位置上相对 Dense 的测量误差。Forecast 的局部误差来自 H3 directional secant-tail 在相位、连续预测长度和 token 规模上的回归：

```text
log e_forecast = β₀ + β₁p + β₂p² + β₃log(h)
                 + β₄p·log(h) + β₅log(T/T₀)
```

其中 `p` 是归一化去噪相位，`h` 是连续 Forecast horizon，`T` 是精确 packed-token 数。提示词语义、场景类别、seed 和历史候选版本均不进入方程。

### 3.2 下游传播

一次局部扰动对最终 latent 的影响用 Grönwall 形式描述：

```text
G_m(p) = exp(∫[p,1] L_m(s) ds)
```

`m` 分别表示音频与视频模态。当前系数由单步反事实 impulse 实验识别，并用低阶连续相位曲线拟合 `log G`。这解释了为什么早期近似通常比尾部近似危险，而不是把“前几步必须 Actual”写成经验规则。

### 3.3 连续 Forecast 与交互

同一 Forecast run 共享方向历史，不能假设为独立噪声。系统先累加由 sigma 步宽加权的误差幅度，再对整段幅度计入能量。Forecast 后的 Actual 若仍使用近似 Attention，会额外产生状态误差与局部算子误差的一阶交互风险。

最终保守风险为：

```text
R = R_attention + R_forecast_audio + R_forecast_video + R_interaction
```

每项使用测量残差和工作负载外推距离形成的上界。`R` 是“数值扰动风险代理”，不是未经验证就等同于 Human 主观质量的分数。

## 4. 计算模型与帕累托求解

每个物理动作的代价来自同一 RTX 4090 实现的热态 wall-time 测量。离线控制问题为：

```text
min_π  R(π) + λ C(π)
```

其中 `π` 是完整联合计划，`C` 是预测 wall cost，`λ ≥ 0` 是计算的影子价格。由于未来代价只依赖当前 Forecast run、可用历史深度和去噪相位，当前实现用动态规划精确求解所声明的有限 Lagrangian 问题；逐层 Attention 决策在同一个 Bellman 转移里求解。

对任意 `λ > 0`，全局最小解都是所声明 `(C,R)` 问题的 supported Pareto 点：不存在另一个计划同时不更慢且风险不更高，并至少严格改善一项。证书记录模型、工作负载和最终物理选择的 digest，可由 verifier 重放。

这项“精确”只针对已声明的计算—风险模型。真实视频质量是否服从该代理，仍必须用未参与识别的 Human holdout 验证。

## 5. 加速力度的连续语义

给定 Human 准入风险上界 `ρ`：

1. 求 Dense 最低风险端点 `π_dense`；
2. 求满足 `R(π) ≤ ρ` 的最快 supported 端点 `π_fast`；
3. 将公开力度映射为连续计算目标：

```text
C_target(α) = C_dense - α/100 · (C_dense - C_fast)
```

随后由同一个联合优化器寻找达到该计算目标的 supported Pareto 计划。力度 100 的含义是“当前 Human 证据允许的最快点”，不是越过证据边界的无限外推。

## 6. 运行时轻量纠偏

现有 Actual 修正本来就会得到 secant-tail 的误差样本，不增加 teacher DiT 评估。控制器维护音频和视频误差比例的 request-local 单调上包络，并在剩余 Forecast 中精确枚举最多两次提升：

```text
min  extra_cost
s.t. projected_R ≤ ρ
     extra_cost ≤ reserved_ms
```

若预算不足以恢复准入边界，系统显式报告 `admitted=false`，而不是伪造安全结论。运行时只允许 Forecast → Actual，不允许把离线 Actual 临时降级，所以异常输入会向质量安全方向退让。

纠偏预算从 `C_target` 内预先扣除；报告同时列出离线计划成本、纠偏 reserve 和二者之和，禁止把纠偏宣传成免费计算。

## 7. 历史版本和 Human 反馈如何使用

历史视频允许提供以下证据：

- 哪类局部数值误差会传播成嘴型模糊、背景抖动、物体结构突变或异常语音；
- 哪些相位、模态和层对这些失效更敏感；
- 数值风险代理在什么上界内可被 Human 接受；
- 哪些机制假设被反例推翻，需要重新做 impulse、消融或交互建模。

历史视频不允许直接提供：

- 某版本的 Actual 步表；
- 某版本的逐层动作表；
- 按分辨率、提示词类别或版本号选择配方的分支；
- 在若干“好样本点”之间插值出的调度表。

部署 admission JSON 只允许保存风险边界、证据 ID、准入 token 区间和最多纠偏次数。加载器拒绝任何额外字段，因此不能把历史 schedule 偷渡进发布入口。

## 8. 当前证据边界

截至当前实现：

- 已有 720p×5s、20-step 的 Forecast/Attention 单 impulse 传播识别；
- 已有 short/medium/long packed-token 的 Forecast 局部误差样本；
- 已有 RTX 4090 上逐层 Attention 误差与动作成本；
- 已完成统一入口、计划证书、运行时 risk-reserve 和失败闭合逻辑；
- M001 与 V007 的 720p×5s 同速度实片已生成，静态灾难性画面检查通过，尚待 Human 连续运动、嘴型和音频审阅。

尚不能声称发布准入已经完成，原因是：

- impulse 传播曲线仍主要来自一个 720p×5s seed/prompt；
- 720p×15s、1080p×15s、参考图片/音频布局还缺独立传播识别和 holdout；
- Human 风险上界 `ρ` 尚未形成带 held-out 证据的 release admission；
- 预测 wall cost 仍需在长视频与不同上下文长度上校准误差带。

在这些证据补齐前，新 selector 只能通过显式 admission 文件 opt-in；默认发布策略不会被悄悄替换。

## 9. 发布判据

一个 workload 区间只有同时满足下列条件，才能从 `experimental` 升为 `release`：

1. 机制参数由 calibration 集识别，Human 阈值由独立 review 标定；
2. 未参与识别的 prompt、seed、时长和可支持条件输入构成 holdout；
3. 同速度 A/B 不劣于当前质量锚点，或同质量显著更快；
4. 嘴型、语音、物理逻辑、背景稳定性和结构一致性分别过门；
5. 预测成本、真实 denoise 时间和端到端时间均报告；
6. 失败输入能够 Dense 回退或显式暴露风险，而不是静默越界。

