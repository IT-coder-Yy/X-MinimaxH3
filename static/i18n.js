(() => {
  'use strict';

  const STORAGE_KEY = 'h3serve_locale';
  const SUPPORTED = new Set(['zh-CN', 'en']);
  const exact = new Map(Object.entries({
    '工作区': 'Workspace',
    '界面语言': 'Interface language',
    '创作台': 'Create',
    '任务管理': 'Tasks',
    '连接服务中': 'Connecting',
    '切换模型': 'Switch model',
    'API设置': 'API settings',
    '选择生成模式': 'Choose a generation mode',
    '同一时间只加载一个 H3 引擎。进入模式后保持热态；退出或切换会完整释放当前引擎。': 'Only one H3 engine is loaded at a time. It stays warm until you exit or switch models.',
    '当前工作空间': 'Current workspace',
    '默认工作空间': 'Default workspace',
    '工作空间绝对路径': 'Absolute workspace path',
    '读取中…': 'Loading…',
    '选择文件夹': 'Choose folder',
    '准备模型引擎': 'Preparing model engine',
    '等待开始': 'Waiting to start',
    '创作记录': 'Creation history',
    '已发送的任务、生成进度和最终成片都会保留在这里。': 'Submitted jobs, generation progress, and finished videos stay here.',
    '新建视频': 'New video',
    '编辑分镜、增强提示词并发送到当前引擎': 'Edit shots, refine the prompt, and send it to the active engine',
    '发送生成任务': 'Submit generation job',
    '当前固定服务': 'Active fixed service',
    'H3 Native 高保真': 'H3 Native Fidelity',
    'H3 Native 多参考': 'H3 Native Multi-reference',
    'H3 Native 多参考 Turbo': 'H3 Native Multi-reference Turbo',
    '原始权重 · 能力、台词与画面稳定优先': 'Base weights · prioritizes capability, dialogue, and visual stability',
    '高保真': 'Fidelity',
    '极速': 'Fast',
    '最快，适合预览；复杂运动建议先验片': 'Fastest; suited to previews. Test complex motion first.',
    '均衡': 'Balanced',
    '推荐默认档，兼顾稳定性与等待时间': 'Recommended default balancing stability and wait time.',
    '高质量': 'High quality',
    '增加完整计算，适合复杂运动和远景': 'Adds full computation for complex motion and distant detail.',
    '超高质量': 'Ultra quality',
    '全部调度点执行完整计算，速度最慢': 'Runs full computation at every scheduled point; slowest.',
    '四步快速预览': 'Fast four-step preview',
    '五步日常生成': 'Five-step everyday generation',
    '六步稳定基线': 'Stable six-step baseline',
    '八步实验档；更多步不保证单调提升': 'Experimental eight-step mode; more steps may not improve quality monotonically.',
    'FL2VA · 文本/首尾帧生成': 'FL2VA · Text / first-last frame generation',
    '支持文生视频、首帧、尾帧和首尾帧约束': 'Supports text-to-video and first-frame, last-frame, or first/last-frame constraints.',
    'Ref2VA · 多模态参考生成': 'Ref2VA · Multimodal reference generation',
    '支持图片、视频和音频参考': 'Supports image, video, and audio references.',
    'INT8高速后端；首代最高1080p，二采最高1440P': 'INT8 high-speed backend; native generation up to 1080p and second sampling up to 1440P.',
    'INT8多参考高速后端；首代最高1080p，二采最高1440P': 'INT8 multi-reference high-speed backend; native generation up to 1080p and second sampling up to 1440P.',
    'INT8紧凑高速后端；实验性首代最高1080p×15秒，二采最高1440P': 'INT8 compact high-speed backend; experimental native generation up to 1080p × 15s and second sampling up to 1440P.',
    'INT8多参考紧凑后端；实验性首代最高1080p×15秒，二采最高1440P': 'INT8 compact multi-reference backend; experimental native generation up to 1080p × 15s and second sampling up to 1440P.',
    '8GB低比特权重；首代最高720p×15秒，二采最高1080p': '8GB low-bit weights; native generation up to 720p × 15s and second sampling up to 1080p.',
    '8GB低比特参考后端；720p×15秒限单参考，二采最高1080p': '8GB low-bit reference backend; 720p × 15s allows one reference and second sampling reaches 1080p.',
    '0=Dense计算；75=Human审阅质量拐点；75–100=明确允许质量风险的激进区': '0 = Dense; 75 = Human-reviewed quality knee; 75–100 = aggressive range with explicit quality risk.',
    '128GB 火力全开': '128GB Full power',
    'Qwen与H3保持热态；生成与原生H3二次采样共享同一热引擎。': 'Keep Qwen and H3 warm; generation and native H3 second sampling share the same hot engine.',
    '96GB 生成优先': '96GB Generation priority',
    '64GB 高效兼容': '64GB Efficient compatibility',
    'Qwen按执行层流水读取；H3按需驻留并支持原生二次采样。': 'Stream Qwen by execution layer; keep H3 resident on demand with native second sampling support.',
    '＞64GB 高速模式': '>64GB high-speed mode',
    '≤64GB 兼容模式': '≤64GB compatibility mode',
    'Qwen与H3保持热态，优先缩短生成和二次采样的切换等待。': 'Keeps Qwen and H3 warm to minimize switching delays between generation and second sampling.',
    '按执行阶段控制CPU权重驻留，在较小主机内存下保持完整生成能力。': 'Controls CPU weight residency by execution stage while retaining full generation capability on lower-memory hosts.',
    '64GB以上': 'Above 64GB',
    '64GB及以下': '64GB or below',
    '模式只改变CPU权重驻留和阶段切换，不改变模型权重、采样步数或画质。': 'This mode changes only CPU weight residency and stage transitions; model weights, sampling steps, and visual quality are unchanged.',
    '决定完整 σ 去噪轨迹长度；系统在这条轨迹内联合安排真实步和预测步。': 'Sets the length of the complete σ denoising trajectory; the system jointly schedules actual and forecast steps along it.',
    '超过8步未经LoRA质量校准；允许运行，但不保证质量随步数单调增加。': 'More than 8 steps has not been quality-calibrated for LoRA. It is allowed, but quality may not improve monotonically.',
    'LoRA 使用完整 Turbo 步；内部加速只分配逐步逐层注意力，不擅自加入预测步。': 'LoRA uses complete Turbo steps; internal acceleration allocates attention per step and layer without inserting forecast steps.',
    '当前服务未安装 SM89 稀疏运行时，只能使用 0（Dense）；请运行项目安装脚本。': 'The SM89 sparse runtime is not installed; only 0 (Dense) is available. Run the project installer.',
    'LoRA 无预测调度：全部 Turbo 步保持真实计算，档位只改变逐步逐层 Attention 配额。': 'LoRA has no forecast scheduling: every Turbo step remains an actual computation, while the level changes only per-step, per-layer attention budgets.',
    'V19认证前沿：仅命中已封存工作负载时加速；其他输入自动Dense回退。': 'V19 certified frontier: acceleration is used only for sealed workloads; other inputs automatically fall back to Dense.',
    '冻结的 Round229 调度：构图锚点、因果层、预测后恢复步和末端细节保护始终开启。': 'Frozen Round229 schedule: composition anchors, causal layers, post-forecast recovery, and terminal detail protection remain enabled.',
    '任务已取消': 'Job cancelled',
    '或': 'or',
    '尚未加载模型': 'No model loaded',
    '关键帧': 'Keyframes',
    '可选': 'Optional',
    '首帧预览': 'First-frame preview',
    '尾帧预览': 'Last-frame preview',
    '添加首帧': 'Add first frame',
    '添加尾帧': 'Add last frame',
    '点击选择或直接拖入图片': 'Click or drop an image here',
    '参考素材': 'References',
    '至少添加一项': 'Add at least one',
    '＋ 添加文件': '+ Add files',
    '图片≤9 · 视频≤3（2–15秒，音轨不作为参考） · 音频≤3': 'Images ≤9 · Videos ≤3 (2–15s; audio tracks ignored) · Audio ≤3',
    '自动等比降分辨率：图片720P · 视频360P': 'Automatic proportional downsampling: images 720P · videos 360P',
    '提示词': 'Prompt',
    '提示词输入方式': 'Prompt input mode',
    '模块化': 'Structured',
    '自由输入': 'Freeform',
    '官方结构与分镜编辑': 'Official structure and shot editor',
    '目标时长': 'Target duration',
    '参考对象': 'Reference subjects',
    '建立素材对象与保留规则': 'Define referenced subjects and retention rules',
    '撤销': 'Undo',
    '润色参考对象': 'Refine references',
    '对象定义 · subject_definitions': 'Subject definitions · subject_definitions',
    '保留规则 · retention_analysis': 'Retention rules · retention_analysis',
    '画面内容': 'Visual content',
    '总体摘要、画面连续性与分镜': 'Summary, visual continuity, and shots',
    '润色画面内容': 'Refine visuals',
    '总体摘要 · summary': 'Overall summary · summary',
    '分镜列表': 'Shot list',
    '＋ 添加分镜': '+ Add shot',
    '声音设计': 'Sound design',
    '画面内声音与画外配乐': 'Diegetic sound and non-diegetic music',
    '润色声音设计': 'Refine sound',
    '画面内声音 · overall_soundscape': 'Diegetic sound · overall_soundscape',
    '画外背景音乐 · non_diegetic_music': 'Background music · non_diegetic_music',
    '关闭 · N/A': 'Off · N/A',
    '背景音乐': 'Background music',
    '配乐描述或 @ 引用音频素材': 'Music description or @ audio reference',
    '查看最终发送给 H3 的提示词': 'View the final prompt sent to H3',
    '撰写建议': 'Writing guide',
    '1. 参考关系': '1. Reference relationships',
    '2. 画面内容': '2. Visual content',
    '3. 声音设计': '3. Sound design',
    '仅多参考模式需要：说明 <Subject N> 来自哪个 <Picture N>、<Video N> 或 <Audio N>，以及需要保留什么。': 'For multi-reference mode only: state which <Picture N>, <Video N>, or <Audio N> each <Subject N> comes from and what to preserve.',
    '按播放顺序写主体、环境、动作、台词、运镜和分镜时间。': 'Describe subjects, setting, actions, dialogue, camera movement, and shot timing in playback order.',
    '场景内声音；': 'diegetic sound; ',
    'N/A 或具体配乐描述。': 'N/A or a specific music description.',
    '输入要直接发送给 H3 的完整提示词……': 'Enter the complete prompt to send directly to H3…',
    '写什么：描述这个连续镜头中实际发生的画面、动作、台词、运镜和同步声音。\n怎么写：按发生顺序写清主体、动作、环境、镜头运动和现场声音；台词保留原语言。\n示例：镜头从桌上的书包特写缓慢拉远，女孩走近并拉开拉链，清晰听见脚步声和拉链声。': 'What to write: Describe the visuals, actions, dialogue, camera movement, and synchronized sound in this continuous shot.\nHow to write: Follow the event order and clearly specify subjects, actions, environment, camera movement, and diegetic sound. Keep dialogue in its original language.\nExample: The camera slowly pulls back from a close-up of a backpack on the table. A girl approaches and opens the zipper as footsteps and the zipper are heard clearly.',
    '写什么：描述这个连续镜头中实际发生的画面、动作、台词、运镜和同步声音。\n怎么写：建议使用英文；按发生顺序写，人物与素材使用 @ 引用，台词保留原语言并写成 <d>[Language] ...</d>。\n示例：A medium shot follows <Subject 1> walking toward the bench. She stops, looks up, and says: <d>[Chinese] 你终于来了。</d>': 'What to write: Describe the visuals, actions, dialogue, camera movement, and synchronized sound in this continuous shot.\nHow to write: English is recommended. Follow event order, use @ references for people and media, and keep dialogue in its original language using <d>[Language] ...</d>.\nExample: A medium shot follows <Subject 1> walking toward the bench. She stops, looks up, and says: <d>[Chinese] 你终于来了。</d>',
    '生成设置': 'Generation settings',
    '推理设置': 'Inference',
    '模型、步数与统一显存加速后端': 'Model, steps, and the unified VRAM-optimized backend',
    '原始权重 · 20步 · Dense': 'Base weights · 20 steps · Dense',
    '推理路线': 'Model route',
    'INT8 原始权重': 'INT8 base weights',
    'LoRA 极速': 'LoRA Turbo',
    '总采样步数': 'Sampling steps',
    '完整去噪轨迹': 'Complete denoising trajectory',
    '加速档位': 'Acceleration',
    '0 为 Dense；数值越大，内部加速越强。': '0 is Dense; higher values apply stronger internal acceleration.',
    '断点任务': 'Checkpoint job',
    '启用断点任务': 'Enable checkpoint job',
    '断点位置': 'Checkpoint position',
    '预览后可从原任务继续，不会重复断点前的正式计算。': 'Resume the original job after preview without repeating completed formal computation.',
    '画面与输出': 'Video and output',
    '尺寸、比例与时长': 'Size, aspect ratio, and duration',
    '尺寸模式': 'Size mode',
    '常用分辨率': 'Resolution preset',
    '自定义尺寸': 'Custom size',
    '分辨率': 'Resolution',
    '1440P（实验）': '1440P (Experimental)',
    '画面比例': 'Aspect ratio',
    '视频时长': 'Video duration',
    '自由输入模式单独设置': 'Set independently in freeform mode',
    '宽度': 'Width',
    '高度': 'Height',
    '拖动等待中的任务调整顺序；正在执行的任务只能取消，不能重排。': 'Drag queued jobs to reorder them. Running jobs can be cancelled but not reordered.',
    '刷新': 'Refresh',
    '主机实时资源': 'Live host resources',
    '读取中': 'Loading',
    '系统内存': 'System memory',
    'GPU 显存': 'GPU VRAM',
    '正在运行': 'Running',
    '等待队列': 'Queue',
    '已完成': 'Completed',
    '服务热态': 'Engine state',
    '检查中': 'Checking',
    '真实阶段进度': 'Real stage progress',
    '拖动卡片改变执行顺序': 'Drag cards to change execution order',
    '历史任务': 'History',
    '可预览、下载或删除记录与输出': 'Preview, download, or delete records and outputs',
    '服务与AI设置': 'Service and AI settings',
    '服务 API Key': 'Service API key',
    '本地无鉴权时留空': 'Leave blank when local authentication is disabled',
    '用于访问设置了 H3_SERVE_API_KEY 的视频服务。': 'Used to access a video service configured with H3_SERVE_API_KEY.',
    '小米 MiMo API Key': 'Xiaomi MiMo API key',
    '填写新Key；留空则保留服务器现有Key': 'Enter a new key; leave blank to keep the server key',
    '正在检查服务器配置……': 'Checking server configuration…',
    '清除服务器 MiMo Key': 'Clear server MiMo key',
    'LoRA 权重版本': 'LoRA weight version',
    '扫描中': 'Scanning',
    '已安装的 H3 LoRA': 'Installed H3 LoRA',
    '正在扫描 models/loras…': 'Scanning models/loras…',
    '加载所选 LoRA': 'Load selected LoRA',
    '只加载 models/loras 中与当前 H3 原生适配器格式兼容的 .safetensors。切换要求任务队列为空，并会重建当前热引擎。': 'Only compatible native H3 adapter .safetensors files under models/loras can be loaded. Switching requires an empty queue and rebuilds the current hot engine.',
    '未发现 LoRA 权重': 'No LoRA weights found',
    '格式不兼容': 'incompatible format',
    '切换中': 'Switching',
    '个可用': 'available',
    '正在释放并重建当前 H3 热引擎，请勿提交任务。': 'Releasing and rebuilding the current H3 hot engine. Do not submit jobs.',
    '当前版本': 'Current version',
    '热引擎已加载': 'Loaded by hot engine',
    '选择一个兼容权重；进入模型后加载会重建热引擎。': 'Choose a compatible weight. Loading it after entering a model rebuilds the hot engine.',
    '正在重建引擎…': 'Rebuilding engine…',
    '切换要求队列为空；当前热引擎将完整释放并重新加载。': 'Switching requires an empty queue. The current hot engine will be fully released and reloaded.',
    '加载失败': 'Load failed',
    '断点预览': 'Checkpoint preview',
    '全局默认': 'Global default',
    '预览分辨率': 'Preview resolution',
    'LoRA预览步数': 'LoRA preview steps',
    '断点后用同一 H3 权重挂载 Turbo LoRA 生成临时预览；不改变正式轨迹、正式权重或继续生成的结果。创作页仅需选择断点位置。': 'At the checkpoint, mount Turbo LoRA on the same H3 weights to create a temporary preview. The formal trajectory, weights, and resumed result remain unchanged. Only choose the checkpoint position on the Create page.',
    '参考素材自动压缩': 'Reference media downsampling',
    '参考图片': 'Reference images',
    '参考视频': 'Reference videos',
    '保持原分辨率': 'Keep original resolution',
    '档位只控制像素采样分辨率：始终按原宽高比等比例缩小，原画幅尺寸、完整构图和视频时长不变；不裁切、不拉伸、不补边，也不放大小素材。该默认同时作用于工作台、API和ComfyUI。': 'This setting only controls sampled pixel resolution. Media is proportionally reduced while preserving aspect ratio, full composition, and duration—without cropping, stretching, padding, or enlarging small inputs. The default applies to the console, API, and ComfyUI.',
    '二次采样时间窗口': 'Second-sampling temporal window',
    '窗口长度': 'Window length',
    '短窗口通常更快；若出现跨窗口动作、人物或背景不连续，请调长。后端自动按 H3 时间相位对齐，并自动处理 17 帧重叠与 latent 交叉融合。': 'Shorter windows are usually faster. Increase the window if motion, subjects, or backgrounds become discontinuous. The backend snaps to the H3 temporal phase and automatically handles 17-frame overlap and latent crossfade.',
    '生成任务上限': 'Generation limits',
    '逐项设置每种分辨率和画面比例允许提交的最长时长。显存信息只供参考，最终上限由你决定。': 'Set the maximum submitted duration for each resolution and aspect ratio. VRAM information is advisory; you control the final limits.',
    '主机内存运行模式': 'Host memory mode',
    '检测中': 'Detecting',
    '模式改变的是CPU权重驻留和冷加载，不改变模型权重、采样步数或画质。切换要求任务队列为空并会重新预加载H3。': 'This mode changes CPU weight residency and cold loading, not model weights, sampling steps, or quality. Switching requires an empty queue and reloads H3.',
    '保存并应用': 'Save and apply',
    '选择工作空间': 'Choose workspace',
    '生成视频、导入素材、任务记录和断点都保存在这个文件夹。模型权重与编译缓存不会重复复制。': 'Generated videos, imported media, job records, and checkpoints are stored here. Model weights and compiled caches are not duplicated.',
    '打开': 'Open',
    '← 上一级': '← Parent',
    '项目默认': 'Project default',
    '使用这个文件夹': 'Use this folder',
    '下载视频': 'Download video',
    'H3 二次采样': 'H3 second sampling',
    '清理Latent缓存': 'Clear latent cache',
    '清理所有Latent与断点缓存？成片和任务记录会保留，但历史任务将不能再二次采样或从断点继续。': 'Clear all latent and checkpoint caches? Videos and job history will remain, but historical jobs can no longer start second sampling or resume from checkpoints.',
    '从已完成的干净 AV latent 放大并低噪细化；原提示词、图片与音频参考保持不变。': 'Upscale a completed clean AV latent and refine it with low noise. The original prompt, image, and audio references remain unchanged.',
    '目标分辨率': 'Target resolution',
    '二采模型': 'Second-pass model',
    '原始权重': 'Base weights',
    'Turbo LoRA': 'Turbo LoRA',
    '二次采样步数': 'Second-sampling steps',
    '加速力度': 'Acceleration',
    '重绘强度': 'Denoise strength',
    '二采固定使用原始 Base 权重与 SA Solver。': 'Second sampling uses the original Base weights and SA Solver.',
    '二采固定使用原始 Base 权重与 SA Solver。四档重绘强度分别使用 Denoise 0.10 / 0.20 / 0.25 / 0.30；加速力度只调度现有稀疏注意力，不插入 Forecast 步。': 'Second sampling is fixed to the original Base weights and SA Solver. The four denoise levels use 0.10 / 0.20 / 0.25 / 0.30; acceleration controls sparse attention only and never inserts Forecast steps.',
    '保真 · 低噪声': 'Preserve · low noise',
    '标准 · 官方基准': 'Standard · author baseline',
    '增强 · 更多细节': 'Enhance · more detail',
    '强重绘 · 高风险': 'Strong redraw · high risk',
    'Denoise 0.20 · 起始 Sigma 约 0.60': 'Denoise 0.20 · starting sigma about 0.60',
    '四档重绘强度分别使用 Denoise 0.10 / 0.20 / 0.25 / 0.30；后端按步数生成完整 Sigma 序列。加速力度只调度现有稀疏注意力，不插入 Forecast 步。': 'The four redraw levels use Denoise 0.10 / 0.20 / 0.25 / 0.30. The backend builds the full sigma schedule from the selected step count. Acceleration controls the existing sparse-attention policy without Forecast steps.',
    '默认 1 步 / 0.20，沿用 UltimateUpscale 的快速低噪工作点。更高重绘强度可能改变动作、物体和人物细节。': 'Default: 1 step / 0.20, using UltimateUpscale’s fast low-noise operating point. Higher denoise strength may change motion, objects, and character details.',
    '加入二次采样队列': 'Add to second-sampling queue',
    '控制台正在初始化': 'Console is initializing',
    '请先选择生成模式': 'Choose a generation mode first',
    '模型正在切换，请稍候': 'Model is switching; please wait',
    '启动模型加载': 'Starting model load',
    '检查运行环境': 'Checking runtime',
    '准备本地权重': 'Preparing local weights',
    '准备文本编码器': 'Preparing text encoder',
    '装配模型组件': 'Building model components',
    '编译预热视频VAE': 'Compiling and warming the video VAE',
    '整理主机内存': 'Preparing host memory',
    '完成运行时初始化': 'Finalizing runtime',
    '正在加载模型引擎': 'Loading model engine',
    '首次加载需要读取并装配模型权重': 'The first load reads and assembles model weights',
    '正在提交模型加载请求': 'Submitting the model-load request',
    '开始加载模型引擎': 'Starting model-engine load',
    '检查模型文件与CUDA执行环境': 'Checking model files and the CUDA runtime',
    '解析Linux本地模型权重': 'Resolving local Linux model weights',
    '准备Qwen文本编码器': 'Preparing the Qwen text encoder',
    '并行装配DiT、VAE与二采模型': 'Assembling DiT, VAE, and second-sampling models in parallel',
    'DiT与VAE模型组件已装配': 'DiT and VAE model components are ready',
    '预热视频VAE编译图': 'Compiling and warming the video VAE graph',
    '整理并锁定模型主机内存': 'Organizing and pinning model memory on the host',
    '完成调度器与运行时会话初始化': 'Finalizing the scheduler and runtime session',
    '模型引擎已就绪': 'Model engine is ready',
    '模型引擎加载失败': 'Model-engine load failed',
    '工作空间': 'Workspace',
    '这个文件夹中没有子文件夹': 'This folder has no subfolders',
    '正在切换…': 'Switching…',
    '正在释放…': 'Releasing…',
    '正在取消': 'Cancelling',
    '准备模型': 'Preparing model',
    '生成中': 'Generating',
    '断点已保存': 'Checkpoint saved',
    '等待抽卡决定': 'Awaiting preview decision',
    '失败': 'Failed',
    '已取消': 'Cancelled',
    '预计完成（含排队）': 'Estimated completion (including queue)',
    '模型就绪后预计生成': 'Estimated generation after model ready',
    '预计剩余': 'Estimated remaining',
    '精确流式': 'Exact streaming',
    '紧凑流式': 'Compact streaming',
    '显存自动优化': 'Automatic VRAM optimization',
    '查看断点预览': 'View checkpoint preview',
    '继续正式生成': 'Resume formal generation',
    '放弃本次抽卡': 'Discard this preview',
    '预览与下载': 'Preview and download',
    '取消': 'Cancel',
    '删除': 'Delete',
    '实际总耗时': 'Actual total time',
    '打开成片预览': 'Open video preview',
    '点击预览成片': 'Click to preview video',
    'H3 二次采样完成': 'H3 second sampling complete',
    '视频生成完成': 'Video generation complete',
    '打开预览与下载': 'Open preview and download',
    '继续二次采样': 'Continue second sampling',
    '你提交的视频任务': 'Your video job',
    '取消任务': 'Cancel job',
    '删除这条创作记录': 'Delete this creation record',
    '从底部开始创建第一条视频': 'Create your first video below',
    '提交后，排队、生成进度、预计完成时间和成片都会显示在这里。': 'After submission, queue status, progress, estimated completion, and the finished video appear here.',
    '当前没有正在执行的任务': 'No jobs are currently running',
    '等待队列为空': 'The queue is empty',
    '还没有历史任务': 'No job history yet',
    '未加载': 'Not loaded',
    '加载中': 'Loading',
    '已热身': 'Warm',
    '加载失败': 'Load failed',
    '可用': 'Available',
    '待选择模式': 'Choose a mode',
    '24GB 独立高速后端': '24GB dedicated high-speed backend',
    '16GB 独立紧凑后端': '16GB dedicated compact backend',
    '8GB 独立低比特后端': '8GB dedicated low-bit backend',
    '服务不可用': 'Service unavailable',
    '原分辨率': 'Original resolution',
    '估算中': 'Estimating',
    '未记录': 'Not recorded',
    '完整注意力': 'Full attention',
    '全程固定': 'Fixed throughout',
    '动态保护': 'Dynamic protection',
    '仅中段': 'Middle only',
    '离线': 'Offline',
    '线程': 'threads',
    '负载': 'load',
    '服务进程': 'service process',
    '已使用': 'used',
    '不可用': 'Unavailable',
    '未检测到 NVIDIA 监控接口': 'NVIDIA monitoring interface not detected',
    'nvidia-smi 未就绪': 'nvidia-smi is not ready',
    '正在取消…': 'Cancelling…',
    '退出当前模式会释放模型热态。确认退出？': 'Exiting this mode releases the warm model. Continue?',
    '删除该任务记录、上传帧和已生成视频？此操作不可恢复。': 'Delete this job record, uploaded frames, and generated video? This cannot be undone.',
    '正在创建 H3 二次采样任务…': 'Creating H3 second-sampling job…',
    '正在检查参数并提交任务…': 'Validating parameters and submitting the job…',
    '控制台仍在初始化，请稍候后重试': 'The console is still initializing. Try again shortly.',
    '请填写每个分镜的镜头内容': 'Fill in the content for every shot.',
    '请填写对象定义（subject_definitions）': 'Fill in subject_definitions.',
    '请填写保留规则（retention_analysis）': 'Fill in retention_analysis.',
    '请填写总体摘要（summary）': 'Fill in the overall summary.',
    '已开启BGM，请填写配乐风格': 'BGM is enabled; enter a music style.',
    '请输入完整提示词': 'Enter the complete prompt.',
    '请至少选择一项参考图片、视频或音频': 'Select at least one reference image, video, or audio file.',
    '参考图片最多9张': 'Up to 9 reference images are allowed.',
    '参考视频最多3段': 'Up to 3 reference videos are allowed.',
    '参考音频最多3段': 'Up to 3 reference audio files are allowed.',
    '开启': 'On',
    '服务器已保存MiMo Key；留空不会覆盖。': 'The server has a saved MiMo key; leaving this blank will not overwrite it.',
    '服务器尚未配置MiMo Key。': 'The server does not have a MiMo key configured.',
    '当前没有实际上传的参考素材。刷新页面后浏览器不会保留本地文件，请重新添加图片、视频或音频。': 'No reference media is currently uploaded. Browsers do not retain local files after refresh; add the image, video, or audio files again.',
    '说明 <Picture 1> 是哪个人物、物体、场景或画面风格，以及需要保留什么': 'Describe which person, object, setting, or visual style <Picture 1> represents and what to preserve.',
    '请先点击右上角设置，填写小米 MiMo API Key。': 'Open Settings in the top-right corner and enter the Xiaomi MiMo API key first.',
    '请先填写对象定义和保留规则，再让 MiMo 润色。': 'Fill in subject definitions and retention rules before asking MiMo to refine them.',
    '请先填写每个分镜的镜头内容，再润色画面。': 'Fill in every shot before refining the visuals.',
    '请先填写总体摘要，再让 MiMo 润色画面。': 'Fill in the overall summary before asking MiMo to refine the visuals.',
    '请先填写画面内声音或配乐描述，再润色声音。': 'Enter diegetic sound or a music description before refining sound.',
    '已开启BGM，请先填写配乐描述或引用音频。': 'BGM is enabled; enter a music description or reference an audio file first.',
    'Dense': 'Dense',
    '写什么：定义目标视频中需要反复引用的人物、物体、场景或声音，以及它来自哪个素材。\n怎么写：建议使用英文；每个对象单独一行，使用 <Subject N>、<Picture N>、<Video N> 或 <Audio N>。\n示例：<Subject 1> is the young girl from <Picture 1>, preserving her face, hairstyle and clothing.': 'What to write: Define each person, object, setting, or sound referenced repeatedly in the target video and identify its source media.\nHow to write: Use one object per line with <Subject N>, <Picture N>, <Video N>, or <Audio N>.\nExample: <Subject 1> is the young girl from <Picture 1>, preserving her face, hairstyle and clothing.',
    '写什么：说明每个参考对象在目标视频中保留、改变或借鉴到什么程度。\n怎么写：建议使用英文；每个标签单独一行，并选用 fully_preserved、partially_preserved、attribute_transfer 或 weak_reference。\n示例：<Subject 1> (appears throughout): fully_preserved - preserve identity, hairstyle and clothing.': 'What to write: State how much each referenced object should be preserved, changed, or borrowed in the target video.\nHow to write: Use one tag per line and choose fully_preserved, partially_preserved, attribute_transfer, or weak_reference.\nExample: <Subject 1> (appears throughout): fully_preserved - preserve identity, hairstyle and clothing.',
    '写什么：简要概括要生成什么视频，以及主要参考素材分别起什么作用。\n怎么写：建议使用英文；以 [reference generation]、[video editing] 等任务类型开头，只概括目标，不展开分镜细节。\n示例：[reference generation] Generate an 8-second scene in which <Subject 1> enters the room and speaks using <Audio 1> as the voice reference.': 'What to write: Briefly summarize the target video and the role of each primary reference.\nHow to write: Start with a task type such as [reference generation] or [video editing]. Summarize the goal without shot-level detail.\nExample: [reference generation] Generate an 8-second scene in which <Subject 1> enters the room and speaks using <Audio 1> as the voice reference.',
    '写什么：概括整个视频中真实存在于场景里的环境声和动作音效。\n怎么写：建议使用英文；写清声音来源，声音发生时应与画面动作对应；可以输入 @ 引用音频素材。\n示例：Quiet corridor ambience, approaching footsteps, restrained breathing, and subtle backpack-fabric friction.': 'What to write: Summarize real environmental and action sounds present in the scene.\nHow to write: Identify each source and synchronize sounds with visible actions. Use @ to reference audio media.\nExample: Quiet corridor ambience, approaching footsteps, restrained breathing, and subtle backpack-fabric friction.',
    '写什么：描述只有观众能听见的画外配乐。\n怎么写：建议使用英文；说明风格、情绪、节奏和音量，也可以输入 @ 引用音频素材。\n示例：A restrained slow piano score, warm and melancholic, mixed quietly below the dialogue.': 'What to write: Describe non-diegetic music heard only by the audience.\nHow to write: Specify style, mood, tempo, and level, or use @ to reference audio media.\nExample: A restrained slow piano score, warm and melancholic, mixed quietly below the dialogue.'
  }));

  const fragments = [
    ['当前GPU ', 'Current GPU '],
    ['任务 ', 'Job '],
    ['DiT 去噪', 'DiT denoising'],
    ['时间窗口', 'temporal window'],
    [' 已加入队列', ' added to the queue'],
    ['源卡片 ', 'Source card '],
    ['；原提示词、参考图片与参考音频会原样复用。', '; the original prompt, reference images, and reference audio will be reused unchanged.'],
    ['当前分辨率的分镜总时长必须在 ', 'The total shot duration at the current resolution must be between '],
    ['当前分辨率的视频时长必须在 ', 'Video duration at the current resolution must be between '],
    ['分镜总时长 ', 'Total shot duration '],
    ['与实际视频时长 ', ' does not match actual video duration '],
    ['不一致，请检查分镜时长', '; check shot durations'],
    ['断点预览设置为', 'Checkpoint preview is set to '],
    ['，不能高于正式生成画布', ', which cannot exceed the formal generation canvas'],
    ['参考音频 ', 'Reference audio '],
    [' 尚未使用：请在人物台词、画面内声音或BGM中引用 ', ' is unused; reference it in dialogue, diegetic sound, or BGM: '],
    ['当前分辨率的分镜总时长已达到', 'The total shot duration at the current resolution has reached '],
    ['，请先缩短现有分镜。', '; shorten the existing shots first.'],
    ['当前没有加载H3引擎', 'No H3 engine is currently loaded'],
    ['正在加载', 'Loading '],
    ['正在提交模型加载请求', 'Submitting model load request'],
    ['（含原始权重与LoRA开关），首次进入可能需要几十秒…', ' (base weights and LoRA switch); the first load may take tens of seconds…'],
    ['首次进入可能需要几十秒', 'the first load may take tens of seconds'],
    ['含原始权重与LoRA开关', 'includes base weights and the LoRA switch'],
    ['模型组件已准备 ', 'Model components ready: '],
    ['仅多参考模式需要：说明 ', 'Multi-reference mode only: specify which '],
    ['来自哪个', 'comes from which'],
    ['，以及需要保留什么。', ', and what must be preserved.'],
    ['用于访问设置了 ', 'Used to access a video service configured with '],
    [' 的视频服务。', '.'],
    ['V24统一帕累托调度：0为Dense，', 'V24 unified Pareto scheduler: 0 is Dense; '],
    ['为Human审核的发布质量拐点，', ' is the Human-reviewed release quality knee; '],
    ['–100为允许肉眼缺陷的激进区。', '–100 is the aggressive range where visible defects are accepted.'],
    ['第 ', 'Stop after step '],
    [' 步后停止', ''],
    ['有效 ', 'Effective '],
    ['当前可用 ', 'currently available '],
    ['精确流式', 'exact streaming'],
    ['显存自动优化', 'automatic VRAM optimization'],
    ['进入失败：', 'Failed to enter: '],
    ['退出失败：', 'Failed to exit: '],
    ['切换失败：', 'Switch failed: '],
    ['调整顺序失败：', 'Failed to reorder: '],
    ['服务响应超时；', 'Service response timed out; '],
    ['请检查8090端口转发，或等待当前计算阶段结束', 'check port 8090 forwarding or wait for the current compute stage to finish'],
    ['预览：', 'Preview: '],
    ['时间窗口约 ', 'Temporal window about '],
    ['；Overlap 与 latent 交叉融合自动处理。', '; overlap and latent crossfade are automatic.'],
    [' 帧', ' frames'],
    ['自动等比降分辨率：图片', 'Automatic proportional downsampling: images '],
    [' · 视频', ' · videos '],
    ['最长时长', 'maximum duration'],
    ['目标时长 · 当前画布最多', 'Target duration · current canvas maximum '],
    ['目标时长', 'Target duration'],
    ['需为', 'Must be '],
    ['镜头时长', 'Shot duration'],
    ['镜头内容', 'shot content'],
    ['持续时间（秒）', 'Duration (seconds)'],
    ['上移', 'Move up'],
    ['下移', 'Move down'],
    ['删除素材', 'Remove media'],
    ['插入当前提示词', 'Insert into current prompt'],
    ['素材用途', 'Reference purpose'],
    ['用途（可选）', 'Purpose (optional)'],
    ['请先添加参考文件', 'Add reference files first'],
    ['润色完成，其他板块未修改。', ' refinement complete; other sections were not changed.'],
    ['正在润色…', 'Refining…'],
    ['约 ', 'About '],
    ['等待 ', 'Waiting '],
    ['自定义 ', 'Custom '],
    ['原始权重', 'Base weights'],
    ['实际/', ' actual / '],
    ['预测', ' forecast'],
    ['注意力', ' attention'],
    ['总步', ' total steps'],
    ['二采实际步', ' second-sampling actual steps'],
    ['实际步', ' actual steps'],
    ['加速 ', 'Acceleration '],
    ['加速', 'Acceleration '],
    ['硬上限 ', 'hard limit '],
    ['保留峰值 ', 'reserved peak '],
    ['不含服务启动与模型预加载', 'Excludes service startup and model preload'],
    ['由 ', 'From '],
    [' 二次采样', ' second sampling'],
    ['已在第 ', 'Stopped after step '],
    [' 步停止', ''],
    ['正式状态已落盘，当前任务不占用 GPU；恢复时重新进入队列。', 'Formal state is saved. The job does not occupy the GPU and re-enters the queue when resumed.'],
    ['在线 · 正在切换引擎', 'Online · switching engine'],
    ['在线 · 正在预加载模型', 'Online · preloading model'],
    ['在线 · 模型加载失败', 'Online · model load failed'],
    ['在线 · ', 'Online · '],
    ['待选择模式', 'Choose a mode'],
    ['未加载', 'Not loaded'],
    ['已热身', 'Warm'],
    ['加载失败', 'Load failed'],
    ['独立紧凑', ' dedicated compact '],
    ['默认计算', 'Default compute'],
    ['自动显存', 'Auto VRAM'],
    ['低比特', ' low-bit '],
    ['独立高速', ' dedicated high-speed '],
    ['后端', 'backend'],
    ['多参考', 'multi-reference'],
    ['文生视频', 'text-to-video'],
    ['原始采样', 'base sampling'],
    ['LoRA 六步', 'six-step LoRA'],
    ['首尾帧生视频', 'first/last-frame-to-video'],
    ['首帧生视频', 'first-frame-to-video'],
    ['尾帧生视频', 'last-frame-to-video'],
    ['图 / ', ' images / '],
    ['视频 / ', ' videos / '],
    ['音频', ' audio'],
    ['帧', ' frames'],
    ['步', ' steps'],
    ['秒', 's'],
    ['分', 'm ']
  ].sort((a, b) => b[0].length - a[0].length);

  const textState = new WeakMap();
  const attributeState = new WeakMap();
  let locale = readInitialLocale();
  let observer = null;

  function readInitialLocale() {
    const requested = new URLSearchParams(window.location.search).get('lang');
    if (SUPPORTED.has(requested)) return requested;
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (SUPPORTED.has(saved)) return saved;
    } catch (_) {}
    return 'zh-CN';
  }

  function shouldSkip(node) {
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return Boolean(element?.closest(
      'textarea, code, pre, [data-i18n-skip], .job-title, .conversation-user p, .reference-chip-copy small'
    ));
  }

  function translateSource(source) {
    if (locale !== 'en' || !source) return source;
    const direct = exact.get(source);
    if (direct !== undefined) return direct;
    if (!/[\u3400-\u9fff]/u.test(source)) return source;
    let translated = source;
    for (const [from, to] of fragments) translated = translated.split(from).join(to);
    return translated;
  }

  function translatedText(source) {
    const match = source.match(/^(\s*)([\s\S]*?)(\s*)$/);
    if (!match || !match[2]) return source;
    return `${match[1]}${translateSource(match[2])}${match[3]}`;
  }

  function translateTextNode(node) {
    if (shouldSkip(node)) return;
    const raw = node.data;
    let state = textState.get(node);
    if (!state || raw !== state.lastApplied) state = {source: raw, lastApplied: raw};
    const next = locale === 'en' ? translatedText(state.source) : state.source;
    state.lastApplied = next;
    textState.set(node, state);
    if (raw !== next) node.data = next;
  }

  function translateAttribute(element, name) {
    if (element.closest('code, pre, [data-i18n-skip], .job-title, .conversation-user p, .reference-chip-copy small')) return;
    const raw = element.getAttribute(name);
    if (raw == null) return;
    let states = attributeState.get(element);
    if (!states) { states = new Map(); attributeState.set(element, states); }
    let state = states.get(name);
    if (!state || raw !== state.lastApplied) state = {source: raw, lastApplied: raw};
    const next = locale === 'en' ? translateSource(state.source) : state.source;
    state.lastApplied = next;
    states.set(name, state);
    if (raw !== next) element.setAttribute(name, next);
  }

  function translateElement(element) {
    for (const name of ['placeholder', 'title', 'aria-label']) {
      if (element.hasAttribute(name)) translateAttribute(element, name);
    }
    if (shouldSkip(element)) return;
    for (const child of element.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) translateTextNode(child);
    }
  }

  function translateTree(root = document.body) {
    if (!root) return;
    if (root.nodeType === Node.TEXT_NODE) { translateTextNode(root); return; }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_NODE) return;
    if (root.nodeType === Node.ELEMENT_NODE) translateElement(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
      else translateElement(node);
    }
  }

  function updateSwitch() {
    document.documentElement.lang = locale;
    document.querySelectorAll('#languageSwitch [data-locale]').forEach(button => {
      const active = button.dataset.locale === locale;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  function setLocale(nextLocale, {persist = true} = {}) {
    if (!SUPPORTED.has(nextLocale) || nextLocale === locale) { updateSwitch(); return; }
    locale = nextLocale;
    if (persist) {
      try { localStorage.setItem(STORAGE_KEY, locale); } catch (_) {}
    }
    updateSwitch();
    translateTree(document.body);
    window.dispatchEvent(new CustomEvent('h3serve:locale-changed', {detail: {locale}}));
  }

  function start() {
    updateSwitch();
    translateTree(document.body);
    document.querySelectorAll('#languageSwitch [data-locale]').forEach(button => {
      button.addEventListener('click', () => setLocale(button.dataset.locale));
    });
    observer = new MutationObserver(records => {
      const roots = new Set();
      for (const record of records) {
        if (record.type === 'characterData') roots.add(record.target);
        else if (record.type === 'attributes') roots.add(record.target);
        else for (const node of record.addedNodes) roots.add(node);
      }
      for (const root of roots) translateTree(root);
    });
    observer.observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ['placeholder', 'title', 'aria-label']
    });
  }

  window.H3I18n = Object.freeze({
    get locale() { return locale; },
    setLocale,
    translateTree,
    t(source) { return translateSource(source); }
  });

  document.addEventListener('DOMContentLoaded', start, {once: true});
})();
