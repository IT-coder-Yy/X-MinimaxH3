import { app } from "../../scripts/app.js";
const DURATION_AWARE_NODES = new Set([
    "H3ServeRef2VAPresetGenerate", "H3ServeFL2VAPresetGenerate",
    "H3ServeRef2VAAdvancedGenerate", "H3ServeFL2VAAdvancedGenerate",
    "H3ServeRef2VACheckpointSubmit", "H3ServeFL2VACheckpointSubmit",
]);
const SHORT_EDGES = { "360p": 360, "480p": 480, "720p": 720, "1080p": 1080 };
const ASPECT_RATIOS = {
    "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4],
    "16:9": [16, 9], "9:16": [9, 16],
};
const MAX_NATIVE_PIXEL_FRAMES = 1920 * 1088 * 192;
const MIN_NODE_HEIGHTS = {
    H3ServeFL2VAPresetGenerate: 720,
    H3ServeRef2VAPresetGenerate: 860,
    H3ServeFL2VAAdvancedGenerate: 760,
    H3ServeRef2VAAdvancedGenerate: 900,
    H3ServeFL2VACheckpointSubmit: 800,
    H3ServeRef2VACheckpointSubmit: 940,
};

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidgetVisible(node, name, visible) {
    const target = widget(node, name);
    if (!target) return;
    if (!visible && !target._h3Hidden) {
        target._h3OriginalType = target.type;
        target._h3OriginalComputeSize = target.computeSize;
        target.type = "h3-hidden-widget";
        target.computeSize = () => [0, -4];
        target._h3Hidden = true;
    } else if (visible && target._h3Hidden) {
        target.type = target._h3OriginalType;
        target.computeSize = target._h3OriginalComputeSize;
        target._h3Hidden = false;
    }
    const computed = node.computeSize();
    node.setSize([
        Math.max(node._h3MinWidth || 640, Number(node.size?.[0] || 0)),
        Math.max(
            node._h3MinHeight || 0,
            Number(node.size?.[1] || 0),
            Number(computed?.[1] || 0),
        ),
    ]);
    node.setDirtyCanvas(true, true);
}

function bindPreviewControls(node) {
    const mode = widget(node, "preview_mode");
    if (!mode) return;
    const controlled = ["预览位置", "预览分辨率", "LoRA预览步数"];
    const jobId = widget(node, "断点任务ID");
    const action = widget(node, "断点动作");
    const stateSentinels = new Set([
        "新建任务", "等待选择", "继续生成", "放弃生成",
    ]);
    if (jobId && stateSentinels.has(String(jobId.value || ""))) {
        // Repair workflows saved before the OUTPUT_NODE
        // control_after_generate placeholder was represented explicitly.
        jobId.value = "";
    }
    if (action && !stateSentinels.has(String(action.value || ""))) {
        action.value = "新建任务";
    }
    if (jobId) setWidgetVisible(node, "断点任务ID", false);
    if (action) setWidgetVisible(node, "断点动作", false);

    const runAction = async (value) => {
        if (!jobId?.value || !action) return;
        action.value = value;
        resumeButton.disabled = true;
        discardButton.disabled = true;
        try {
            await app.queuePrompt(0, 1);
        } catch (error) {
            action.value = "等待选择";
            resumeButton.disabled = false;
            discardButton.disabled = false;
            throw error;
        }
    };
    const resumeButton = node.addWidget(
        "button", "继续生成", null, () => runAction("继续生成"),
    );
    const discardButton = node.addWidget(
        "button", "放弃生成", null, () => runAction("放弃生成"),
    );
    resumeButton.serialize = false;
    discardButton.serialize = false;

    const showActions = (visible) => {
        setWidgetVisible(node, resumeButton.name, visible);
        setWidgetVisible(node, discardButton.name, visible);
        resumeButton.disabled = !visible;
        discardButton.disabled = !visible;
    };
    const refresh = () => {
        const enabled = mode.value === "开启";
        const sampling = Math.max(2, Number(widget(node, "sampling_steps")?.value || 8));
        const step = widget(node, "预览位置");
        if (step) {
            step.options.max = sampling - 1;
            step.value = Math.min(Math.max(1, Number(step.value || 1)), sampling - 1);
        }
        controlled.forEach((name) => setWidgetVisible(node, name, enabled));
        showActions(enabled && Boolean(jobId?.value));
    };
    const originalCallback = mode.callback;
    mode.callback = function (value) {
        originalCallback?.call(this, value);
        refresh();
    };
    const sampling = widget(node, "sampling_steps");
    if (sampling) {
        const originalSamplingCallback = sampling.callback;
        sampling.callback = function (value) {
            originalSamplingCallback?.call(this, value);
            refresh();
        };
    }
    const originalExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        originalExecuted?.call(this, message);
        const state = message?.h3_checkpoint?.[0];
        if (!state || !jobId || !action) return;
        if (state.status === "checkpointed") {
            jobId.value = state.job_id;
            action.value = "等待选择";
            showActions(true);
        } else if (state.status === "succeeded" || state.status === "reset") {
            jobId.value = "";
            action.value = "新建任务";
            showActions(false);
        }
    };
    requestAnimationFrame(refresh);
}

function nearest32(value) {
    return Math.max(32, Math.floor(Number(value) / 32 + 0.5) * 32);
}

function nodeGeometry(node) {
    const width = Number(widget(node, "width")?.value);
    const height = Number(widget(node, "height")?.value);
    if (width && height) return { width, height };
    const resolution = widget(node, "resolution")?.value || "480p";
    const ratio = widget(node, "aspect_ratio")?.value || "16:9";
    const shortEdge = SHORT_EDGES[resolution] || 480;
    const [rw, rh] = ASPECT_RATIOS[ratio] || [16, 9];
    return rw >= rh
        ? { width: nearest32(shortEdge * rw / rh), height: nearest32(shortEdge) }
        : { width: nearest32(shortEdge), height: nearest32(shortEdge * rh / rw) };
}

function bindDurationBudget(node) {
    const duration = widget(node, "duration_seconds");
    if (!duration) return;
    const refresh = () => {
        const { width, height } = nodeGeometry(node);
        const rawFrames = Math.min(362, Math.floor(MAX_NATIVE_PIXEL_FRAMES / (width * height)));
        const legalFrames = 5 + 17 * Math.max(0, Math.floor((rawFrames - 5) / 17));
        const exactMaximum = Math.min(15, legalFrames / 24);
        const sliderMaximum = Math.floor(exactMaximum * 2 + 1e-9) / 2;
        duration.options = duration.options || {};
        duration.options.max = sliderMaximum;
        duration.value = Math.min(Number(duration.value || 5), sliderMaximum);
        node.setDirtyCanvas(true, true);
    };
    ["resolution", "aspect_ratio", "width", "height"].forEach((name) => {
        const control = widget(node, name);
        if (!control) return;
        const originalCallback = control.callback;
        control.callback = function (value) {
            originalCallback?.call(this, value);
            refresh();
        };
    });
    requestAnimationFrame(refresh);
}

function normalizeAcceleration(node) {
    // Older saved workflows serialize widget values positionally. If they
    // predate the acceleration widget, ComfyUI may hydrate a string into this
    // numeric control and display NaN. Repair it after workflow hydration.
    requestAnimationFrame(() => {
        const acceleration = widget(node, "acceleration");
        if (acceleration && !Number.isFinite(Number(acceleration.value))) {
            acceleration.value = 0;
            node.setDirtyCanvas(true, true);
        }
    });
}

function ensureReadableNodeSize(node, nodeType) {
    // Saved workflows may carry the old narrow 520px node size. Re-apply the
    // readable minimum after hydration while preserving any larger user size.
    node._h3MinWidth = 640;
    node._h3MinHeight = MIN_NODE_HEIGHTS[nodeType] || 720;
    requestAnimationFrame(() => {
        const computed = node.computeSize();
        node.setSize([
            Math.max(node._h3MinWidth, Number(node.size?.[0] || 0), Number(computed?.[0] || 0)),
            Math.max(node._h3MinHeight, Number(node.size?.[1] || 0), Number(computed?.[1] || 0)),
        ]);
        node.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "H3Serve.GenerationControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!DURATION_AWARE_NODES.has(nodeData.name)) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            normalizeAcceleration(this);
            ensureReadableNodeSize(this, nodeData.name);
            bindDurationBudget(this);
            bindPreviewControls(this);
        };
    },
});
