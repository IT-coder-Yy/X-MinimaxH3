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
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

function bindPreviewControls(node) {
    const mode = widget(node, "preview_mode");
    if (!mode) return;
    const controlled = ["预览位置", "预览分辨率", "LoRA预览步数"];
    const refresh = () => {
        const enabled = mode.value === "开启";
        const sampling = Math.max(2, Number(widget(node, "sampling_steps")?.value || 8));
        const step = widget(node, "预览位置");
        if (step) {
            step.options.max = sampling - 1;
            step.value = Math.min(Math.max(1, Number(step.value || 1)), sampling - 1);
        }
        controlled.forEach((name) => setWidgetVisible(node, name, enabled));
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

app.registerExtension({
    name: "H3Serve.GenerationControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!DURATION_AWARE_NODES.has(nodeData.name)) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            bindDurationBudget(this);
            bindPreviewControls(this);
        };
    },
});
