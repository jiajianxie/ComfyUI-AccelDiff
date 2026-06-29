import { app } from "../../scripts/app.js";

// 模型配置 - 与 Python 端的 MODEL_PARAMS 保持一致
// node_type:
//   "sampler" -> 没有 model 输入，只输出 SAMPLER
//   "model"   -> 输入 MODEL，输出 MODEL
const MODEL_CONFIG = {
    "AdaptiveDiff": {
        node_type: "sampler",
        params: ["max_skip_steps", "threshold"],
    },
    "EasyCache": {
        node_type: "sampler",
        params: ["threshold", "ret_steps"],
    },
    "TeaCache": {
        node_type: "model",
        params: ["teacache_model_type", "rel_l1_thresh", "start_percent", "end_percent", "cache_device"],
    },
    "MagCache": {
        node_type: "model",
        params: ["magcache_model_type", "magcache_thresh", "retention_ratio", "magcache_K", "start_step", "end_step"],
    },
    "Model_B": {
        node_type: "model",
        params: ["param1", "param2", "param3"],
    },
    "Model_C": {
        node_type: "model",
        params: ["param1", "param2", "param3", "param4", "param5", "param6"],
    },
};

// 所有可能的 widget 参数名（除 model_name 之外，sampler_name 单独处理）
delete MODEL_CONFIG.Model_B;
delete MODEL_CONFIG.Model_C;

const ALL_PARAM_WIDGETS = [
    "max_skip_steps", "threshold", "ret_steps",
    "param1", "param2", "param3", "param4", "param5", "param6",
    "teacache_model_type", "rel_l1_thresh", "start_percent", "end_percent", "cache_device",
    "magcache_model_type", "magcache_thresh", "retention_ratio", "magcache_K", "start_step", "end_step",
];

// sampler_name 只在 node_type === "sampler" 时显示
const SAMPLER_ONLY_WIDGETS = ["sampler_name"];

// 受动态显隐管理的 input slot 名（在 INPUT_TYPES 中存在的 optional input）
const DYNAMIC_INPUT_NAMES = ["model"];
// 受动态显隐管理的 output slot 名（顺序需与 RETURN_NAMES 一致）
const DYNAMIC_OUTPUT_NAMES = ["sampler", "model"];

// 每种 node_type 期望显示的输入/输出
const TYPE_IO = {
    sampler: { inputs: [],        outputs: ["sampler"] },
    model:   { inputs: ["model"], outputs: ["model"]   },
};

// ---------- widget 显隐 ----------
function hideWidget(node, widget, suffix = "") {
    if (widget.hidden) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;
    widget.hidden = true;
    widget.type = "hidden" + suffix;
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => undefined;
}

function showWidget(node, widget) {
    if (!widget.hidden) return;
    widget.type = widget.origType;
    widget.computeSize = widget.origComputeSize;
    widget.serializeValue = widget.origSerializeValue;
    delete widget.hidden;
}

// ---------- input slot 显隐 ----------
// ComfyUI 中 input slot 没有 type="hidden" 概念，最稳的做法是 add/removeInput
function ensureInput(node, name, type) {
    const idx = node.inputs ? node.inputs.findIndex(i => i.name === name) : -1;
    if (idx === -1) {
        node.addInput(name, type);
    }
}

function ensureNoInput(node, name) {
    const idx = node.inputs ? node.inputs.findIndex(i => i.name === name) : -1;
    if (idx !== -1) {
        // 如果已有连接，先断开
        const slot = node.inputs[idx];
        if (slot.link != null && app.graph) {
            app.graph.removeLink(slot.link);
        }
        node.removeInput(idx);
    }
}

// ---------- output slot 显隐 ----------
function ensureOutput(node, name, type) {
    const idx = node.outputs ? node.outputs.findIndex(o => o.name === name) : -1;
    if (idx === -1) {
        node.addOutput(name, type);
    }
}

function ensureNoOutput(node, name) {
    const idx = node.outputs ? node.outputs.findIndex(o => o.name === name) : -1;
    if (idx !== -1) {
        const slot = node.outputs[idx];
        // 移除 slot 上的所有连线
        if (slot.links && slot.links.length && app.graph) {
            // 拷贝一份，避免遍历过程中修改
            for (const linkId of [...slot.links]) {
                app.graph.removeLink(linkId);
            }
        }
        node.removeOutput(idx);
    }
}

// 输入类型映射
const INPUT_TYPE_MAP = {
    model: "MODEL",
};
// 输出类型映射
const OUTPUT_TYPE_MAP = {
    sampler: "SAMPLER",
    model: "MODEL",
};

// ---------- 整体更新 ----------
function updateNodeIO(node, modelName) {
    const cfg = MODEL_CONFIG[modelName];
    if (!cfg) return;

    // 1. widget 显隐
    const visibleParams = cfg.params || [];
    for (const paramName of ALL_PARAM_WIDGETS) {
        const w = node.widgets && node.widgets.find(x => x.name === paramName);
        if (!w) continue;
        if (visibleParams.includes(paramName)) {
            showWidget(node, w);
        } else {
            hideWidget(node, w);
        }
    }

    // 1.5 sampler_name 仅 sampler 类节点显示
    for (const name of SAMPLER_ONLY_WIDGETS) {
        const w = node.widgets && node.widgets.find(x => x.name === name);
        if (!w) continue;
        if (cfg.node_type === "sampler") {
            showWidget(node, w);
        } else {
            hideWidget(node, w);
        }
    }

    // 2. input/output slot 显隐
    const io = TYPE_IO[cfg.node_type] || { inputs: [], outputs: [] };

    for (const name of DYNAMIC_INPUT_NAMES) {
        if (io.inputs.includes(name)) {
            ensureInput(node, name, INPUT_TYPE_MAP[name]);
        } else {
            ensureNoInput(node, name);
        }
    }

    for (const name of DYNAMIC_OUTPUT_NAMES) {
        if (io.outputs.includes(name)) {
            ensureOutput(node, name, OUTPUT_TYPE_MAP[name]);
        } else {
            ensureNoOutput(node, name);
        }
    }

    // 3. 重新计算节点尺寸
    node.setSize(node.computeSize());
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "AccelDiff.Unified",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AccelDiffUnified") return;

        // onNodeCreated: 绑定 model_name 变更回调
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const modelWidget = node.widgets && node.widgets.find(w => w.name === "model_name");

            if (modelWidget) {
                const origCallback = modelWidget.callback;
                modelWidget.callback = function () {
                    if (origCallback) origCallback.apply(this, arguments);
                    updateNodeIO(node, modelWidget.value);
                };

                // 初次创建时同步 UI
                setTimeout(() => updateNodeIO(node, modelWidget.value), 0);
            }
            return result;
        };

        // onConfigure: 加载工作流时恢复 UI 状态
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const node = this;
            const modelWidget = node.widgets && node.widgets.find(w => w.name === "model_name");
            if (modelWidget) {
                setTimeout(() => updateNodeIO(node, modelWidget.value), 0);
            }
        };
    },
});
