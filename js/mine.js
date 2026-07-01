import { app } from "../../scripts/app.js";

// ========== Sampler 方法对应的参数 ==========
const SAMPLER_METHOD_CONFIG = {
    "None": { params: [] },
    "AdaptiveDiff": { params: ["max_skip_steps", "threshold"] },
    "EasyCache": { params: ["ec_threshold", "ret_steps"] },
    "SADA": { params: ["max_interval", "acc_start", "acc_end", "lagrange_term", "lagrange_int", "lagrange_step"] },
    "ZEUS": { params: ["zeus_denominator", "zeus_modular", "zeus_acc_start", "zeus_acc_end", "zeus_interp_mode", "zeus_caching_mode", "zeus_max_interval", "zeus_lagrange_term", "zeus_lagrange_int", "zeus_lagrange_step"] },
};

// ========== Model 方法对应的参数 ==========
const MODEL_METHOD_CONFIG = {
    "None": { params: [] },
    "TeaCache": { params: ["teacache_model_type", "rel_l1_thresh", "start_percent", "end_percent", "cache_device"] },
    "MagCache": { params: ["magcache_model_type", "magcache_thresh", "retention_ratio", "magcache_K", "start_step", "end_step"] },
    "TaylorSeer": { params: ["taylorseer_model_type", "max_order", "fresh_threshold", "first_enhance"] },
    "HiCache": { params: ["hicache_model_type", "hicache_prediction_mode", "hicache_max_order", "hicache_fresh_threshold", "hicache_first_enhance", "hicache_scale_factor"] },
    "SeaCache": { params: ["seacache_model_type", "seacache_thresh", "seacache_power_exp", "seacache_ret_steps"] },
};

// SADA 已改为纯 sampler 方法，不再有 model 侧参数

// ========== TaylorSeer 各模型默认参数（来自官方代码） ==========
const TAYLORSEER_DEFAULTS = {
    "flux":          { max_order: 1, fresh_threshold: 6, first_enhance: 3 },
    "hunyuan_video": { max_order: 1, fresh_threshold: 5, first_enhance: 1 },
    "wan2.1":        { max_order: 1, fresh_threshold: 5, first_enhance: 1 },
    "hidream":       { max_order: 2, fresh_threshold: 4, first_enhance: 1 },
};

// ========== HiCache 各模型默认参数 ==========
const HICACHE_DEFAULTS = {
    "flux":          { hicache_max_order: 1, hicache_fresh_threshold: 6, hicache_first_enhance: 3, hicache_scale_factor: 0.7 },
    "hunyuan_video": { hicache_max_order: 1, hicache_fresh_threshold: 5, hicache_first_enhance: 1, hicache_scale_factor: 0.5 },
    "wan2.1":        { hicache_max_order: 1, hicache_fresh_threshold: 5, hicache_first_enhance: 1, hicache_scale_factor: 0.5 },
    "hidream":       { hicache_max_order: 1, hicache_fresh_threshold: 4, hicache_first_enhance: 1, hicache_scale_factor: 0.5 },
};

// ========== SeaCache 各模型默认参数（来自官方代码） ==========
const SEACACHE_DEFAULTS = {
    "flux":          { seacache_thresh: 0.3, seacache_power_exp: 2.0, seacache_ret_steps: 1 },
    "hunyuan_video": { seacache_thresh: 0.20, seacache_power_exp: 3.0, seacache_ret_steps: 1 },
    "wan2.1":        { seacache_thresh: 0.2, seacache_power_exp: 3.0, seacache_ret_steps: 5 },
};

// 所有可能的参数 widget 名
const ALL_PARAM_WIDGETS = [
    "max_skip_steps", "threshold", "ec_threshold", "ret_steps",
    "lagrange_term", "lagrange_int", "lagrange_step",
    "max_interval", "acc_start", "acc_end",
    "teacache_model_type", "rel_l1_thresh", "start_percent", "end_percent", "cache_device",
    "magcache_model_type", "magcache_thresh", "retention_ratio", "magcache_K", "start_step", "end_step",
    "taylorseer_model_type", "max_order", "fresh_threshold", "first_enhance",
    "hicache_model_type", "hicache_prediction_mode", "hicache_max_order", "hicache_fresh_threshold", "hicache_first_enhance", "hicache_scale_factor",
    "seacache_model_type", "seacache_thresh", "seacache_power_exp", "seacache_ret_steps",
    "zeus_denominator", "zeus_modular", "zeus_acc_start", "zeus_acc_end", "zeus_interp_mode", "zeus_caching_mode", "zeus_max_interval", "zeus_lagrange_term", "zeus_lagrange_int", "zeus_lagrange_step",
];

// 输入类型映射
const INPUT_TYPE_MAP = { model: "MODEL" };

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

// ---------- input slot 显隐（动态增删，input 不涉及 RETURN_TYPES 索引） ----------
function ensureInput(node, name, type) {
    const idx = node.inputs ? node.inputs.findIndex(i => i.name === name) : -1;
    if (idx === -1) {
        node.addInput(name, type);
    }
}

function ensureNoInput(node, name) {
    const idx = node.inputs ? node.inputs.findIndex(i => i.name === name) : -1;
    if (idx !== -1) {
        const slot = node.inputs[idx];
        if (slot.link != null && app.graph) {
            app.graph.removeLink(slot.link);
        }
        node.removeInput(idx);
    }
}

// ---------- output slot 动态增删（参考 input slot 的 ensureInput/ensureNoInput 策略） ----------
// 固定的 output 定义：RETURN_TYPES index → 类型
const OUTPUT_SLOT_DEFS = [
    { name: "sampler", type: "SAMPLER", returnIndex: 0 },  // RETURN_TYPES[0]
    { name: "model",   type: "MODEL",   returnIndex: 1 },  // RETURN_TYPES[1]
];

function ensureOutput(node, name, type, slotIndex) {
    // 检查是否已存在该名称的 output
    const idx = node.outputs ? node.outputs.findIndex(o => o.name === name) : -1;
    if (idx === -1) {
        // 添加 output slot
        node.addOutput(name, type);
        // 设置 slot_index 以确保与 RETURN_TYPES 的索引对应正确
        const newSlot = node.outputs[node.outputs.length - 1];
        newSlot.slot_index = slotIndex;
    } else {
        // 已存在，确保 slot_index 正确
        node.outputs[idx].slot_index = slotIndex;
    }
}

function ensureNoOutput(node, name) {
    const idx = node.outputs ? node.outputs.findIndex(o => o.name === name) : -1;
    if (idx !== -1) {
        const slot = node.outputs[idx];
        // 断开已有连接
        if (slot.links && slot.links.length && app.graph) {
            for (const linkId of [...slot.links]) {
                app.graph.removeLink(linkId);
            }
        }
        node.removeOutput(idx);
    }
}

// ---------- 分组标题 widget ----------
// 这些是纯展示性的标题 widget，用于分隔 sampler 和 model 两组参数
const SEPARATOR_SAMPLER = "__sep_sampler__";
const SEPARATOR_MODEL = "__sep_model__";

function addSeparatorWidget(node, name, label) {
    // 检查是否已存在
    if (node.widgets && node.widgets.find(w => w.name === name)) return;
    const w = node.addCustomWidget({
        name: name,
        type: "separator",
        value: label,
        computeSize: function () { return [0, 26]; },
        draw: function (ctx, node, widgetWidth, y, widgetHeight) {
            const margin = 10;
            const midY = y + widgetHeight / 2;
            ctx.save();
            // 绘制分隔线
            ctx.strokeStyle = "#666";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin, midY);
            ctx.lineTo(widgetWidth - margin, midY);
            ctx.stroke();
            // 绘制标题背景
            ctx.font = "bold 12px Arial";
            const textWidth = ctx.measureText(this.value).width;
            const textX = (widgetWidth - textWidth) / 2;
            ctx.fillStyle = node.bgcolor || "#353535";
            ctx.fillRect(textX - 8, midY - 8, textWidth + 16, 16);
            // 绘制标题文字
            ctx.fillStyle = "#0cf";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(this.value, widgetWidth / 2, midY);
            ctx.restore();
        },
        serializeValue: function () { return undefined; },
        // 不可交互
        mouse: function () { return false; },
    });
    return w;
}

// ---------- 重排 widgets 顺序确保分组正确 ----------
function reorderWidgets(node) {
    if (!node.widgets) return;

    const samplerMethodW = node.widgets.find(w => w.name === "sampler_method");
    const modelMethodW = node.widgets.find(w => w.name === "model_method");
    const samplerSep = node.widgets.find(w => w.name === SEPARATOR_SAMPLER);
    const modelSep = node.widgets.find(w => w.name === SEPARATOR_MODEL);
    const samplerNameW = node.widgets.find(w => w.name === "sampler_name");

    if (!samplerMethodW || !modelMethodW) return;

    const samplerMethod = samplerMethodW.value;
    const modelMethod = modelMethodW.value;

    const samplerCfg = SAMPLER_METHOD_CONFIG[samplerMethod] || { params: [] };
    const modelCfg = MODEL_METHOD_CONFIG[modelMethod] || { params: [] };

    // 构建新的 widget 排列顺序
    const newOrder = [];

    // 1. Sampler 分组: 标题 → sampler_method 选择器 → sampler_name → sampler参数
    if (samplerSep) newOrder.push(samplerSep);
    newOrder.push(samplerMethodW);
    if (samplerMethod !== "None") {
        if (samplerNameW) newOrder.push(samplerNameW);
        for (const paramName of samplerCfg.params) {
            const pw = node.widgets.find(x => x.name === paramName);
            if (pw) newOrder.push(pw);
        }
    }

    // 2. Model 分组: 标题 → model_method选择器 → model参数
    if (modelSep) newOrder.push(modelSep);
    newOrder.push(modelMethodW);
    if (modelMethod !== "None") {
        for (const paramName of modelCfg.params) {
            const pw = node.widgets.find(x => x.name === paramName);
            if (pw) newOrder.push(pw);
        }
    }

    // 3. 剩余隐藏的 widget（保证序列化正常）
    for (const w of node.widgets) {
        if (!newOrder.includes(w)) {
            newOrder.push(w);
        }
    }

    node.widgets = newOrder;
}



// ---------- 整体更新 ----------
function updateNodeIO(node) {
    const samplerMethodW = node.widgets && node.widgets.find(w => w.name === "sampler_method");
    const modelMethodW = node.widgets && node.widgets.find(w => w.name === "model_method");
    if (!samplerMethodW || !modelMethodW) return;

    const samplerMethod = samplerMethodW.value;
    const modelMethod = modelMethodW.value;

    const samplerCfg = SAMPLER_METHOD_CONFIG[samplerMethod] || { params: [] };
    const modelCfg = MODEL_METHOD_CONFIG[modelMethod] || { params: [] };

    // 合并两个选择器应显示的参数
    const visibleParams = new Set([...samplerCfg.params, ...modelCfg.params]);

    // 1. 参数 widget 显隐
    for (const paramName of ALL_PARAM_WIDGETS) {
        const w = node.widgets && node.widgets.find(x => x.name === paramName);
        if (!w) continue;
        if (visibleParams.has(paramName)) {
            showWidget(node, w);
        } else {
            hideWidget(node, w);
        }
    }

    // 2. sampler_name 仅在 sampler_method !== "None" 时显示
    const samplerNameW = node.widgets && node.widgets.find(x => x.name === "sampler_name");
    if (samplerNameW) {
        if (samplerMethod !== "None") {
            showWidget(node, samplerNameW);
        } else {
            hideWidget(node, samplerNameW);
        }
    }

    // 3. 分组标题 separator 显隐
    addSeparatorWidget(node, SEPARATOR_SAMPLER, "━━━ Sampler Method ━━━");
    addSeparatorWidget(node, SEPARATOR_MODEL, "━━━ Model Method ━━━");

    const samplerSep = node.widgets.find(w => w.name === SEPARATOR_SAMPLER);
    const modelSep = node.widgets.find(w => w.name === SEPARATOR_MODEL);

    // 分组标题始终显示
    if (samplerSep) showWidget(node, samplerSep);
    if (modelSep) showWidget(node, modelSep);

    // 4. model_method 选择器始终显示（SADA 不再影响它）
    showWidget(node, modelMethodW);

    // 清理遗留的 SADA model label widget（兼容旧 workflow）
    const legacySadaLabel = node.widgets && node.widgets.find(w => w.name === "__sada_model_label__");
    if (legacySadaLabel) {
        node.widgets = node.widgets.filter(w => w.name !== "__sada_model_label__");
    }

    // 5. input slot: model 仅在 model_method !== "None" 时显示
    if (modelMethod !== "None") {
        ensureInput(node, "model", INPUT_TYPE_MAP["model"]);
    } else {
        ensureNoInput(node, "model");
    }

    // 6. output slots — 动态增删（与 input slot 策略一致）
    // 移除不需要的 output
    if (samplerMethod === "None") {
        ensureNoOutput(node, "sampler");
    }
    if (modelMethod === "None") {
        ensureNoOutput(node, "model");
    }

    // 添加需要的 output（按 RETURN_TYPES 顺序：sampler=0, model=1）
    if (samplerMethod !== "None") {
        ensureOutput(node, "sampler", "SAMPLER", 0);
    }
    if (modelMethod !== "None") {
        ensureOutput(node, "model", "MODEL", 1);
    }

    // 7. 重排 widget 顺序使分组正确
    reorderWidgets(node);

    // 8. 重新计算节点尺寸
    node.setSize(node.computeSize());
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "AccelDiff.Unified",

    async setup() {
        // Hook graphToPrompt 来修正动态 output slot 的索引映射
        // 因为 ComfyUI 前端用 link.origin_slot（数组位置索引）来序列化，
        // 但我们动态删除了 slot 导致位置索引与 RETURN_TYPES 不一致
        const origGraphToPrompt = app.graphToPrompt.bind(app);
        app.graphToPrompt = async function (...args) {
            // 在序列化之前，临时修正所有 AccelDiffUnified 节点的 output link 的 origin_slot
            const graph = args[0] || app.graph;
            const fixups = []; // 记录需要恢复的修改
            
            const nodes = graph && (graph._nodes || graph.nodes);
            if (nodes) {
                for (const node of nodes) {
                    if (node.type !== "AccelDiffUnified" || !node.outputs) continue;
                    for (let i = 0; i < node.outputs.length; i++) {
                        const slot = node.outputs[i];
                        // 如果 slot 有 slot_index 属性且与数组位置不同，需要修正所有 link
                        if (slot.slot_index !== undefined && slot.slot_index !== i && slot.links) {
                            for (const linkId of slot.links) {
                                const link = graph.links
                                    ? (graph.links instanceof Map
                                        ? graph.links.get(linkId)
                                        : graph.links[linkId])
                                    : null;
                                if (link && link.origin_slot !== slot.slot_index) {
                                    fixups.push({ link, origValue: link.origin_slot });
                                    link.origin_slot = slot.slot_index;
                                }
                            }
                        }
                    }
                }
            }
            
            try {
                return await origGraphToPrompt(...args);
            } finally {
                // 恢复 link 的 origin_slot（避免影响正常的图操作）
                for (const { link, origValue } of fixups) {
                    link.origin_slot = origValue;
                }
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AccelDiffUnified") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // 绑定 sampler_method 变更回调
            const samplerMethodW = node.widgets && node.widgets.find(w => w.name === "sampler_method");
            if (samplerMethodW) {
                const origCb = samplerMethodW.callback;
                samplerMethodW.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    updateNodeIO(node);
                };
            }

            // 绑定 model_method 变更回调
            const modelMethodW = node.widgets && node.widgets.find(w => w.name === "model_method");
            if (modelMethodW) {
                const origCb = modelMethodW.callback;
                modelMethodW.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    updateNodeIO(node);
                };
            }

            // 绑定 taylorseer_model_type 变更回调 — 切换模型时自动更新参数默认值
            const tsModelTypeW = node.widgets && node.widgets.find(w => w.name === "taylorseer_model_type");
            if (tsModelTypeW) {
                const origCb = tsModelTypeW.callback;
                tsModelTypeW.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    const defaults = TAYLORSEER_DEFAULTS[tsModelTypeW.value];
                    if (defaults) {
                        for (const [key, val] of Object.entries(defaults)) {
                            const w = node.widgets.find(x => x.name === key);
                            if (w) w.value = val;
                        }
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                };
            }

            // 绑定 hicache_model_type 变更回调
            const hcModelTypeW = node.widgets && node.widgets.find(w => w.name === "hicache_model_type");
            if (hcModelTypeW) {
                const origCb = hcModelTypeW.callback;
                hcModelTypeW.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    const defaults = HICACHE_DEFAULTS[hcModelTypeW.value];
                    if (defaults) {
                        for (const [key, val] of Object.entries(defaults)) {
                            const w = node.widgets.find(x => x.name === key);
                            if (w) w.value = val;
                        }
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                };
            }

            // 绑定 seacache_model_type 变更回调 — 切换模型时自动更新参数默认值
            const scModelTypeW = node.widgets && node.widgets.find(w => w.name === "seacache_model_type");
            if (scModelTypeW) {
                const origCb = scModelTypeW.callback;
                scModelTypeW.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    const defaults = SEACACHE_DEFAULTS[scModelTypeW.value];
                    if (defaults) {
                        for (const [key, val] of Object.entries(defaults)) {
                            const w = node.widgets.find(x => x.name === key);
                            if (w) w.value = val;
                        }
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                };
            }

            // 初次创建时同步 UI
            setTimeout(() => updateNodeIO(node), 0);
            return result;
        };

        // onConfigure: 加载工作流时恢复 UI 状态
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const node = this;
            setTimeout(() => updateNodeIO(node), 0);
        };
    },
});
