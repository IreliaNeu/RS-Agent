# 第九阶段：无 GPU 条件下的候选级评估与成对 RS-CC 消融

## 本阶段目标

服务器临时切换为无卡模式后，本阶段先划分任务的资源边界，再继续完成不依赖 GPU 的工作：

1. 增加候选描述级参考指标，而不只评估最终选中的 `C*`；
2. 为选中描述和候选模型增加可复现的 Bootstrap 置信区间；
3. 设计同样本、同模型集合的 RS-CC 纯文本与混合图文成对消融；
4. 在 20 个 LEVIR-MCI 平衡样本上完成 API 推理、离线导出和成对比较；
5. 明确当前证据能够支持的结论、实验限制和 GPU 恢复后的任务。

## 服务器资源判断

本阶段实际读取到的容器限制为：

| 资源 | 实际限制 |
| --- | --- |
| CPU | `cpu.max = 50000 100000`，约 0.5 核 |
| 内存 | `memory.max = 2147483648`，即 2 GB |
| GPU | 不存在 `/dev/nvidia0`，无可用 GPU |

### 本阶段可以继续的工作

- Python 单元测试、Ruff 和配置检查；
- 调用 OpenRouter、SiliconFlow 的 API 模型；
- 单样本串行的批处理调度；
- BLEU、ROUGE-L、变化标志和 Bootstrap 离线统计；
- 实验导出、文档、Git 和备份。

### 暂时搁置的 GPU 工作

- 使用 Change-Agent MCI checkpoint 重新生成 100 条原始描述；
- 重新生成三类变化 mask；
- 任何本地 VLM/LLM 推理或 CUDA 模型实验；
- 基于局部变化区域裁剪的视觉模型实验。

API RS-CC 本身不需要本地 GPU。混合图文组只在本地读取双时相图像并编码请求，实际 VLM 推理由远端服务完成。为适应半核 CPU，本阶段使用批级并发 `1`，并顺序执行两组实验。

## 已完成的实现

### 1. 候选级参考指标

新增 `src/rs_agent/evaluation/candidate_metrics.py`。导出现在会保留每个 RS-CC 候选的：

- BLEU-1；
- 无平滑 sentence BLEU-4；
- best-reference ROUGE-L；
- 与 LEVIR-MCI 官方 change flag 的一致性；
- 模型、Provider、model ID、实际输入模式、Judge 分数和是否入选；
- 生成失败记录。失败候选不会被错误地当作 0 分样本。

新增两个导出文件：

- `caption_candidate_reference_metrics.csv`：逐样本、逐候选指标；
- `caption_candidate_summary.csv`：按输入模式和模型汇总的均值、95% 区间与入选次数。

### 2. 可复现 Bootstrap

新增 `src/rs_agent/evaluation/bootstrap.py`，使用无额外依赖的 percentile bootstrap。默认参数为：

- 重采样次数：2,000；
- 置信度：95%；
- 随机种子：`20260820`。

`rs-agent-export` 新增 `--bootstrap-samples`、`--bootstrap-seed` 和 `--confidence` 参数。选中描述及候选模型的指标均可输出区间。

### 3. 成对消融比较

新增正式模块 `src/rs_agent/evaluation/ablation.py` 和 CLI 脚本 `scripts/compare_rs_cc_ablation.py`。比较器会：

- 检查两组导出使用同一参考清单；
- 要求样本 ID 集合完全一致；
- 逐样本计算“增强组减基线组”的指标差；
- 对均值差执行 paired percentile bootstrap；
- 输出 `summary.json` 和 `paired_metrics.csv`。

新增配置：

- `configs/rs_cc.ablation_text_only.yaml`：五个模型全部只读参考文本；
- `configs/rs_cc.ablation_mixed.yaml`：DeepSeek-V3、Qwen3-30B 只读文本，Mistral Small 3.2、LLaMA 4 Maverick、Qwen3-VL-30B 读取原始图像对和参考文本。

两组使用完全相同的五个生成模型、Selector、Evaluator 和生成参数，仅三个候选模型的输入模式不同。两组均明确标记为 `enhancement`，不能作为论文原模型复现结果。

## 实验设置与身份

从现有 Change-Agent 100 条清单中确定性选取 10 个无变化和 10 个变化样本：

| 项目 | 值 |
| --- | --- |
| 样本清单 | `levir_mci_balanced_20.jsonl` |
| 清单 SHA-256 | `3bc4b8f70f27b3273e6eb8db2c6a8ec1906680adf232a4facd54d1d28ae3b389` |
| 参考清单 SHA-256 | `eb2dcc7f04f4909e15c93f01c8c13b3107d1668d8b062a615e516c52491dac7a` |
| 纯文本实验 fingerprint | `4704a6899de3195b144925325362933a9d03e051ef30b1c7e02f9cbfd0f1d401` |
| 混合图文实验 fingerprint | `bf92e4c55ba3b80289a7d8d3953401b16d77a21ff072dec18705ed31c3e0b244` |
| Bootstrap | 2,000 次，95%，seed `20260820` |
| Mask | 不使用 |
| 批级并发 | 1 |

网络预检中 OpenRouter 和 SiliconFlow 均返回 HTTP 200；五个生成模型和 Qwen3-30B Judge 均可见。两组都完成 20/20 样本、100/100 候选和 140/140 次请求，没有生成失败、请求失败或重试。纯文本组约用时 119 秒，混合图文组约用时 208 秒。

## 实验结果

### 1. 最终选中描述

| 指标 | 全文本 | 混合图文 | 成对差值（混合减文本） | 差值 95% CI |
| --- | ---: | ---: | ---: | ---: |
| BLEU-1 | 0.4073 | 0.3571 | -0.0502 | [-0.0892, -0.0098] |
| BLEU-4 | 0.0093 | 0.0257 | +0.0165 | [-0.0083, 0.0578] |
| ROUGE-L | 0.3352 | 0.3097 | -0.0255 | [-0.0623, 0.0194] |
| change-flag accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| Judge 分数 | 9.1500 | 8.8500 | -0.3000 | [-0.8000, 0.1000] |
| 选中请求延迟 | 1925 ms | 2410 ms | +484 ms | [-229, 1210] |

在这 20 个样本中，只有 BLEU-1 的差值区间没有跨越 0，且方向有利于纯文本组。BLEU-4 略有上升，但区间较宽并跨 0；ROUGE-L、Judge 分数和延迟差也不能排除 0。两组均正确保留了 10 个变化和 10 个无变化样本的变化标志。

### 2. 候选模型观察

| 模型 | 全文本 BLEU-1 / 入选 | 混合组输入 | 混合组 BLEU-1 / 入选 |
| --- | ---: | --- | ---: |
| DeepSeek-V3 | 0.4042 / 17 | text_only | 0.3916 / 13 |
| Qwen3-30B | 0.4320 / 2 | text_only | 0.4165 / 0 |
| Mistral Small 3.2 | 0.3699 / 1 | image_text | 0.2306 / 4 |
| LLaMA 4 Maverick | 0.4153 / 0 | image_text | 0.4549 / 0 |
| Qwen3-VL-30B | 0.4666 / 0 | image_text | 0.2822 / 3 |

图像输入不是统一的正收益：LLaMA 4 Maverick 的 BLEU-1 上升，而 Mistral Small 3.2 和 Qwen3-VL-30B 明显下降。混合组中三个图文模型的平均请求延迟约为 3.3-4.5 秒，高于多数文本请求的 1.5-2.5 秒。Judge 的入选次数与参考指标并不总一致，说明后续需要单独校准 Judge 偏好和图文扩写提示。

## 结果解释边界

本次结果是流程验证和假设生成，不是论文规模结论，原因包括：

1. 样本只有 20 个，虽然变化/无变化平衡，但尚未按 road、building、mixed change 等类型分层；
2. 两组按同样本成对比较，但外部 API 即使 temperature 为 0 也可能存在非确定性；两组中的两个公共文本候选是重新调用而非复用完全相同的响应；
3. 当前 Selector 和 Evaluator 都使用 Qwen3-30B，可能存在模型家族偏好；
4. 无平滑 sentence BLEU-4 对短描述非常敏感；
5. 图文模型可能生成更多可见细节，但参考描述通常较短，词面指标可能惩罚合理扩写；反过来，图文模型也确实可能把季节、颜色和阴影差异误认为变化；
6. 当前比较的是最终 `C*` 的系统效果，不等同于对单个模型输入模式的严格因果估计。

下一轮严格消融应增加“候选重放/冻结”能力：复用相同的公共文本候选，只重新生成被切换为图文输入的三个候选，从而隔离 API 随机性和共享候选漂移。

## 自动化验证

- Pytest：`73 passed`；
- Ruff：`src/`、`scripts/`、`tests/` 全部通过；
- `legacy/` 保留上游迁移代码，不纳入当前维护范围的 Ruff；
- 两份配置 dry-run：均为 20 条且身份独立；
- Provider 网络预检：两家 Provider HTTP 200，目标模型均可见；
- 两个真实批次：均 20 completed、0 failed；
- 两次导出和一次成对比较：均完成 checksum 校验。

## 当前项目状态

- 论文主流水线：约 95%。RS-CC、`C*`、Knowledge Bridge、RS-VQA、可选 mask、Evidence Bundle 和 Main Agent 已贯通；仍需更大规模 paper-profile 对照。
- 可重复评估系统：约 96%。候选账本、参考指标、请求遥测、Bootstrap、成对比较、身份、恢复、缓存和 checksum 导出已具备。
- 图文 RS-CC 增强：功能完整，但当前证据不支持替换纯文本基线；应按模型能力选择并继续做提示消融。
- GPU 数据生成：原有 100 条数据可继续用于流程实验，重新推理已延期到 GPU 可用时。
- Web Demo：尚未开始，继续放在实验逻辑冻结之后。

## 当前问题

1. 缺少冻结公共候选的严格输入模式消融机制；
2. 样本规模和变化类型分层不足；
3. Judge 与参考指标存在偏好差异，需要独立 Judge 或多 Judge 稳健性实验；
4. 图文提示对伪变化和过度扩写的约束仍不充分；
5. 正式论文指标还应评估 corpus BLEU、CIDEr 等与原论文一致的实现；
6. Change-Agent 原始描述和 mask 的重新生成需要 GPU；
7. 当前 API 模型是可用模型替代组合，不能冒充论文中的原模型组合。

## 后续任务

1. 增加候选重放/冻结机制，完成严格控制的 RS-CC 输入模式消融；
2. 扩展到 50-100 个分层样本，报告变化类型、Judge、参考指标、延迟、Token 和失败率；
3. 设计图文模型的伪变化抑制提示，并评估按模型启用图像输入，而不是统一启用；
4. GPU 恢复后重新生成一批 Change-Agent 原始描述和 mask，并固定数据版本；
5. 冻结论文基线后，再单独设计 synthesis/arbitration Agent 与长期 memory，不混入论文基线；
6. 最后接入 Streamlit Web Demo，展示原始图像、五候选、Judge、`C*`、VQA、mask 与冲突证据。
