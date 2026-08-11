# 第六阶段：Main Agent 调度、可选 Mask 证据与冲突审计

## 本阶段目标

本阶段在第五阶段固定的 `RS-CC -> C* -> RS-VQA` 流程上，补齐论文中 Main Agent 的任务调度语义和可选变化掩膜 `M~` 的工程契约。同时保持以下边界：

1. 论文中的 Main Agent 负责理解任务并调度子 Agent，论文没有定义额外的 LLM 最终融合裁决器。
2. Knowledge Bridge 只向 RS-VQA 传递最终选中的 `C*`。
3. VLM 只接收原始双时相图像；mask 不进入 VLM 消息。
4. mask 是可选证据，不自动覆盖 caption 或 VQA 结论。
5. 论文基线和新增工程增强必须在产物中可以区分。

## 已完成内容

### 1. 可审计的 Main Agent 计划器

新增 `src/rs_agent/orchestration/main_agent.py`，根据显式任务类型生成并保存执行计划：

| 请求类型 | 实际执行阶段 |
| --- | --- |
| `caption` | `RS-CC` |
| `vqa` 且启用 KB | `RS-CC -> Knowledge Bridge -> RS-VQA` |
| `vqa` 且关闭 KB | `RS-VQA` |
| `combined` | `RS-CC -> Knowledge Bridge -> RS-VQA` |
| 任意任务附带 mask | 在上述阶段外增加 `mask_evidence` |

每次运行新增不可变 `main_agent_plan` 产物，记录任务类型、问题数量、是否启用 KB、是否提供 mask 和调度理由。原先每次固定运行完整链路的问题已经消除，论文的 KB 消融现在是真实跳过 RS-CC/KB 的执行路径。

当前 Main Agent 使用显式 `--task-type`，还没有使用另一个 LLM 从自由文本中推断任务。这样更适合作为可复现实验基线；网页阶段可以在其上增加意图解析层。

### 2. Change-Agent 同次前向输出预测 Mask

确认 `Multi_change/model/model_encoder_att.py` 中 `AttentiveEncoder.forward` 同时返回 caption 特征和三分类变化检测 logits，类别为 background、road 和 building。原批处理脚本过去丢弃了第三个返回值。

现已扩展 `scripts/generate_change_agent_captions.py`：

- 新增 `--mask-output-dir`。
- 对 logits 执行 `argmax` 并保存单通道类别索引 PNG。
- 每条 JSONL 增加 `predicted_mask` 路径。
- manifest 记录 mask 输出目录。
- 不需要新增模型，也不需要再次单独执行 CD 网络。

使用现有 `MCI_model.pth` 成功重新生成 LEVIR-MCI test split 前 100 个样本：

- Caption/Mask JSONL：`/root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100_with_masks.jsonl`
- Mask 目录：`/root/autodl-tmp/rs-agent-data/change-agent/masks/test-100`
- 样本数：100
- 空 caption：0
- GPU：NVIDIA GeForce RTX 4090
- 模型推理耗时：11.812 秒

### 3. 可选 Mask 结构化证据

新增 `src/rs_agent/domains/remote_sensing/mask_evidence.py`。输入必须是单通道类别索引图，输出包括：

- 文件 SHA-256、尺寸和来源类型；
- 每个类别的像素数与占比；
- 非背景变化像素数与占比；
- 四连通区域数量；
- 满足阈值的显著区域数量；
- 最大区域像素数、占比和边界框；
- 保守的 `has_change` 标记。

来源明确区分 `predicted`、`ground_truth` 和 `external`。Ground truth 只能用于评估，不应伪装成推理证据。

### 4. 领域无关 Evidence Bundle

新增 `src/rs_agent/orchestration/evidence.py`，为以下证据分配稳定来源 ID：

- Change-Agent 原始 caption；
- RS-CC 选中的 `C*`；
- 每个选中的 RS-VQA 回答；
- 可选 mask 摘要。

系统使用保守规则标注 `change`、`no_change` 或 `uncertain`，并记录 change/no-change 来源对。该模块是工程增强层，产物 metadata 标记为 `engineering_enhancement`。它只报告冲突和保守共识，不替代论文 Judge，不生成未经论文定义的新最终分数。

### 5. CLI 与产物更新

`rs-agent-run` 新增：

- `--task-type caption|vqa|combined`
- `--without-knowledge-bridge`
- `--mask`
- `--mask-source predicted|ground_truth|external`
- `--mask-min-component-pixels`
- `--mask-min-changed-ratio`

最终结果现在关联 `main_agent_plan`、`rs_cc_result`、`rs_vqa_result`、`mask_evidence` 和 `evidence_bundle`。未执行阶段明确为 `null`，便于批量统计真实调用路径。

## 验证结果

### 自动化验证

- 完整测试：`46 passed`
- Ruff：`All checks passed`
- 无 KB 的 VQA dry-run 只规划 `rs_vqa`。
- 单元测试覆盖全黑 mask、多类别 mask、连通区域、RGB 非法输入、三种任务路由和 caption-mask 冲突。

### 三个真实预测 Mask

- `test_000001`：全背景，0 个变化像素，与 no-change caption 一致。
- `test_000044`：495 个 building-change 像素，占比约 0.755%，最大区域边界框 `[225, 170, 255, 200]`，与 building caption 一致。
- `test_000068`：全背景，但 Change-Agent caption 为新道路出现，形成真实的跨证据冲突。

### 100 样本 Caption-Mask 一致性诊断

该统计只比较 Change-Agent 同一模型的 caption 输出与 mask 输出，不是相对 ground truth 的准确率：

| Caption | Mask | 数量 |
| --- | --- | ---: |
| change | change | 33 |
| change | no-change | 3 |
| no-change | change | 7 |
| no-change | no-change | 57 |

总计 90 个一致、10 个不一致。不一致结果表明 mask 不能被当作自动否决文本结论的真值，也说明后续应保留跨证据一致性指标。

### 真实 Pipeline 验证

对 `test_000068` 执行 `caption-only + predicted mask`：

- Main Agent 计划仅包含 `rs_cc` 和 `mask_evidence`，没有调用 RS-VQA。
- RS-CC 五个候选均成功，Judge 选择候选 B。
- `C*` 为新道路在中央未开发区域出现。
- mask 为全背景。
- Evidence Bundle 记录两条 change/no-change 冲突，`consensus=uncertain`。
- 产物目录：`/root/autodl-tmp/rs-agent-artifacts/phase6-route-mask/phase6-route-mask-000068/`

## 当前项目状态

按“论文实验优先”的目标估计：

- 论文方法主干：约 85%。RS-CC、Judge、C*、KB、RS-VQA、任务调度和可选 `M~` 已具有统一入口和产物契约。
- 可重复批量实验系统：约 60%。单样本链路稳定，尚缺批量恢复、缓存、正式导出和 provider preflight。
- 可迁移跨领域 Agent 平台：约 45%。Provider、artifact、Main Agent plan 和 Evidence Bundle 已领域无关；问题模板、mask 类别和具体 Agent 仍在遥感域模块中。
- Web Demo：尚未进入正式实现，符合“最后进行”的既定顺序。

当前框架是静态、类型化、可审计的 Agent 工作流，不依赖 Lagent 或 LangGraph 的运行时。现阶段没有必要迁移框架：流程分支数量有限，Python 编排更容易复现实验。未来出现循环反思、人工中断恢复或大量工具调用时，再评估 LangGraph。

记忆方面，目前保留运行级 artifact memory，不启用跨样本语义记忆。论文正式实验应保持样本隔离，避免历史样本污染评价。跨样本检索记忆只能作为后续独立消融。

## 目前存在的问题

1. Evidence stance 使用保守词表，只适合冲突预警，不能作为论文定量评价或最终语义裁决。
2. Mask 阈值当前默认为最宽松设置，需要在验证集上确定最小区域和最小占比。
3. `caption_from_scratch` 在当前 LEVIR-MCI 配置中未开放；项目仍遵循“一对图像一条 Change-Agent 原始描述”的要求。
4. 自由文本任务意图解析尚未实现，当前由 CLI/Web 上层明确传入 task type。
5. OpenRouter 的 OpenAI、Google、Anthropic 模型仍受服务器出口限制；论文 profile 尚未在服务器全量复现。
6. 还缺面向论文表格的批量结果聚合、失败恢复、缓存、provider 健康检查和评估导出。
7. 还没有额外的最终答案综合 Agent。论文没有给出该 Agent 的实验定义，因此应在冻结论文基线后作为独立增强实现。

## 后续任务

1. 实现批处理 runner：JSONL 输入、并发限制、断点续跑、缓存与失败清单。
2. 保留现有 LLM-as-Judge 逻辑，增加 caption/VQA/mask 一致性与 no-change 误报诊断导出。
3. 增加 provider preflight，启动实验前报告模型可达性、配置身份和缺失密钥，不静默替换论文模型。
4. 冻结论文 baseline 配置和结果 schema，区分 paper、operational smoke 与 enhancement 三种运行身份。
5. 基线稳定后再设计可选 synthesis/arbitration Agent，并单独做消融。
6. 最后将统一入口接入 `Multi_change/web_demo.py` 的网页交互。
