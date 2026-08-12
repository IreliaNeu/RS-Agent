# 第八阶段：多模态 RS-CC 增强、参考指标与小批量验证

## 本阶段目标

在不改变论文主基线的前提下，为 RS-CC 增加按模型能力选择输入的扩写方式，并完成一轮可复现的小批量端到端验证：

1. 论文轨继续只使用原始 Change-Agent 描述；
2. 增强轨允许部分 VLM 同时读取原始双时相图像和参考描述；
3. 保留 5 候选、Selector、Evaluator 和最高分优先的 LLM-as-Judge 逻辑；
4. 增加 LEVIR-MCI 参考描述指标及 Provider 稳定性统计；
5. 使用 10 个变化/无变化均衡样本验证 combined pipeline。

## 已完成内容

### 1. 明确三类实验协议

新增 `paper / operational / enhancement` 三类协议，并把协议、基线 ID、方法变体和是否替换论文模型写入配置与 artifacts。

- `paper`：论文方法复现，RS-CC 严格为纯文本输入；
- `operational`：使用当前可调用模型验证流水线；
- `enhancement`：允许论文之外的图文 RS-CC、证据审计等增强。

配置校验会拒绝在 paper 轨中使用 `image_text`，也会拒绝把图文 RS-CC 误标为 operational。

### 2. Capability-aware RS-CC

新增 `configs/rs_cc.adapted.yaml`，当前候选组合为：

- 2 个 `text_only` 模型：只接收原始参考描述；
- 3 个 `image_text` 模型：接收原始 A/B 图像和参考描述；
- Selector/Evaluator 使用文本候选进行独立选择和 1-10 评分。

图文提示明确把参考描述视为可能有误的模型输出，要求以图像为准纠正冲突，并忽略光照、色彩、阴影和季节外观伪变化。输入图像只在请求时编码，artifact 仅保存图像 SHA-256 和实际输入模式，不保存 Base64。

### 3. Provider 请求遥测

每次生成和 Judge 调用记录：

- 尝试次数；
- 总延迟；
- 最终 HTTP 状态；
- 每次 HTTP 状态和 transport/HTTP outcome；
- Prompt、Completion 和 Total Token（Provider 返回时）；
- 失败错误信息。

导出新增 `request_telemetry.csv`，`model_summary.csv` 增加成功率、重试数、均值/P95 延迟和 Token 汇总。

### 4. LEVIR-MCI 参考描述评估

从 `LevirCCcaptions.json` 生成了 1,929 条 test 参考记录，每条包括全部参考描述和官方 `change_flag`。新增指标：

- BLEU-1；
- 无平滑 sentence BLEU-4；
- best-reference ROUGE-L（beta=1.2）；
- 选中描述与官方 change flag 的一致率。

LLM-as-Judge 仍是主评价，以上指标只作为补充。导出记录参考清单 SHA-256、指标版本 `levir_mci_caption_metrics_v1.1` 和导出源码哈希，允许同一批不可变生成 artifacts 在指标修订后可审计地重新导出。

### 5. 可复现均衡抽样

新增 `scripts/select_balanced_pilot.py`，按官方 change flag 从现有 100 样本中确定性选择 5 个无变化和 5 个变化样本。

- Pilot 清单 SHA-256：`12af15dc01d3f2df969380c874da454826d6f9b04a63fb4cb2d1b0bb9a4aba85`
- Combined experiment fingerprint：`d088b88451495d705e069b16b9315df3f5e32da005ea21151769beac6e9fe83a`
- 任务：`RS-CC -> C* -> RS-VQA -> predicted mask -> Evidence Bundle`
- 样本并发：2

## 验证结果

### 自动化验证

- Pytest：`65 passed`
- Ruff：`All checks passed!`
- 配置解析、100 样本增强 RS-CC dry-run、Provider `/models` 预检均通过；
- OpenRouter 和 SiliconFlow 均返回 HTTP 200，适配配置中的模型均可见。

### 单样本真实 API smoke

`test_000001` 为官方无变化样本，5 个 RS-CC 候选全部成功。3 个图文模型分别误报了植被、季节或小结构变化，Judge 给出 1、3、1 分；两个文本模型给出 10、9 分，最终正确选择文本候选。

该 smoke 发现 DeepSeek-V3 在 SiliconFlow 上两次出现约 90 秒 transport timeout 后重试成功。因此 adapted 配置将 Judge 改为 Qwen3-30B，并收紧 timeout/retry；调整后 10 样本未出现 RS-CC Judge 重试。

### 10 样本 combined pilot

- Batch：10 completed，0 failed；
- RS-CC：50/50 候选成功；
- RS-VQA：50/50 候选成功；
- 请求账本：140 条；
- Caption 与 predicted mask 的 change/no-change 判断：10/10 一致；
- 选中 caption 的 change-flag accuracy：1.0；
- BLEU-1：0.3897；
- 无平滑 BLEU-4：0.0141；
- ROUGE-L：0.3190。

RS-CC 输入模式对比：

| 输入模式 | 候选数 | 平均 Judge 分 | 入选次数 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: |
| text_only | 20 | 8.45 | 8/10 | 2474 ms |
| image_text | 30 | 4.50 | 2/10 | 4518 ms |

这组结果只用于验证流程和形成下一步假设，不是论文规模结论。当前图文扩写并未普遍优于文本扩写，且更容易把季节外观当作变化，因此应作为独立 enhancement/ablation 保留，不能替代论文纯文本基线。

### Evidence conflict 审计

原始 pilot artifacts 报告 3 个 conflict：

- `test_000005` 和 `test_000006`：VQA 在官方无变化样本中报告了清地、新树或小建筑，与 caption 和 mask 冲突，属于真实伪变化；
- `test_000002`：VQA 明确回答 `no meaningful structural change`，旧 Evidence 正则因修饰词误判为 change，属于规则误报。

已扩展 no-change 正则并加入回归测试。为了保持实验身份严格性，没有用新源码静默覆盖旧 batch；文档同时保留原始告警数和审计后的实质冲突数 2。

## 当前项目状态

- 论文主流水线：约 94%。核心 Agent、Knowledge Bridge、VQA、mask 证据、Judge 和批处理已完整；尚需更大规模 paper-profile 对照实验。
- 可重复评估系统：约 93%。已具备实验协议、参考指标、请求遥测、源码/配置/输入身份、恢复、缓存和 checksum 导出。
- 多模态 RS-CC 增强：功能完成，初步结果不支持替代文本基线，需要更细的提示、模型与样本分层消融。
- 跨领域迁移：实验基础设施和 Provider 层可复用；遥感问题模板、图文提示、mask 类别和 Evidence 规则仍需领域适配。
- Web Demo：尚未开始，继续放在实验逻辑稳定之后。

## 目前存在的问题

1. 当前只验证 10 个均衡样本，不能据此判断模型总体优劣；
2. 图文模型对配准误差、季节和色彩变化敏感，需要专门的伪变化控制实验；
3. BLEU-4 为无平滑 sentence BLEU，短句容易得到 0；正式论文报告前需确认是否增加 corpus BLEU、CIDEr 或 SPICE；
4. 当前 change flag 是显式词表分类器，虽已修正本次错误，仍应在更大样本上审计；
5. Provider `/models` 可见不保证每次 completion 成功，必须继续保留实际请求遥测；
6. 已完成 batch 的 experiment fingerprint 对源码敏感，Evidence 规则变化后不能严格复用旧缓存，这是预期的可重复性约束；
7. 现有生成 artifacts 包含 Provider 原始响应，正式公开数据前需再次检查服务条款和隐私要求。

## 后续任务

1. 扩展到 50-100 个分层样本，分别运行 paper text-only、operational text-only 和 mixed image-text enhancement；
2. 对 no-change、building、road 和 mixed-change 分层报告 Judge 分数、参考指标、延迟、失败率和 Token；
3. 增加候选级参考指标及 bootstrap 置信区间；
4. 针对图文模型设计配准/季节伪变化提示消融，并评估是否需要变化区域裁剪或检索证据；
5. 基线冻结后再设计可选 synthesis/arbitration Agent 和长期 memory，且与论文结果独立报告；
6. 最后接入 Streamlit Web Demo，展示原始图像、5 候选、Judge 分数、C*、VQA、mask 和冲突证据。
