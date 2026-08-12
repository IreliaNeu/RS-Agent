# 第七阶段：可恢复批处理、Provider 预检与评估导出

## 本阶段目标

在第六阶段稳定的单样本 Agent 流程上，补齐面向论文实验的运行基础设施：

1. 以 JSONL 批量运行 `caption`、`vqa` 或 `combined` 任务。
2. 支持受控并发、中断恢复、失败重试和可选跨批缓存。
3. 在调用 API 前检查密钥、端点和配置模型可见性。
4. 从不可变 artifacts 导出逐样本结果、完整候选账本、Judge 分数和模型汇总。
5. 保持论文 Judge 逻辑不变，不引入跨样本语义记忆。

## 已完成内容

### 1. 内容派生的实验身份

新增 `src/rs_agent/experiments/identity.py`。每个 batch 的 fingerprint 由以下内容共同决定：

- 输入 JSONL 的 SHA-256；
- 每个样本的 caption、问题和 metadata；
- 原始 A/B 图像及可选 mask 的内容 SHA-256；
- RS-CC 和 RS-VQA 配置文件 SHA-256；
- `rs_agent` Python 源码树 SHA-256；
- Python、httpx、Pillow、Pydantic 和 PyYAML 版本；
- task type、Knowledge Bridge、mask 阈值等运行选项。

样本 fingerprint 不包含绝对路径，因此数据目录迁移后仍能识别相同内容。任一图像、问题、配置、源码、关键依赖或选项改变都会得到新实验身份，避免不同实验结果混用。

### 2. 可恢复批处理 Runner

新增 `src/rs_agent/experiments/batch.py`、`state.py` 和 `cache.py`，以及 `rs-agent-batch` CLI。

能力包括：

- JSONL 输入验证和相对路径解析；
- `--concurrency` 受控样本并发；
- 每个样本独立的 `pending/running/completed/failed` 状态；
- 原子更新 `state.json`，运行状态与 write-once 科学 artifacts 分离；
- 启动时将遗留 `running` 状态恢复为 `pending`；
- 对 completed result 重新校验 checksum、artifact type 和 item ID；
- 相同 batch ID 仅在 experiment fingerprint 完全一致时允许恢复；
- `--max-items` 用于小批试跑和人工构造 partial 状态；
- `--retry-failed` 只重试失败项；
- 每次尝试使用独立 run ID，不覆盖旧失败 artifacts。

跨批缓存默认关闭，只有传入 `--cache-dir` 才启用。命中条件包含样本内容、配置、源码、运行时版本和运行选项，并再次校验 result artifact 的 checksum、type、item ID 和 run ID。缓存状态使用独立 `cache_hit` 字段，不与错误字段混用。

### 3. Provider Preflight

新增 `src/rs_agent/providers/preflight.py` 和 `rs-agent-preflight` CLI。

两种检查模式：

- 默认模式：解析 RS-CC/RS-VQA 配置并确认所需环境变量存在，不联网。
- `--network`：携带认证头访问每个 Provider 的 `/models`，报告 HTTP 状态和配置模型是否在目录中可见。

preflight 输出只包含环境变量名，不包含 key 值，也不发送推理 prompt。RS-CC 与 RS-VQA 对同一 endpoint 可以配置不同 timeout/retry/concurrency；只要 base URL、凭据环境变量和认证头一致，就会合并检查。

### 4. Checksum 校验评估导出

新增 `src/rs_agent/evaluation/batch_export.py` 和 `rs-agent-export` CLI。导出前会校验最终结果及其引用的 generation/evaluation artifacts。

输出包括：

- `summary.json`：批次状态、失败数、冲突数、caption-mask 一致性和候选生成失败数；
- `experiment_identity.json`：完整实验身份；
- `items.jsonl`：选中 caption、VQA 答案、mask、Evidence Bundle 结论与来源；
- `failures.jsonl`：未完成或失败的样本；
- `caption_scores.csv`：全部 RS-CC 配置模型的生成状态、错误、Judge 分数和选择结果；
- `vqa_scores.csv`：全部 RS-VQA 配置模型的生成状态、错误、Judge 分数和选择结果；
- `model_summary.csv`：每模型候选数、成功/失败数、有效分数数、均值、总体标准差、选中次数和 Judge choice 次数。

失败模型不会从表中消失，也不会被计为 0 分；其 score 与 std 保持为空，错误信息保留。这可以避免 minimum-successful 阈值造成幸存者偏差。

### 5. Evidence 否定短语修正

真实 smoke run 发现 `no discernible changes` 被旧词表先漏过否定匹配，再由通用 `changes` 误标为 change。现已扩展 no-change 模式，支持 discernible、meaningful、notable、visible、apparent、major 等限定词及 `without changes`，并增加回归测试。

该修改只影响工程增强层的 Evidence Bundle 冲突预警，不改变论文 Judge 提示词、分数或选优结果。

## 验证结果

### 自动化验证

- 完整测试：`58 passed`
- Ruff：`All checks passed`
- 覆盖实验身份、路径迁移、状态恢复、失败重试、源码感知缓存、缓存完整性、Provider 预检、artifact 篡改检测、失败候选导出和 evidence 否定短语。

### Provider Preflight

在当前 AutoDL 服务器执行 operational profiles：

- credential-only：OpenRouter 与 SiliconFlow 均通过；
- `/models`：两个端点均返回 HTTP 200；
- operational profiles 中 9 个去重模型 ID 全部可见。

这只证明认证、路由和模型目录正常，不保证每个模型每次推理成功。

### 100 样本 Dry-run

对 `/root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100_with_masks.jsonl` 完成：

- 100 个样本及图像/mask 均通过内容验证；
- RS-CC/RS-VQA 配置均通过解析；
- 最终 dry-run fingerprint：`d04ef57ab3e6c45dda80b7c4f815672a662475afaf5d7c82fd76b43a713e82a0`。

dry-run 不调用 API，也不创建 batch state。

### 断点续跑

使用 `test_000001` 和 `test_000044` 进行 caption-only 批处理：

1. 第一次使用 `--max-items 1`，结果为 `partial`：1 completed、1 pending。
2. 第二次使用相同 batch ID，结果为 `completed`：2 completed、0 failed。
3. 两个样本的 attempts 均为 1，已完成样本未重复执行。

### Corrected Caption Smoke

修正 evidence 否定短语与源码感知缓存后，新 batch `phase7-caption-corrected` 结果：

- 2/2 completed；
- 10 条 RS-CC 候选分数；
- 1 个 no-change consensus、1 个 change consensus；
- 1 个 caption-no-change/mask-no-change；
- 1 个 caption-change/mask-change；
- 0 个冲突项。

### Combined Smoke 与失败可见性

对 `test_000044` 运行 `RS-CC -> C* -> RS-VQA -> mask -> evidence`：

- batch completed，1/1 样本成功；
- 5 条 RS-CC 候选全部成功；
- 5 条 RS-VQA 配置候选中 4 条成功；
- `Mistral-Small-3.1` 经 3 次尝试后收到 OpenRouter 上游 HTTP 500；
- 流程按 operational profile 的 `minimum_successful_answers_per_question=4` 正常完成；
- exporter 在 `vqa_scores.csv` 保留失败模型，并在汇总中报告 `candidate=1/success=0/failure=1/score_count=0`；
- Evidence consensus 为 change，caption 与 building-change mask 一致，无冲突。

真实 combined 批次耗时约 139 秒，说明后续较大实验必须使用 batch state、并发限制和失败重试，而不应依赖一次性脚本。

### 最终缓存验证

最终版本用单个 caption 样本建立缓存后，以新 batch ID 运行相同实验：

- experiment fingerprint 完全一致；
- `cache_hit=true`；
- `attempts=0`；
- `error=null`；
- result artifact 与原始 run ID 均可追溯。

## 当前项目状态

按“论文实验优先”目标估计：

- 论文方法主干：约 88%。核心流程、任务调度、Knowledge Bridge、mask 契约与批量运行入口已稳定。
- 可重复批量实验系统：约 82%。具备身份锁定、恢复、重试、缓存、预检、完整候选导出和校验；尚缺正式大样本 paper-profile pilot 与数据集参考指标。
- 可迁移跨领域 Agent 平台：约 55%。实验 identity/state/cache/exporter 均为领域无关；具体问题模板与 mask 类别仍属于遥感域。
- Web Demo：尚未正式实现，继续保持最后阶段处理。

当前仍不需要迁移到 Lagent 或 LangGraph。批处理恢复是样本级状态机，不需要图运行时；现有纯 Python 编排更容易复现实验。跨样本语义记忆继续关闭，缓存只复用完全相同实验的已完成输出，不向模型提供历史样本内容。

## 目前存在的问题

1. operational combined smoke 中 Mistral-Small-3.1 出现 OpenRouter 上游 500，需在更大 pilot 中观察失败率。
2. `/models` 可见不等于 chat completions 一定可用，paper profile 仍需逐模型 pilot。
3. 当前 batch concurrency 是样本级，单样本内部模型并发仍由各 Provider 的 `max_concurrency` 控制；大样本运行前需要按 API 限额设置。
4. Exporter 已输出 Judge 分数和一致性统计，但还没有 BLEU/CIDEr、embedding similarity 或基于参考标注的任务指标。
5. Evidence stance 仍是工程预警词表，不应作为论文定量结论。
6. 旧 smoke artifacts 在代码修正后仍保留在持久盘；正式报告必须使用 experiment identity 区分，不能混合。
7. 尚未实现自动将失败清单分 provider/model 聚合成稳定性报告。

## 后续任务

1. 选取 10 至 20 个 LEVIR-MCI 样本进行 paper-profile 逐模型 pilot，记录模型可用性、延迟和失败率。
2. 明确并冻结 `paper / operational / enhancement` 运行身份字段和 baseline schema。
3. 基于 LEVIR-MCI 可用参考 caption/mask 增加数据集级指标，同时保留现有 LLM-as-Judge 特色。
4. 增加 provider/model 稳定性、延迟和 token 用量聚合。
5. baseline 冻结后，再设计可选 synthesis/arbitration Agent 及独立消融。
6. 最后接入 Streamlit Web Demo。
