# 第三阶段进展：论文 RS-CC 流水线

日期：2026-08-08

## 当前完成了什么

1. 明确并实现 RS-CC 的输入边界：每个双时相图像对只接收一条由 Change-Agent 生成的原始变化描述，不向 CC-Agent 传递图像。
2. 按论文设置实现五个独立候选模型并发生成，标签固定为 A-E：Claude-Sonnet-4、DeepSeek V3、GPT-4o-mini、Qwen3-30B、LLaMA-4-Maverick。
3. 保留论文 `TextAgent-LLMasJudge.py` 的核心评估思路：GPT-4o 先选择最佳候选，再对全部候选进行 1-10 分评分。
4. 明确 C* 选择规则：优先选择评分最高的候选；最高分并列时采用 Judge Choice；仍无法区分时按 A-E 稳定顺序选择。
5. 将候选生成、评估和最终 C* 分别保存为不可变 JSON 产物，并保留模型原始响应、模型 ID、token usage、错误和校验哈希。
6. 实现 `rs-agent-cc` CLI，支持标准 JSONL、旧版 `Original Caption` 字段、单条描述运行以及不需要 API key 的 `--dry-run`。
7. 增加论文 OpenRouter 配置 `configs/rs_cc.paper.yaml`，Selector 与 Evaluator 即使使用相同 GPT-4o，也保持独立角色配置。
8. 在服务器 `rs-agent` Conda 环境中完成验证：Ruff 通过、26 个测试全部通过、`pip check` 无依赖冲突、CLI dry-run 通过。
9. 下载并校验 LEVIR-MCI：压缩包 2,771,943,190 字节，SHA-256 为 `8f7d52298fa3ca32aec983e84addef80f61ad7d981c0ff97a92c5880c4b83e9d`，ZIP 完整性检查通过；解压后约 2.8 GB，共 40,311 个文件。

## 目前存在哪些问题

1. `.env` 中还没有有效的 `OPENROUTER_API_KEY`，所以尚未进行真实模型联调和费用/限流测试。
2. LEVIR-MCI 提供的是每对图像五条人工 ground truth；正式 RS-CC 需要的是 Change-Agent 每对图像生成的一条模型描述。目前还缺少这份一对一输入 JSONL。
3. OpenRouter 模型 ID 已按当前可用名称配置，但正式实验前仍需检查账户权限、模型是否下线、上下文限制和单价。
4. 当前为严格论文复现模式，五个候选必须全部成功；真实 API 联调后可以再增加一个非论文的容错运行配置，但不能与正式实验结果混用。
5. 论文叙述与旧代码在部分提示词和 Agent 输入上存在偏差。本阶段按用户确认的真实实验逻辑实现：CC 只接收原始文本；后续需要在实验记录中明确版本号。

## 后续需要实现什么

1. 运行迁移后的 Change-Agent 模块，或导入已有推理结果，生成 LEVIR-MCI 一对一原始描述 JSONL。
2. 用户填写 OpenRouter key 后，先对 1-5 个样本进行真实 API 冒烟测试，确认五模型返回格式、GPT-4o 评分解析、限流和成本。
3. 实现与 RS-CC 不同模型组的 RS-VQA：MiniMax-01、GPT-4o-mini、LLaMA-4-Maverick、Qwen2.5-VL-32B-Instruct、Mistral-small-3.2-24B-Instruct，并严格限制为原始双时相图像输入。
4. 重新设计遥感变化预设问题模板，并实现用户问题、模板问题和混合问题的规范化。
5. 建立 Knowledge Bridge，只把最终 C* 传递给 VQA/Main Agent；随后加入可选 mask 证据。
6. 完成端到端实验和评估后，再基于 `Multi_change/web_demo.py` 实现网页交互演示。

## 需要用户后续确认

正式联调前需要用户把 `OPENROUTER_API_KEY` 写入 `/root/autodl-tmp/RS-Agent/.env`。此外需要确认服务器或原项目中是否已有 Change-Agent 对 LEVIR-MCI 的逐图像对生成结果；如果没有，下一阶段将从模型权重与推理入口开始生成。
