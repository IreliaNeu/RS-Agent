# 第二阶段：Pipeline 基础设施

## 当前完成内容

1. 创建公开 GitHub 仓库 `IreliaNeu/RS-Agent`。
2. 定义论文流程所需的类型化数据结构，包括任务输入、双时相图像、候选描述、
   模型响应、候选评分、描述选择、VQA 问题与答案、Pipeline 输出。
3. 为 Caption Enrichment 增加输入约束：必须提供双时相图像和一条图像对原始描述。
4. 实现基于 YAML 的 Provider 与模型配置，并通过环境变量读取 API key。
5. 实现异步 OpenAI-compatible Provider，可用于 OpenRouter、SiliconFlow 等接口。
6. Provider 支持并发限制、超时、可重试状态码和原始响应记录；非重试型 4xx
   不会重复调用，避免产生额外请求和费用。
7. 实现只写一次的 JSON Artifact Store，包含 SHA-256 完整性校验，防止实验产物
   被静默覆盖或修改。
8. 将 `TextAgent-LLMasJudge.py` 的最佳候选判断和逐候选 1-10 分评分逻辑迁移到
   新评估模块。
9. 保留旧版 JSONL 字段，并增加选择策略与模型响应 provenance。
10. `C*` 使用最高评分候选；评分并列时使用 Judge Choice 破平，再并列时按固定
    标签顺序选择，从而保证实验可复现。
11. Selector 与 Evaluator 使用独立配置，论文复现时允许指向同一模型。
12. 在服务器 Python 3.9 Conda 环境中完成测试：16 项测试全部通过，Ruff 检查
    通过，`pip check` 无依赖冲突。
13. 创建服务器 `.env` 占位文件，真实 key 不进入 Git 或配置 YAML。

## 当前存在的问题

1. 尚未使用真实 OpenRouter API 联调，当前 Provider 测试采用 MockTransport。
2. 尚未实现 RS-CC 的五模型并行候选生成，目前只完成候选与评估基础设施。
3. 尚未实现双时相图像的多模态消息编码和图片大小控制。
4. 尚未实现预设 RS-VQA 问题模板、Knowledge Bridge 和完整 Main-Agent 工作流。
5. MCI 的 PyTorch、MMCV 和 MMSEG GPU 环境尚未安装验证。
6. LLM-as-Judge 的评分稳定性仍需通过重复评估和人工样本校准。
7. 服务器通过 AutoDL 加速访问 GitHub 仍可能出现 503 或超时。

## 后续需要实现

1. 实现双时相图像编码与 OpenRouter VLM 请求格式。
2. 实现 RS-CC Caption Enrichment：原始图像对和一条 Change-Agent 描述同时输入
   五个不同候选模型，并行生成五条增强描述。
3. 将候选、原始响应、Judge 结果和 `C*` 写入不可变 Artifact Store。
4. 使用少量真实样本完成 OpenRouter 联调，并核对 API 返回差异与费用信息。
5. 设计 RS-VQA 预设问题集与用户问题归一化规则。
6. 实现 Knowledge Bridge 开关和论文消融配置。
7. 在核心 API 流程稳定后安装并验证可选 MCI Mask 环境。

## 需要用户提供的内容

开始真实 API 联调前，请按照 `docs/API配置说明.md` 填写服务器 `.env`。当前阶段
不需要提供 key。首轮联调模型 ID 可以在填写 key 时一并确认。

