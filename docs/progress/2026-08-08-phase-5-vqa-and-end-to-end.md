# 第五阶段进展：RS-VQA、Knowledge Bridge 与端到端流程

日期：2026-08-08

## 当前完成了什么

1. 按论文实验角色建立独立 RS-VQA 配置，与纯文本 RS-CC 模型组完全分离。论文配置保留 MiniMax-01、GPT-4o-mini、LLaMA-4-Maverick、Qwen2.5-VL-32B-Instruct、Mistral-Small-3.2 和 GPT-4o Judge。
2. 新增 operational RS-VQA 配置。当前服务器真实联调使用可访问的 OpenRouter MiniMax/LLaMA/Mistral、硅基流动 Qwen3-VL 候选，以及独立 Qwen3-VL-32B Judge；该配置只用于验证逻辑，不能作为论文复现实验结果。
3. 实现预设问题与用户问题策略：无用户问题时采用配置中的模板；用户问题匹配建筑、道路、植被、水体、空间位置或结构变化模板时保留原始措辞并标记为 `hybrid`；未知问题保持 `user`。
4. RS-VQA 每个问题由 5 个 VLM 并发生成候选，Judge 同时查看原始 A/B 图像完成单选和 1-10 分评分，最终仍采用“最高分、Judge 仅作为并列裁决、标签顺序兜底”的论文逻辑。
5. VLM 的视觉输入严格只有原始双时相图像。图像在内存中转为 Base64，artifact 只保存图像 SHA-256，不保存 Base64 数据。
6. 实现 `C*`-only Knowledge Bridge：只传递 RS-CC 最终选中描述和来源 artifact，不暴露被淘汰候选。VQA 将 `C*` 作为可纠错的辅助文本，Judge 仍独立根据原始图像评分。
7. 新增端到端入口 `rs-agent-run`：`原始 Change-Agent 描述 -> RS-CC -> C* -> RS-VQA -> 统一结果 artifact`。单条完整运行产生 7 类可追溯 JSON artifact。
8. OpenAI-compatible Provider 支持模型级 `request_options`，可发送 OpenRouter `reasoning` 参数；完整保留 `reasoning_details`，可在多轮调用中原样回传。
9. 修复 OpenRouter HTTP 200 中嵌套 `finish_reason=error` 的情况。此类残缺输出现在会重试或作为失败候选记录，不再被当作正常答案。
10. 自动化测试增至 38 条，完整 pytest 与 Ruff 均通过。

## 真实流程验证

### RS-VQA 单阶段

- 输入：LEVIR-MCI `test_000001` 原始 A/B 图像，一个用户问题。
- 结果：5 候选调用、失败候选容忍、视觉 Judge、评分选择和 3 类 artifact 均可正常执行。
- 运行目录：`/root/autodl-tmp/rs-agent-artifacts/phase5-vqa/`。

### 端到端流程

- 运行 ID：`phase5-end-to-end-001`。
- 原始 Change-Agent 描述：`the scene is the same as before`。
- RS-CC 最终 `C*`：`The observed scene exhibits no discernible changes when compared to the previous imagery.`
- `C*` 已通过 Knowledge Bridge 传入 RS-VQA，完整 7 类 artifact 均写入成功。
- 运行目录：`/root/autodl-tmp/rs-agent-artifacts/phase5-end-to-end/phase5-end-to-end-001/`。

## 网络问题结论

1. 同一个 OpenRouter key 在用户本地电脑可调用 OpenAI、Google 和 Anthropic 模型。
2. 当前 AutoDL 节点使用官方 requests 形式、httpx 形式、带/不带 `reasoning`、带/不带 OpenRouter 推荐头均对上述模型返回同一 HTTP 403。
3. 同一节点和 key 可以成功调用 OpenRouter MiniMax、LLaMA、Mistral，因此 key、请求序列化和 Provider 主逻辑均正常；根因是服务器出口 IP 或上游区域路由限制。
4. 当前阶段不再让该网络问题阻塞开发。论文 profile 保留精确角色，真实流程统一使用 smoke profile。后续可在本地运行论文 profile，或通过合规 `HTTPS_PROXY`/OpenAI-compatible 网关更换出口。

## 目前存在哪些问题

1. `test_000001` 的官方变化标注为全黑：65,536 个像素全部为无变化。尽管如此，多个 VLM 仍把季节/色彩差异误报为植被或土地覆盖结构变化，说明视觉问答存在真实幻觉风险。
2. RS-CC 的 `C*` 与 RS-VQA 最终答案可能冲突。当前系统完整记录两类证据，但尚未实现 Main Agent 的冲突检测、可信度校准和最终仲裁。
3. operational profile 的 Mistral-Small-3.1 偶发返回 HTTP 200 包裹的上游 500；Provider 已能正确识别，VQA 以最少 4 个成功候选继续运行。
4. 论文使用的 Qwen2.5-VL-32B 已从当前硅基流动服务下线；论文配置保留该 ID 用于实验 provenance，实际 smoke 使用 Qwen3-VL 替代。
5. 可选变化 mask、批量 RS-VQA 输入、断点续跑、缓存和 Web 演示尚未接入完整流程。

## 后续需要实现什么

1. 优先实现 Main Agent：检测 `C*`、VQA、可选 mask 之间的冲突，输出带证据来源和不确定性的最终回答，不允许任一文本 Agent 自动覆盖视觉/掩膜证据。
2. 接入可选 mask 证据门控。VLM 仍只读取原始 A/B 图像；mask 的面积、连通域和位置摘要作为结构化证据交给 Main Agent。
3. 为 operational profile 增加启动前能力探测、模型健康矩阵、熔断与明确的替代模型记录；论文 profile 禁止静默替换。
4. 增加 JSONL 批处理、断点续跑、按图像哈希/问题/模型缓存，控制模板问题带来的 API 成本。
5. 扩展现有评估导出，加入跨 Agent 一致性、无变化误报率、候选失败率和 mask 一致性指标。
6. 完成上述实验流水线后，再基于 `Multi_change/web_demo.py` 实现 Streamlit 交互演示。

## 关机前检查

代码、配置、测试、中文进度文档、英文 README、真实运行 artifact 和数据路径需完成检查并推送 GitHub。确认仓库与 `/root/autodl-tmp` 数据完整后再关闭服务器。
