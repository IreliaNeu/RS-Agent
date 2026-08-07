# 第四阶段进展：双 Provider、Change-Agent 小批量与真实流程验证

日期：2026-08-08

## 当前完成了什么

1. 在不输出 key 内容的前提下验证 `.env`：OpenRouter 和硅基流动的 key 均能访问各自 `/models` 接口。
2. 更新论文 RS-CC 配置：DeepSeek V3 与 Qwen3-30B 改走硅基流动，其余候选和 GPT-4o Judge 保持 OpenRouter。
3. 完成真实 API 可用性检查：硅基流动 DeepSeek/Qwen、OpenRouter LLaMA/Mistral 调用成功；OpenRouter Claude、GPT-4o-mini 受服务器区域限制返回 HTTP 403。
4. 保留 `configs/rs_cc.paper.yaml` 作为论文角色配置，新增 `configs/rs_cc.smoke.yaml` 作为明确的非论文流程验证配置，避免混淆实验结果。
5. 从官方仓库 `lcybuaa/Change-Agent` 下载 `MCI_model.pth`，SHA-256 为 `34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9`。
6. 建立独立 Conda 环境 `rs-agent-mci`，使用 RTX 4090、Torch 2.8.0、CUDA 12.8；旧 MCI 推理依赖不进入 `rs-agent` API 环境。
7. 新增 `scripts/generate_change_agent_captions.py`：模型只加载一次，输出逐条刷新，manifest 记录权重哈希、Torch/CUDA、GPU、样本数和耗时。
8. 对 LEVIR-MCI 测试集固定前 100 条重新推理，生成 100 条描述，空描述为 0，耗时约 10.2 秒。
9. 抽取无变化、建筑新增、道路新增三条代表样本，完整执行五候选生成、Selector、Evaluator、C* 选择和不可变产物写入，3 条全部成功。
10. 对三条结果的 9 个 JSON 产物执行 checksum 验证，全部通过。真实流程验证还确认：最高评分候选可以覆盖不同的 Judge Choice，符合既定 C* 规则。
11. 自动化测试增至 29 条；Ruff、pytest、pip check 和凭据扫描将在提交前再次执行。

## 真实流程结果摘要

- `test_000001`：Judge Choice 为 B，C 得分 10，最终 C* 为 C。
- `test_000044`：Judge Choice 与最高分均为 B，最终 C* 为 B。
- `test_000068`：Judge Choice 为 C，D 得分 10，最终 C* 为 D。

结果目录：

```text
/root/autodl-tmp/rs-agent-artifacts/change-agent-smoke-3-20260808/
```

## 目前存在哪些问题

1. 当前 AutoDL 服务器区域无法通过 OpenRouter 调用 Anthropic、OpenAI 和 Google 模型，因此不能在该节点完成严格论文模型组合的真实复现。
2. `configs/rs_cc.smoke.yaml` 使用替代模型和 DeepSeek Judge，只能说明流程逻辑正确，不能用于论文结果比较。
3. Change-Agent 官方代码在 Torch 2.8 下会出现 timm 旧导入和 attention mask 类型的兼容性警告，但 100 条推理结果完整、无空描述。
4. MCI 输出本身存在重复措辞，例如 `on the bareland on the bareland`；这也说明后续多模型文本增强与评估阶段具有实际必要性。

## 数据与产物位置

```text
/root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100.jsonl
/root/autodl-tmp/rs-agent-data/change-agent/levir_mci_test_100.jsonl.manifest.json
/root/autodl-tmp/rs-agent-data/change-agent/levir_mci_pipeline_smoke_3.jsonl
/root/autodl-tmp/rs-agent-artifacts/change-agent-smoke-3-20260808/
/root/autodl-tmp/models/change-agent/MCI_model.pth
```

## 后续需要实现什么

1. 实现与 RS-CC 独立的 RS-VQA 五模型配置和原始双时相图像输入适配。
2. 设计遥感变化预设问题模板，并实现用户问题、模板问题和混合问题的规范化。
3. 建立 Knowledge Bridge，只传递最终 C*；随后加入可选 mask 证据。
4. 完成主 Agent 编排和评估后，再实现 Streamlit 网页交互。
5. 如需严格论文复现，应更换允许 OpenRouter Anthropic/OpenAI 模型的运行区域，或为这些角色配置合规可用的等价 API。

## 关机前检查

代码、100 条输入、manifest、API 产物、校验摘要、模型权重和环境说明均需完成哈希/路径检查并归档；GitHub 推送成功后再关闭服务器。
