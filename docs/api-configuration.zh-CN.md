# API 配置说明

## 需要修改的文件

服务器文件：

```text
/root/autodl-tmp/RS-Agent/.env
```

该文件已被 `.gitignore` 排除，不会提交到 GitHub。不要把真实 API key 写入 `configs/*.yaml`、Python 文件、产物或进展文档。

## 当前 RS-CC 所需配置

```dotenv
OPENROUTER_API_KEY=替换为你的OpenRouterKey
RS_AGENT_ARTIFACT_DIR=./artifacts
RS_AGENT_CACHE_DIR=./cache
```

目前论文 RS-CC 配置中的五个候选模型、Selector 和 Evaluator 均通过 OpenRouter 调用，所以只需要填写 `OPENROUTER_API_KEY`。未使用的供应商 key 可以暂时留空。

论文配置文件：

```text
/root/autodl-tmp/RS-Agent/configs/rs_cc.paper.yaml
```

YAML 只保存模型 ID、接口地址、温度和并发参数，不保存 key。当前候选模型为 Claude-Sonnet-4、DeepSeek V3、GPT-4o-mini、Qwen3-30B 和 LLaMA-4-Maverick；Selector 与 Evaluator 都使用 GPT-4o，但在配置中保持为两个独立角色。

## 无 Key 校验

以下命令只校验配置和 JSONL，不调用 API：

```bash
cd /root/autodl-tmp/RS-Agent
conda activate rs-agent
rs-agent-cc \
  --config configs/rs_cc.paper.yaml \
  --input examples/rs_cc_input.jsonl \
  --dry-run
```

## 正式运行

填写 key 后可执行：

```bash
cd /root/autodl-tmp/RS-Agent
conda activate rs-agent
rs-agent-cc \
  --config configs/rs_cc.paper.yaml \
  --input /path/to/change_agent_captions.jsonl \
  --artifact-dir /root/autodl-tmp/rs-agent-artifacts
```

输入中每个图像对只对应一条 Change-Agent 生成描述：

```json
{"item_id":"test_000001","original_caption":"Two buildings appeared near the road."}
```

兼容旧评估脚本中的 `Original Caption` 字段。不要直接把 LEVIR-MCI 的五条人工 ground truth 当成正式 RS-CC 的原始输入。
