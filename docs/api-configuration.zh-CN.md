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
SILICONFLOW_API_KEY=替换为你的SiliconFlowKey
RS_AGENT_ARTIFACT_DIR=./artifacts
RS_AGENT_CACHE_DIR=./cache
```

两个 key 已在服务器上通过 `/models` 接口验证，但文档和日志不会记录其具体内容。

## 论文配置

```text
/root/autodl-tmp/RS-Agent/configs/rs_cc.paper.yaml
```

论文角色保持不变：

- DeepSeek V3：硅基流动 `deepseek-ai/DeepSeek-V3`
- Qwen3-30B：硅基流动 `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Claude-Sonnet-4、GPT-4o-mini、LLaMA-4-Maverick：OpenRouter
- GPT-4o Selector 和 Evaluator：OpenRouter，保持两个独立角色

服务器所在区域调用 OpenRouter 的 Anthropic、OpenAI 和 Google 模型时返回 HTTP 403。该限制来自模型区域策略，不是 key 无效。LLaMA、Mistral 以及硅基流动的 DeepSeek/Qwen 已完成真实调用验证。

## 流程验证配置

```text
/root/autodl-tmp/RS-Agent/configs/rs_cc.smoke.yaml
```

该配置仅用于验证代码和产物逻辑，采用 OpenRouter 上地区可用的 Mistral/LLaMA，以及硅基流动的 DeepSeek/Qwen。Selector 和 Evaluator 使用硅基流动 DeepSeek V3。

该配置不是论文模型组合，输出不能作为论文复现实验结果。

## 无 API 请求校验

```bash
cd /root/autodl-tmp/RS-Agent
conda activate rs-agent
rs-agent-cc \
  --config configs/rs_cc.paper.yaml \
  --input examples/rs_cc_input.jsonl \
  --dry-run
```

## 可运行流程验证

```bash
cd /root/autodl-tmp/RS-Agent
conda activate rs-agent
rs-agent-cc \
  --config configs/rs_cc.smoke.yaml \
  --input /root/autodl-tmp/rs-agent-data/change-agent/levir_mci_pipeline_smoke_3.jsonl \
  --artifact-dir /root/autodl-tmp/rs-agent-artifacts
```

输入中每个图像对只对应一条 Change-Agent 生成描述。不要直接把 LEVIR-MCI 的五条人工 ground truth 当成正式 RS-CC 原始输入。
