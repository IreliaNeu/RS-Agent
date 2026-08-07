# API 配置说明

## 需要修改的文件

服务器文件：

```text
/root/autodl-tmp/RS-Agent/.env
```

该文件已被 `.gitignore` 排除，不会提交到 GitHub。不要把真实 API key
写入 `configs/*.yaml`、Python 文件或进展文档。

## 填写格式

```dotenv
OPENROUTER_API_KEY=替换为你的OpenRouterKey
SILICONFLOW_API_KEY=替换为你的SiliconFlowKey
OPENAI_API_KEY=替换为你的OpenAIKey
RS_AGENT_ARTIFACT_DIR=./artifacts
RS_AGENT_CACHE_DIR=./cache
```

当前优先使用 OpenRouter。没有使用的供应商可以暂时留空。

## 模型配置

模型 ID、Provider、并发数和超时配置位于：

```text
/root/autodl-tmp/RS-Agent/configs/models.example.yaml
```

该 YAML 文件只填写模型 ID 和环境变量名称，不填写 key。正式联调前会复制为
独立实验配置，避免覆盖示例文件。

## 临时加载环境变量

```bash
cd /root/autodl-tmp/RS-Agent
set -a
source .env
set +a
```

后续 CLI 和网页入口实现后会自动读取 `.env`，不再需要手动执行上述命令。

