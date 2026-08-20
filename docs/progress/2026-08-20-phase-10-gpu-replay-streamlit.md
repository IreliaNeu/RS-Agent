# 第十阶段：GPU 数据复现、严格候选消融与 Streamlit Demo

## 本阶段目标

1. 在 GPU 恢复后重新运行 Change-Agent，固定 100 条 LEVIR-MCI caption 与 mask 数据版本；
2. 解决上一阶段公共候选被重复调用的问题，完成严格候选 replay 消融；
3. 将稳定后的研究流水线接入独立 Streamlit Web Demo；
4. 保持论文基线、增强实验和 Web 交互边界清晰；
5. 补齐统一项目架构与论文对应文档。

## 已完成工作

### 1. 服务器与 GPU 审计

- GPU：NVIDIA RTX 4090，显存约 24 GB；
- 容器资源：16 核 CPU、120 GB 内存；
- API 环境：`/root/miniconda3/envs/rs-agent`；
- Change-Agent 环境：`/root/miniconda3/envs/rs-agent-mci`；
- 数据集、MCI checkpoint 和上游 Change-Agent 源码均完整保留。

### 2. Change-Agent 100 条数据重新生成

输出目录：

```text
/root/autodl-tmp/rs-agent-data/change-agent/phase10-20260820/
```

结果：

- 100/100 条 caption 生成成功；
- 100/100 张三类 argmax mask 生成成功；
- 空 caption 为 0；
- 推理耗时约 11.25 秒；
- JSONL SHA-256：`393d0f6a535380e8a8739d19e998c6dcdc6d2880a44b8865975db31d6f73636b`；
- MCI checkpoint SHA-256：`34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9`。

新旧版本逐样本比较后，caption 差异为 0，mask 二值/类别像素差异为 0，说明当前数据生成路径可复现。

### 3. 候选 replay 机制

新增 `src/rs_agent/experiments/replay.py` 和 batch CLI 参数：

```text
--replay-caption-state
--replay-caption-label
```

replay 会验证来源 state、artifact 类型、artifact checksum、item ID、候选标签、模型身份、输入模式和文本内容。复用候选会生成新的 provenance 记录，但不会发送 provider 请求，也不会被错误计入本轮 HTTP 遥测。

### 4. 严格控制 RS-CC 消融

- 样本：同一组 10 change + 10 no-change；
- baseline：上一阶段 text-only 结果；
- enhancement：A/B 公共文本候选完全 replay，只重生成 C/D/E 图文候选；
- 结果：20/20 completed，0 failed；
- 候选：100 条，其中 40 条 replay；
- 本轮请求遥测：100 条，符合 `20 x (3 generation + 2 Judge)`；
- batch fingerprint：`ba32df532cdb0333dc674ca4cdfced545e41a5ddf39cb5263f058c93f32fbce9`。

| 指标 | enhancement - baseline | 95% paired CI |
| --- | ---: | ---: |
| BLEU-1 | -0.0532160 | [-0.0955675, -0.0176810] |
| BLEU-4 | +0.0024589 | [-0.0109158, 0.0207516] |
| ROUGE-L | -0.0382595 | [-0.0726018, -0.0078039] |
| change flag | 0.0000000 | [0.0000000, 0.0000000] |
| Judge score | -0.3500000 | [-0.7500000, 0.0000000] |

严格控制后，BLEU-1 和 ROUGE-L 仍显著下降。因此图文 RS-CC 继续保留为可选 enhancement，不替换论文纯文本基线。

### 5. Streamlit Web Demo

新增：

- `src/rs_agent/applications/web_demo.py`；
- `src/rs_agent/web/io.py`；
- `src/rs_agent/web/service.py`；
- `src/rs_agent/web/app.py`；
- `rs-agent-web` 命令和 `web` optional dependency。

页面支持 manifest 样本、图像上传、可选 mask、三类 profile、Caption/VQA/Combined、Knowledge Bridge 开关、预设或用户问题、五候选账本、Judge 分数、证据冲突、artifact 和连续追问。

未复用旧 `Multi_change/web_demo.py` 中的硬编码 key、代理和 Lagent 内部调用。新页面只调用稳定的 application service，并通过 `.env` 读取密钥。

### 6. 追问路径优化

首次分析产生的 `C*` 现在以 `KnowledgeBridgePacket` 保存来源。后续追问复用该 packet，只执行 RS-VQA，不重复运行 RS-CC。系统会校验 item ID，并在 Main Agent plan 与 Evidence Bundle 中记录 replay 来源，避免成本增加和同一会话的 `C*` 漂移。

### 7. 文档

新增 `docs/PROJECT_ARCHITECTURE_AND_PAPER_ALIGNMENT.zh-CN.md`，统一说明：

- 整体完成内容；
- 与论文逐项对应关系；
- 论文基线与工程增强边界；
- 项目架构、数据流、artifact 和评估；
- Streamlit 使用方式；
- 记忆策略；
- 迁移医学领域的最小改动路径。

## 验证结果

- Pytest：`82 passed`；
- Ruff：`src/`、`scripts/`、`tests/` 全部通过；
- Streamlit AppTest：页面可加载真实格式 manifest，无组件异常；
- Streamlit 本机运行：`http://127.0.0.1:8501/_stcore/health` 返回 `ok`；
- 启动日志无应用异常；
- 页面仅绑定服务器本机，未将无鉴权界面暴露到公网。

## 当前项目状态

- 论文主流水线：已形成完整可运行闭环；
- 评估与实验系统：候选级记录、恢复、缓存、身份、replay、Bootstrap、成对比较和 checksum 已具备；
- Change-Agent 数据：100 条 GPU 再生成版本已固定并验证可复现；
- Web Demo：研究演示所需主流程和交互追问已完成；
- 正式论文数值复现：仍受原模型访问、样本规模和指标协议限制。

## 当前问题

1. 严格 replay 消融只有 20 条样本，统计能力有限；
2. 当前可用 API 模型是显式替代组合，不能冒充论文原模型；
3. 图文模型对短参考描述可能过度扩写或误判季节差异，需要逐模型 prompt 校准；
4. 单一 Judge 可能有模型家族偏好，需增加独立 Judge 或多 Judge 实验；
5. 论文正式指标还需补充 corpus-level BLEU、METEOR、CIDEr 等一致实现；
6. Web Demo 没有认证、限流、后台队列，不应直接公网部署；
7. 长期检索记忆尚未实现，且不应加入论文基线。

## 后续任务

1. 扩展到 50-100 个按道路、建筑、混合变化分层的严格 replay 样本；
2. 在网络条件允许时运行 paper profile 原模型，并与替代模型结果分开报告；
3. 校准图文 RS-CC prompt，增加防伪变化、可见证据和长度约束；
4. 完成多 Judge 稳健性实验和论文全指标导出；
5. 为需要公网展示的 Web 版本增加认证、任务队列、配额和部署配置；
6. 在新领域需求明确后新增 domain package，不修改论文基线核心。
