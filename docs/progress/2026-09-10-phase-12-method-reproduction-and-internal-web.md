# Phase 12：50 条方法复现与服务器内部 Web Demo

## 本阶段目标

本阶段只完成方法层面的复现和服务器内部演示，不追求论文原模型数值复现，不计算人工偏好率、评审者一致性，也不整理可直接写入论文或毕业论文的实验表格与结论。

## 当前完成了什么

1. 在服务器重启后核对 Git、conda 环境、GPU、内存、磁盘、API 配置和 Phase 11 备份。
2. 生成 Phase 12 服务器资源快照 `resources.json`。
3. 对方法配置中的全部候选模型、Selector 和 Evaluator 执行真实文本或双图 completion preflight，所有角色均返回 HTTP 200。
4. 对 50 条 LEVIR-MCI 分层清单执行独立 dry-run，确认图像、原始描述、predicted mask、配置和实验身份有效。
5. 新建并完成 50 条 combined batch，不混用早期 5 条 pilot 的旧 source fingerprint。
6. 完成 checksum-verified export，保留全部候选、Judge 评分、请求遥测、mask 证据、冲突和 provenance。
7. Streamlit 支持通过 `RS_AGENT_WEB_ARTIFACT_DIR` 将结果写到仓库外的持久目录。
8. 新增 `scripts/manage_internal_web_demo.sh`，支持 `start`、`status`、`stop`，并拒绝绑定非 loopback 地址。
9. 新增 GitHub Actions，在 push 和 pull request 时运行 pytest 与 Ruff。
10. Web Demo 已部署在服务器 `127.0.0.1:8501`，只能通过 SSH 隧道访问。

## 服务器资源

- CPU：64 个物理核心，128 个逻辑核心；
- 内存：约 1 TB，总可用约 898 GB；
- GPU：NVIDIA GeForce RTX 4090，24 GB，运行前为空闲；
- `/root/autodl-tmp`：约 43 GB 可用；
- API 主流程以网络 I/O 为主，不依赖本地 GPU；GPU 只在重新运行 Change-Agent 时需要。

资源快照：

```text
/root/autodl-tmp/rs-agent-runs/phase12/resources.json
```

## 方法复现协议

使用配置：

```text
configs/rs_cc.method_image_text.yaml
configs/rs_vqa.adapted.yaml
```

执行逻辑：

1. RS-CC 的 5 个 VLM 接收原始双时相图像和一条 Change-Agent 原始描述；
2. Caption Selector 和 Evaluator 保留完整候选账本并选择最高分 `C*`；
3. Knowledge Bridge 只传递最终 `C*` 和来源；
4. RS-VQA 的 5 个 VLM 只接收原始双时相图像，文本上下文包含系统问题或用户问题以及可选 `C*`；
5. predicted mask 只进入结构化 Evidence Bundle，不进入 VLM 图像消息；
6. Main Agent 进行确定性规划，不增加论文未定义的答案融合模型。

该配置明确标记为 substitute-model method profile，因此用于复现论文方法结构，不冒充论文原模型组合。

## 50 条运行结果

输入：

```text
/root/autodl-tmp/rs-agent-data/phase11-20260821/levir_mci_stratified_50.jsonl
```

Batch：

```text
batch_id: phase12-method-combined-50
status: completed
experiment_fingerprint: 226f5b02bdf36aea1245c64080e50e8a6a6c1f1c02ca1c70787e14c4e217b814
```

完整性检查：

- 50/50 items completed，0 failed；
- RS-CC：250/250 候选成功，50 条 selected caption；
- RS-VQA：250/250 候选成功，50 条 selected answer；
- 请求遥测：700 条，全部最终成功；
- OpenRouter：400 条 HTTP 200；
- SiliconFlow：300 条 HTTP 200；
- 8 条请求发生一次瞬时重试，均恢复成功；
- `failures.jsonl` 为空；
- 18 条结果被 Evidence Bundle 标记为 uncertain conflict，需要在交互界面中保留证据来源，不应被静默覆盖。

本阶段只把自动参考指标作为工程诊断输出，不据此形成论文结论。Exporter 默认生成的 bootstrap 字段未用于本阶段分析。

## 结果路径

```text
/root/autodl-tmp/rs-agent-runs/phase12/state/phase12-method-combined-50/state.json
/root/autodl-tmp/rs-agent-runs/phase12/artifacts/
/root/autodl-tmp/rs-agent-runs/phase12/exports/method-combined-50/
/root/autodl-tmp/rs-agent-runs/phase12/preflight-method-current.json
```

## Web Demo

当前部署：

```text
listen: 127.0.0.1:8501
manifest: /root/autodl-tmp/rs-agent-data/phase11-20260821/levir_mci_stratified_50.jsonl
runtime: /root/autodl-tmp/rs-agent-runs/phase12/web/runtime
uploads: /root/autodl-tmp/rs-agent-runs/phase12/web/work
artifacts: /root/autodl-tmp/rs-agent-runs/phase12/web/artifacts
```

本地访问命令：

```bash
ssh -p 36247 -L 8501:127.0.0.1:8501 root@connect.bjb1.seetacloud.com
```

保持该 SSH 会话后，在本地浏览器打开 `http://127.0.0.1:8501`。

管理脚本：

```bash
scripts/manage_internal_web_demo.sh start
scripts/manage_internal_web_demo.sh status
scripts/manage_internal_web_demo.sh stop
```

当前部署不开放公网端口，也不增加反向代理、TLS、公共用户注册、配额或生产级任务队列。

## 验证

- `pytest -q`：93 passed；
- `ruff check .`：通过；
- `bash -n scripts/manage_internal_web_demo.sh`：通过；
- Streamlit `/_stcore/health`：`ok`；
- 重复 `start`：正确识别已运行进程；
- GitHub Actions：提交 `7bf24fb` 的 main 与发布分支两次运行均通过；
- GitHub 默认 `main` 与 `agent/rs-cc-pipeline` 均已发布到 `7bf24fb`。

## 当前问题

1. 方法复现使用可用替代模型，不是论文原 API 模型组合；
2. Web Demo 的 detached 进程在服务器重启后不会自动启动，需要重新执行管理脚本；
3. 18 条 evidence conflict 反映 caption、VQA 或 mask 之间存在分歧，目前按设计保守呈现；
4. 本阶段没有人工评价，也不输出论文式显著性结论；
5. 当前只运行 50 条分层样本，尚未扩展到 LEVIR-MCI 完整测试集。

## 后续可以实现

1. 在不改变方法协议的前提下扩展到完整测试集；
2. 为服务器内部 Demo 增加可选的开机恢复机制；
3. 增加运行历史浏览、artifact 下载和失败任务恢复界面；
4. 将 retrieval memory 作为独立、默认关闭的非论文实验配置；
5. 新增 `domains/medical/`，复用当前候选、Judge、Knowledge Bridge、artifact 和评估基础设施。
