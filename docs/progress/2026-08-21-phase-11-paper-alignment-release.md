# Phase 11：论文协议对齐、完整评估与发布验证

## 当前完成了什么

1. 重新审计论文 PDF 和原 `TextAgent-LLMasJudge.py`，区分方法章节图文 RS-CC 与正式实验 text-only RS-CC。
2. 新增 `paper_experiment_text_only` 与 `paper_method_image_pair_plus_caption_interpretation` 两类显式协议。
3. 新增 RS-CC 有/无遥感背景知识的 CoT 配置。
4. 批处理结果传播 `dataset`、`change_type` 和完整 metadata。
5. 新增 checksum 校验的 `C*` state replay，支持 partial batch 和 `--max-items` pilot。
6. 新增论文式模型/变化类型均值标准差、corpus BLEU、CIDEr、盲评、Qwen3 嵌入、多 Judge 和 KB 比较工具。
7. preflight 从“模型目录可见”升级为“按角色进行文本或双图真实 completion 探针”，并重试瞬时错误。
8. 生成 50 条 GT 标签分层清单，完成修复前后两轮 5 条端到端验证。
9. 冻结 25 个候选完成 Qwen3/DeepSeek 多 Judge replay。
10. 完成 5 条 with/without Knowledge Bridge 严格消融。
11. Streamlit 增加可选 `RS_AGENT_WEB_PASSWORD` 门禁。

## 验证结果

- Pytest：最终 93 项测试通过；
- Ruff：`ruff check .` 通过，`legacy/` 作为上游快照显式排除；
- 修复后端到端：5/5 items、25/25 RS-CC、25/25 RS-VQA；
- KB 两组遥测均为 35 个 RS-VQA 请求和 0 个 RS-CC 请求；
- multi-Judge：selection agreement 1.0，kappa 1.0，Pearson 0.8581；
- embedding：human-agent 0.5062，human-human 0.6968；
- Streamlit：AppTest 和本地 health endpoint 验证；
- `.env` 和 API key 未进入 Git、artifact、state 或 export。

## 当前问题

1. OpenRouter 上 Claude/GPT-4o 系列在服务器区域返回 403；
2. SiliconFlow 不提供论文使用的 Qwen2.5-VL-32B；
3. 5 条 pilot 只能验证流程，不能给出论文规模统计结论；
4. 人工盲评工具已完成，但真实评分尚未收集；
5. 公网 Web 仍缺少生产级队列、限流和用户配额。

## 后续需要实现什么

1. 在允许调用论文原模型的本地或其他区域运行 exact paper profile；
2. 根据 API 预算把 50 条分层清单扩展为正式实验；
3. 组织论文一致的专家/学生盲评并使用现有评分工具汇总；
4. 如需公网部署，增加反向代理、TLS、任务队列、限流和监控；
5. 新领域任务通过独立 domain package 接入，不修改论文基线协议。

详细架构和论文对应关系见 `docs/PROJECT_STATUS_AND_PAPER_ALIGNMENT_PHASE11.zh-CN.md`。
