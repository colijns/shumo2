# AI 工具信息表

> 本表与每份绘图代码头部注释、ai_declaration_draft.md 保持一致。使用新工具时追加行。

| 工具名称 | 版本 | 开发公司 | 使用日期 | 主要用途 |
|----------|------|----------|----------|----------|
| ZCode（GLM） | GLM-5.3 | 北京智谱华章科技有限公司 | 2026-08-16 起 | 问题1 求解代码编写（OR-Tools 两阶段管线）、算法诊断与调参、结果校验、附录文档维护 |
| Claude Code | DeepSeek-V4-Pro | Anthropic（模型：DeepSeek） | 2026-08-17 | 问题1 第二轮优化（Q1/tight_search.py：CP-SAT 紧档可行性模型、GLS 固定 N 精修、四算例压缩轨道、8 进程编排、promote 选优与独立校验）、文档同步 |
| Codex | GPT-5（Codex） | OpenAI | 2026-08-18 | 问题3 冲突聚焦邻域设计、短时对照试验、合法性复核与试验说明整理 |

## 使用方式说明
- ZCode 为命令行 AI 编程代理，经本地终端交互使用
- Codex 为本地代码协作代理，本次仅在隔离工作树中开展小样本试验
- 完整交互记录：interaction_logs/prompts/ 与 interaction_logs/transcripts/
