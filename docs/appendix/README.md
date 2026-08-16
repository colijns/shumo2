# docs/appendix — AI 使用全程留存附录

| 内容 | 位置 | 维护方式 |
|------|------|----------|
| 用户提示词归档 | interaction_logs/prompts/ | hook 自动（UserPromptSubmit） |
| 会话记录（jsonl） | interaction_logs/transcripts/ | hook 自动（Stop，整体复制幂等） |
| 人工修改痕迹 | interaction_logs/edit_trace.md | skill 步骤 5 |
| AI 工具信息表 | interaction_logs/ai_tools.md | skill 步骤 6 |
| 图配套信息 | figure_notes.md | skill 步骤 3 |
| 矢量图 + png 汇总池 | figures/（废弃图在 _abandoned/） | skill 步骤 4 |
| AI 使用声明草稿 | ai_declaration_draft.md | skill 步骤 7 |
| 附件源程序池 | source_code/ | skill 步骤 8 |
| 附件 readme | readme.txt | skill 步骤 9 |
| hook 故障日志 | interaction_logs/hook_errors.log | 自动产生，正常为空/不存在 |

维护纪律：见 `.claude/skills/competition-record/SKILL.md`（只增不改、时间戳 `YYYY-MM-DD HH:MM`、三处工具信息一致）。

提交时：本目录全部材料 + 论文正文打包提交。
