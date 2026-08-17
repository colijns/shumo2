---
name: competition-record
description: Use when 用户要求生成图表、写或修改绘图代码、完成模型求解、写论文报告、画流程图，或需要维护 docs/appendix/ 下的 AI 使用全程留存材料（AI 辅助注释头、矢量图、图配套信息、AI 工具信息表、人工修改痕迹、AI 使用声明）。适用于数学建模比赛要求附录留存 AI 交互记录的场景。
---

# AI 使用全程留存

组委会要求论文附录与附件留存本次比赛的 AI 使用全过程。本 skill 只对**本会话的新产出**生效，不回填存量材料。存档根目录：`docs/appendix/`。其中 `interaction_logs/prompts/` 与 `interaction_logs/transcripts/` 由项目级 hook 自动写入，本 skill 不负责。

## 何时使用（分支）

- 写或改**绘图 .py** → 步骤 1（AI 头注释）+ 步骤 2（双格式存图）
- 完成一张图 → 步骤 3（图配套信息）+ 步骤 4（矢量图入池）
- 人工修改了 AI 写的代码 → 步骤 5（修改痕迹）
- 使用新 AI 工具 → 步骤 6（工具表核对）
- 每完成一个小问 → 步骤 7（声明草稿）+ 步骤 8（源程序入池）
- 首次运行或目录缺失 → 步骤 0（骨架检查）

## 执行步骤

0. **骨架检查**：确认 `docs/appendix/` 各子目录与文件齐全（interaction_logs/prompts、interaction_logs/transcripts、interaction_logs/ai_tools.md、interaction_logs/edit_trace.md、figure_notes.md、ai_declaration_draft.md、figures/、source_code/、readme.txt）。缺失的按 `references/` 同名模板补齐（ai_tools.md 与 edit_trace.md 在 interaction_logs/ 下）。
   完成：上述路径全部存在。

1. **AI 辅助注释头**：新建或修改任何绘图 .py 时，文件顶部插入注释头（模板 `references/ai_header_template.py`）。工具名称/版本/开发机构与 `ai_tools.md` 当前行完全一致，使用日期填当天。
   完成：该 .py 头部含"本绘图程序在AI工具辅助下完成"与"AI工具名称"两行，且与 ai_tools.md 一致。

2. **双格式存图**：每张图调用 `save_fig`（或 `fig.savefig`）同时保存 `.png` 与 `.pdf`（同名同目录，dpi>=300）。沿用 `templates/common/plot_style.py` 的中文样式约定（模板本身不修改，需要时复制出来用）。
   完成：每张新图存在成对的 `.png` 与 `.pdf` 文件。

3. **图配套信息**：每完成一张图，按 `references/figure_notes_entry.md` 条目格式在 `figure_notes.md` 末尾追加一条（图号图题、坐标轴单位、图例释义、子图标注、数据来源、读图结论/分析话术）。
   完成：figure_notes.md 中每张新图有对应条目，且"读图结论"段可直接复制进论文正文。

4. **矢量图入池**：把论文要用的 `.pdf` 与 `.png` 复制到 `docs/appendix/figures/`；被弃用的中间版本图放入 `figures/_abandoned/`（附件要求含中间废弃图）。
   完成：figures/ 与当前题目已完成图一一对应，废弃图有保留。

5. **人工修改痕迹**：用户手动改动了 AI 生成的代码/文件时，按 `references/edit_trace_entry.md` 格式在 `edit_trace.md` 追加一行（时间、文件、改动摘要、原因）。
   完成：该次改动在 edit_trace.md 中有记录。

6. **AI 工具表核对**：本会话使用任何 AI 工具（含版本/开发公司/日期）后，确认 `ai_tools.md` 表内有对应行；新工具则追加。
   完成：表内覆盖本会话全部工具，每行非空。

7. **声明草稿更新**：每完成一个小问，把该小问的 AI 用途描述（按绘图/建模计算/报告撰写三环节）按 `references/ai_declaration_draft.md` 的节结构追加进 `ai_declaration_draft.md`。
   完成：草稿覆盖到当前小问为止的全部环节，无"待补"占位遗留（占位仅允许出现在"提交前核对"清单）。

8. **源程序入池**：把本会话新写/修改的绘图、计算 .py 与流程图源码（.dot/.mmd/.drawio 文本）复制到 `docs/appendix/source_code/`（保留相对路径结构），并核对绘图代码 AI 头注释齐全。
   完成：source_code/ 与代码一一对应，所有绘图代码头部注释完整。

9. **readme 同步**：source_code/ 有新增时，按 `references/readme.txt` 模板更新 `docs/appendix/readme.txt`（运行环境、运行方法、文件清单）。
   完成：readme.txt 与 source_code/ 实际文件列表一致。

## 规则

- **只增不改**：所有留存文件只追加条目，不覆盖、不删除历史。
- **一致性**：AI 头注释、ai_tools.md、ai_declaration_draft.md 三处的工具名称/版本/开发机构必须一致（以 ai_tools.md 为准）。
- **时间戳**：一律本地时间 `YYYY-MM-DD HH:MM`。
- **边界**：不修改 `templates/`，不修改存量代码与报告，只登记本会话新产出。
- **话术可复用**：图配套信息的读图结论按论文正文口吻写，不用对话式语言。

## 常见错误

| 错误 | 补救 |
|------|------|
| 只存 .png 不存 .pdf | 补存 .pdf，并更新图配套信息 |
| 改了 AI 代码未记修改痕迹 | 补记 edit_trace.md，时间写实际修改时刻 |
| 头注释与 ai_tools.md 不一致 | 以 ai_tools.md 为准回改 |
| 误以为 hook 归档了图 | 交互记录不等于图配套信息，图必须走步骤 3 |
