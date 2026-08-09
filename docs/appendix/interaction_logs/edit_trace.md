# 人工修改痕迹记录

> 记录用户对 AI 生成内容的每次人工修改。只追加行。触发场景：手动改写 AI 生成的代码/文字、删除 AI 生成的段落、调整 AI 输出的参数。

| 时间 | 文件 | 改动摘要 | 修改原因 |
|------|------|----------|----------|
| 2026-08-09 | Q2/plot_question2.py | 图4 重绘：数据源 question2_result.csv 换 solid_boundary_comparison.json，单线改内接/外切双线夹逼带 + fill_between，删内嵌标题 | 正式结果（8000 共同样本，K=64）口径同步 |
| 2026-08-09 | Q2/plot_question2_statistics.py | 图7 重绘：数据源换 solid_boundary_comparison.json，删接触边线，改内/外接片段与跨壁 4 线 | 正式结果口径同步 |
| 2026-08-09 | Q2/plot_question2_crossing_ratio.py | 图8 新增：跨壁比例诊断柱状图（实测柱 + 理论虚线） | 正式报告 §4.5 跨壁比例诊断配图 |
| 2026-08-09 | Q2/plot_question2.py | 图4 微调：内/外接线横轴错位 ±0.25 pp 区分重合曲线 | 蓝橙线条视觉重合 |
| 2026-08-09 | Q2/plot_question2_statistics.py | 图7 精简：4 线合并为内接 2 线（外切与内接重合不另画） | 内外接曲线重合 |
| 2026-08-09 | Q2/plot_question2_crossing_ratio.py | 图8 修改：移除轴线柱、轴线理论线与 0.73pp 注释，仅留实体两柱 + 实体理论线 | 口径统一后不提及旧口径 |
| 2026-08-09 | Q2/plot_question2_crossing_ratio.py | 图8 恢复轴线对照（三柱 + 两理论线 + 0.73pp 注释），措辞去"旧口径"标签，改称"轴线截断/轴线模型" | 轴线作实体退化基线验证有独立价值，保留对照 |
| 2026-08-09 | Q2/plot_question2_crossing_ratio.py | 图8 删除图上"实体 - 轴线 = 0.73 个百分点"两行注释 | 图中注释冗余，论证保留在图注与正文 |
