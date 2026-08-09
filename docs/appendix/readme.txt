============================================================
 数学建模竞赛 源程序与附件使用说明
 赛题：A题 微构体中填充导电介质的仿真优化（问题1/2/3）
 队伍编号：（待填）
 提交日期：2026-08-09
============================================================

一、内容构成
1. 全部源程序（建模计算 + 绘图；绘图代码头部含 AI 辅助注释）
2. 论文全部配图（矢量图 .pdf + 位图 .png，含中间废弃图）
3. AI 使用声明、AI 工具信息表、图配套信息
4. 完整 AI 交互记录（docs/appendix/interaction_logs/）

二、运行环境
- Python 3.10+；依赖 numpy pandas openpyxl matplotlib（安装：pip install -r requirements.txt）
- 中文 Windows 控制台如出现乱码，改用支持 UTF-8 的终端运行

三、运行方法
1. 问题1 主计算：cd Q1 && python run_question1.py
   （输出：三组导电结论 + results/question1_result.json）
2. 问题2 主计算（蒙特卡洛，4 φ 并行）：cd Q2 && python run_question2.py
   （输出：results/question2_result.csv + 终端表；全量约 14min，各 φ M=2000）
3. 全部测试：cd Q1 && python -m unittest discover -s tests -v；cd Q2 && python -m pytest tests -q；
   cd Q3 && python -m pytest tests -q
4. 出图：cd Q1 && python plot_question1.py；cd Q2 && python plot_question2.py（图4）；
   cd Q2 && python plot_question2_statistics.py（图7 几何统计量）；
   cd Q2 && python plot_question2_crossing_ratio.py（图8 跨壁比例诊断）
   （三图数据源：Q2/results/solid_boundary_comparison.json + solid_boundary_sensitivity.json，
   正式版 M=2000/φ，seed=20260808，K=64；输出各题 figures/ png 300dpi + pdf；
   论文用图见 docs/appendix/figures/）
   cd Q3 && python plot_solid_question3.py（图1 导通概率曲线）；
   cd Q3 && python plot_question3_zoom_critical.py（图2 临界区放大）；
   cd Q3 && python plot_question3_geometry_bracket.py（图3 实体夹逼几何示意）；
   cd Q3 && python plot_question3_sensitivity_critical.py（图4 边界口径敏感性对照）；
   cd Q3 && python plot_question3_early_stop.py（图5 早停检查根数诊断）
   （Q3 五图数据源：Q3/results/question3_solid_curve.csv + question3_solid_summary.json
   + question3_result_hypothesis1.csv，正式版 M=10000，seed=20260808+i，K=32；
   输出各题 figures/ png 300dpi + pdf；论文用图见 docs/appendix/figures/）
5. 附件读取按工作表索引 0/1/2（表名 GBK 乱码，勿按名字读）

四、AI 使用说明
- AI 工具信息表：docs/appendix/interaction_logs/ai_tools.md
- AI 使用声明：docs/appendix/ai_declaration_draft.md
- 人工修改痕迹：docs/appendix/interaction_logs/edit_trace.md
- 交互记录：docs/appendix/interaction_logs/prompts/ 与 transcripts/

五、文件清单（source_code/ 目录，随源程序入池同步更新）
Q1/core.py                                  几何内核：GJK 距离、周期边界、AABB 加速、并查集、BFS
Q1/run_question1.py                         附件三组导电判定主程序（输出 JSON）
Q1/plot_question1.py                        3D 接触网络与导电路径配图（AI 辅助注释头）
Q1/tests/test_core.py                       单元测试（几何/周期/电极/图连通，35 断言）
Q1/tests/test_question1_integration.py      集成测试（附件固定结果/PBC 敏感性/穷举对照，6 断言）
Q1/results/question1_result.json            三组完整结果（接触边、周期平移、见证路径、敏感性）
Q2/geometry.py                              问题2 几何核心：随机生成、PBC 截断、批量轴距、临界带 GJK、并查集判导通
Q2/monte_carlo.py                           每体积分数蒙特卡洛循环 + Wilson 95% CI
Q2/run_question2.py                         4 体积分数并行入口（ProcessPoolExecutor，seed=42+i）
Q2/plot_question2.py                        导通概率 vs 体积分数配图（AI 辅助注释头，png+pdf 双格式）
Q2/plot_question2_statistics.py             图7 几何统计量配图（AI 辅助注释头）
Q2/plot_question2_crossing_ratio.py         图8 跨壁比例诊断配图（AI 辅助注释头）
Q2/tests/test_clip.py / test_distance.py / test_monte_carlo.py  问题2 单元测试（32 断言）
Q2/results/question2_result.csv             4 体积分数导通概率、Wilson CI、平均片段/接触边统计
Q3/first_passage.py                         问题3 核心：逐根加入、首通数量、内外接夹逼、GJK 实体距离
Q3/run_question3.py                         问题3 正式模拟入口（M=10000，8 并行，约 57 min）
Q3/plot_solid_question3.py                  图1 导通概率曲线配图（AI 辅助注释头，png+pdf 双格式）
Q3/plot_question3_zoom_critical.py          图2 临界区放大配图（AI 辅助注释头）
Q3/plot_question3_geometry_bracket.py       图3 实体夹逼几何示意配图（AI 辅助注释头）
Q3/plot_question3_sensitivity_critical.py   图4 边界口径敏感性对照配图（AI 辅助注释头）
Q3/plot_question3_early_stop.py             图5 早停检查根数诊断配图（AI 辅助注释头）
Q3/tests/test_question3.py                  问题3 单元测试（9 项）
Q3/results/question3_solid_curve.csv        750 根逐点导通概率与 Wilson 区间（外切/内接）
Q3/results/question3_solid_summary.json     临界根数、夹逼区间、诊断统计汇总
