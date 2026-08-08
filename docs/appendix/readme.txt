============================================================
 数学建模竞赛 源程序与附件使用说明
 赛题：A题 微构体中填充导电介质的仿真优化（问题1）
 队伍编号：（待填）
 提交日期：2026-08-07
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
2. 全部测试：cd Q1 && python -m unittest discover -s tests -v
3. 出图：cd Q1 && python plot_question1.py
   （输出 Q1/figures/ png 300dpi + pdf；论文用图见 docs/appendix/figures/）
4. 附件读取按工作表索引 0/1/2（表名 GBK 乱码，勿按名字读）

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
