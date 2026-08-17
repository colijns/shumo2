============================================================
 数学建模竞赛 源程序与附件使用说明
 赛题：A题 园区微电网风光储协调优化配置
 队伍编号：（待填）
 提交日期：YYYY-MM-DD
============================================================

一、内容构成
1. 全部源程序（建模计算 + 绘图；绘图代码头部含 AI 辅助注释）
2. 论文全部配图（矢量图 .pdf + 位图 .png，含中间废弃图）
3. AI 使用声明、AI 工具信息表、图配套信息
4. 完整 AI 交互记录（docs/appendix/interaction_logs/）

二、运行环境
- Python 3.10；依赖 numpy scipy pandas matplotlib seaborn scikit-learn statsmodels networkx pytest
- 安装：pip install -r requirements.txt

三、运行方法
1. Q1（独立运营）：cd Q1 && python q1_main.py && python plot_results.py
2. Q2（联合运营）：cd Q2 && python q2_main.py
3. 出图：python <绘图脚本>（输出见对应 output*/ 目录；论文用图见 docs/appendix/figures/）

四、AI 使用说明
- AI 工具信息表：docs/appendix/interaction_logs/ai_tools.md
- AI 使用声明：docs/appendix/ai_declaration_draft.md
- 人工修改痕迹：docs/appendix/interaction_logs/edit_trace.md
- 交互记录：docs/appendix/interaction_logs/prompts/ 与 transcripts/

五、文件清单（source_code/ 目录，随源程序入池同步更新）
（列出与 source_code/ 一致的文件列表）
