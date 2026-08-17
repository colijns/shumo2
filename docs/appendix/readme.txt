============================================================
 数学建模竞赛 源程序与附件使用说明
 赛题：A题 低空经济背景下的多无人机协同巡检路径优化（问题1）
 队伍编号：（待填）
 提交日期：2026-08-16
============================================================

一、内容构成
1. 全部源程序（建模计算 + 绘图；绘图代码头部含 AI 辅助注释）
2. 论文全部配图（矢量图 .pdf + 位图 .png，含中间废弃图）
3. AI 使用声明、AI 工具信息表、图配套信息
4. 完整 AI 交互记录（docs/appendix/interaction_logs/）

二、运行环境
- Python 3.10（conda env math）；依赖 numpy scipy pandas openpyxl ortools
  （安装：pip install -r requirements.txt；ortools 版本 9.15.6755）
- Windows 控制台中文乱码时设 PYTHONIOENCODING=utf-8 再运行

三、运行方法
1. 问题1 全量求解（四算例约 40 min）：cd Q1 && python solve_q1.py
   （输出 outputs/workbooks/：result1.xlsx、summary_q1.xlsx、
     q1_solution_Case{1..4}.json、q1_verification_report.{json,md}）
2. 断点续跑：python solve_q1.py --resume（跳过已有完整解档案的算例）
3. 独立重放验证（不调用求解器，秒级）：python solve_q1.py --verify
4. 冒烟测试（Case4，30 s）：python solve_q1.py --smoke
5. 运行日志与阶段2收敛曲线：Q1/logs/
6. 求解口径：I/II/III 级展开 3/2/1 任务；坐标×0.1 km；飞行时间逐段
   向上取整到秒；单次巡检 300 s；单机上限 32400 s；同点任务禁止相邻
   （先离开再返回）；两阶段词典序（最小可行 N → 压 Tmax）。

四、AI 使用说明
- AI 工具信息表：docs/appendix/interaction_logs/ai_tools.md
- AI 使用声明：docs/appendix/ai_declaration_draft.md
- 人工修改痕迹：docs/appendix/interaction_logs/edit_trace.md
- 交互记录：docs/appendix/interaction_logs/prompts/ 与 transcripts/

五、文件清单（source_code/ 目录，随源程序入池同步更新）
Q1/solve_q1.py                          问题1 主程序：任务展开/双下界/两阶段
                                        OR-Tools 求解/sweep+kmeans 构造器/
                                        独立校验器/--verify 重放/--resume 续跑
Q1/logs/solve_q1_<时间戳>.log            全量运行日志（served 计数、约束数、耗时）
Q1/logs/convergence_Case{1..4}.csv      阶段2 GLS 收敛曲线（时刻, 目标值）
Q1/results/result1.xlsx                 正式结果：Case1~Case4 四表，
                                        列名照抄官方模板（UAV ID / 1th...）
Q1/results/summary_q1.xlsx              表2（N/Tmax/Tmin/最优性标记）+ 每机明细
Q1/results/q1_solution_Case{1..4}.json  解档案（复现锚点：任务序列/参数快照）
Q1/results/q1_verification_report.{json,md}  独立校验报告（六项检查）
Q1/diagnostics/diag_firstsol.py         首解策略对照实验（NextVar 约束使构造
                                        策略失效的定位过程）
Q1/diagnostics/diag_kmeans.py           kmeans 构造器再平衡插桩（震荡问题定位）
Q1/diagnostics/diag_rebal.py            再平衡平台期插桩
Q1/diagnostics/diag_ctor.py             双构造器独立验证脚本
Q1/tight_search.py                      问题1 第二轮优化：紧档（Case1=3、
                                        Case3=4）双轨道搜索（CP-SAT 可行性 +
                                        GLS 精修）+ 四算例压缩轨道 +
                                        --smoke 门/--orchestrate 编排/
                                        --promote 选优（2026-08-17）
Q1/logs/tight_<case>_N<n>_<track>_<ts>.log  第二轮各轨道日志
Q1/logs/tight_curve_<case>_N<n>.csv     轨道A GLS 收敛曲线
outputs/workbooks/tight_checkpoint_*.json  轨道检查点（原子写，按 case/n/track 唯一化）
outputs/workbooks/baseline_20260816/    第二轮开工前现行成果备份
