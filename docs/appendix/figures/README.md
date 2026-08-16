# 图池说明（docs/appendix/figures/）

本目录为全部论文配图的统一图池：每张图按 competition-record 步骤 2+4 双格式入池（png 300 dpi 位图 + pdf 矢量版，论文插图建议用 pdf，缩放不糊）。图配套信息（图号/图题、生成时间、坐标轴单位、图例释义、数据来源、读图结论）逐图记录于 `docs/appendix/figure_notes.md`，本文件只列清单与生成方式。

## 论文用图（15 张）

| 图号 | 文件（png / pdf 双份） | 论文位置 | 生成脚本 | 数据来源 |
|------|------------------------|----------|----------|----------|
| 图1 | `问题1_组1_接触网络与导电路径` | 问题1报告（组1 导通性判定段） | `Q1/plot_question1.py` | 附件.xlsx 组1（12 根介质A）+ Q1/results/question1_result.json |
| 图2 | `问题1_组2_接触网络与导电路径` | 问题1报告（组2 导通性判定段） | `Q1/plot_question1.py` | 附件.xlsx 组2（49 根介质A）+ 同上 |
| 图3 | `问题1_组3_接触网络与导电路径` | 问题1报告（组3 导通性判定段） | `Q1/plot_question1.py` | 附件.xlsx 组3（535 根介质A）+ 同上 |
| 图4 | `问题2_导通概率_vs_体积分数` | 问题2报告 图4（完整实体内外接夹逼曲线） | `Q2/plot_question2.py` | Q2/results/solid_boundary_comparison.json |
| 图5 | `问题2_几何统计量_vs_体积分数` | 问题2报告 §4.5（几何统计量随体积分数变化） | `Q2/plot_question2_statistics.py` | 同上 |
| 图6 | `问题2_跨壁比例_诊断` | 问题2报告_完整实体正式版 图8（轴线/实体口径对照） | `Q2/plot_question2_crossing_ratio.py` | solid_boundary_comparison.json + solid_boundary_sensitivity.json |
| 图7 | `问题3_实体夹逼_导通概率_vs_介质数量` | 问题3报告 图1（§4.2 节首） | `Q3/plot_solid_question3.py` | Q3/results/question3_solid_curve.csv + question3_solid_summary.json |
| 图8 | `问题3_临界区放大图` | 问题3报告 图2（图1 之后、逐点表之前） | `Q3/plot_question3_zoom_critical.py` | 同上（N=600~640 截取） |
| 图9 | `问题3_实体夹逼几何示意` | 问题3报告 图3（§3.3 径向误差界段后） | `Q3/plot_question3_geometry_bracket.py` | 纯几何（K=32、r_A=30 nm、阈值 1.8 nm） |
| 图10 | `问题3_边界口径敏感性_临界对照` | 问题3报告 图4（§4.4 敏感性对照段后） | `Q3/plot_question3_sensitivity_critical.py` | question3_result_hypothesis1.csv + curve.csv + summary.json |
| 图11 | `问题3_早停检查根数_诊断` | 问题3报告 图5（§4.1 汇总表之后） | `Q3/plot_question3_early_stop.py` | question3_solid_curve.csv（M=10000、K=32） |
| 图12 | `问题4_混合导通热图` | 问题4报告 图1（§4.1 基线与搜索定位） | `Q4/plot_question4.py` | Q4/results/question4_points.csv + question4_result.csv + question4_verify.csv |
| 图13 | `问题4_可行边界验证` | 问题4报告 图2（§4.2 主种子验证带） | `Q4/plot_question4.py` | Q4/results/question4_boundary.csv |
| 图14 | `问题4_独立复算置信区间` | 问题4报告 图3（§4.3 独立种子复算） | `Q4/plot_question4_verify_ci.py` | Q4/results/question4_verify.csv |
| 图15 | `问题4_严格实体复核夹逼` | 问题4报告 图4（§4.6 严格实体复核） | `Q4/plot_question4_strict_mixed.py` | Q4/results/question4_strict_mixed_candidates.csv |

> 注：图号沿用 `docs/appendix/figure_notes.md` 的全局编号（图1~图16，按入池顺序），与各题报告内的局部图号不对应（如问题4报告 图1 = 本池图12）；`问题3_实体夹逼_导通概率_vs_介质数量`（Q3 报告 图1）暂无 notes 条目。

## 存档图（已不再被报告引用，保留供追溯）

| 文件（png / pdf 双份） | 说明 | 生成脚本 | 数据来源 |
|------------------------|------|----------|----------|
| `问题3_导通概率_vs_介质数量` | 问题3 轴线口径旧版（假设二曲线，N̂90=611/保守 612），问题3报告已换实体夹逼版（本池图7） | `docs/appendix/source_code/Q3/plot_question3.py`（工作区 Q3/ 已退役） | Q3/results/question3_curve.csv |
| `问题3_边界口径敏感性` | 问题3 旧口径敏感性对照（假设一 vs 假设二全高曲线），报告已换临界对照版（本池图10） | 同上 | question3_curve.csv + question3_curve_hypothesis1.csv |

## 如何重新生成

环境：`conda activate math`（非交互 shell 用全路径 `E:\Software\Scoop\apps\miniconda3\current\envs\math\python.exe`，GBK 问题见 CLAUDE.md）。

```bash
# 问题1（3 张：组1/组2/组3）
python Q1/plot_question1.py
# 问题2（3 张：导通概率曲线、几何统计量、跨壁比例诊断）
python Q2/plot_question2.py
python Q2/plot_question2_statistics.py
python Q2/plot_question2_crossing_ratio.py
# 问题3（5 张论文图）
python Q3/plot_solid_question3.py
python Q3/plot_question3_zoom_critical.py
python Q3/plot_question3_geometry_bracket.py
python Q3/plot_question3_sensitivity_critical.py
python Q3/plot_question3_early_stop.py
# 问题4（4 张）
python Q4/plot_question4.py
python Q4/plot_question4_verify_ci.py
python Q4/plot_question4_strict_mixed.py
```

每个脚本读取对应题 `Qx/results/` 下的结果文件（缺数据时脚本会报错提示先跑生成程序），png+pdf 双格式写入 `Qx/figures/` 并自动复制入本图池。图配套信息按新图追加到 `docs/appendix/figure_notes.md`（模板 `.claude/skills/competition-record/references/figure_notes_entry.md`），源程序复制入 `docs/appendix/source_code/`。

## 数据口径要点

- 问题2/3 正式运行：M=2000~20000 试验、K=32~64 边形实体夹逼、seed=20260808 系列；问题3 另含假设一（自动电连续）验证运行 M=2000
- 问题4：搜索点估计 M=100、独立复算与严格复核 M=4000（共同随机数）、Wilson 95% 区间、目标概率 P=0.90
- 旧图（存档区）为轴线口径，勿与实体夹逼口径混用
