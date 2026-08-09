# 数学建模竞赛最终提交附件

A 题:微构体中填充导电介质的仿真优化

本附件包含支撑参赛论文模型、结果、结论的所有必要文件:各题全部源程序、正式结果数据、中间/备份结果数据、论文全部配图,以及 4 份最终版论文(快照)。

## 1. 附件构成与范围

**包含**:

- 各题正式源程序(主程序 `run_*.py`、核心计算模块、绘图脚本)
- 正式结果数据与中间/备份结果数据(`results/` 目录,含扫描与验证中间产物)
- 论文全部配图(png 300 dpi + pdf 矢量双格式,及 2 张过程流程图)
- 4 份最终版论文(`reports/`)

**不包含**(有意排除):

- 赛题提供的原始数据 `attachment/附件.xlsx`(组委会统一提供,见 §8 注意事项)
- 测试代码、AI 使用留存材料(另行按组委会要求提交)、代码模板库 `templates/`
- 已废弃程序:`Q2/run_question2.py`、`Q3/first_passage.py` 及其旧口径输出
  (`question2_result*`、`run_log*`、`question3_curve.csv`、`question3_result.csv` 等)

## 2. 目录结构

```
submission/
├── README.md                # 本说明
├── requirements.txt         # Python 依赖
├── reports/                 # 4 份最终版论文
│   ├── 问题1报告_修订版.md
│   ├── 问题2报告_完整实体正式版.md
│   ├── 问题3报告.md
│   └── 问题4报告.md
├── Q1/                      # 问题一:接触网络与导电路径判定
│   ├── run_question1.py     # 主程序(确定性,秒级)
│   ├── core.py              # 几何内核:GJK 距离/周期镜像/接触图/连通
│   ├── plot_question1.py    # 3 张配图
│   ├── results/             # question1_result.json 最终结果
│   └── figures/             # 问题1_组1/2/3 配图(png+pdf)
├── Q2/                      # 问题二:导通概率随体积分数变化(实体口径)
│   ├── run_solid_boundary_comparison.py  # 正式主程序(实体夹逼,耗时较长)
│   ├── solid_geometry.py    # 完整圆柱实体裁剪内核(内接/外切多棱柱)
│   ├── geometry.py          # 轴线口径几何内核(历史基线,被正式程序 import)
│   ├── monte_carlo.py       # 蒙特卡洛采样 + 95% Wilson 置信区间
│   ├── solid_boundary_sensitivity.py     # 跨壁比例敏感性诊断
│   ├── plot_question2.py    # 图4 导通概率 vs 体积分数
│   ├── plot_question2_statistics.py      # 图7 几何统计量
│   ├── plot_question2_crossing_ratio.py  # 图8 跨壁比例诊断
│   ├── results/             # solid_boundary_comparison.json 正式结果 + 备份
│   └── figures/             # 图4/图7/图8 配图 + 蒙特卡洛流程示意图
├── Q3/                      # 问题三:首次导通 90% 最低填充量(实体夹逼)
│   ├── run_question3.py     # 正式主程序(实体首次导通,M=10000,耗时较长)
│   ├── solid_first_passage.py   # 实体首次导通核心
│   ├── plot_solid_question3.py  # 图1 夹逼导通概率曲线
│   ├── plot_question3_zoom_critical.py        # 图2 临界区放大
│   ├── plot_question3_geometry_bracket.py     # 图3 几何夹逼示意
│   ├── plot_question3_sensitivity_critical.py # 图4 边界口径敏感性
│   ├── plot_question3_early_stop.py           # 图5 早停检查诊断
│   ├── results/             # question3_solid_curve.csv + summary.json 正式结果
│   └── figures/             # 图1~图5 配图 + 解题流程图
└── Q4/                      # 问题四:混合介质最低成本配置
    ├── run_question4.py     # 正式主程序(成本搜索 M=100 + 独立复算 M=4000)
    ├── run_strict_mixed_audit.py   # 严格实体复核(图4 数据源)
    ├── run_unified_a_audit.py      # 纯 A 退化情形严格复核
    ├── run_cost_scan.py / run_refined_scan.py   # 中间扫描脚本
    ├── geometry.py          # 混合几何基础内核
    ├── geometry_mix.py      # 混合接触网络与导通判定
    ├── monte_carlo.py       # 混合 MC 采样与成本计算
    ├── solid_mix_geometry.py # 严格复核实体几何核(A 多棱柱 + B 多面体夹逼)
    ├── plot_question4.py    # 图1 混合导通热图 + 图2 可行边界验证
    ├── plot_question4_verify_ci.py    # 图3 独立复算置信区间
    ├── plot_question4_strict_mixed.py # 图4 严格实体复核夹逼
    ├── results/             # 正式结果四件套 + 严格复核 + 8 个中间扫描/验证 json
    └── figures/             # 图1~图4 配图(取自已定稿图池)
```

## 3. 运行环境

- Python 3.10+,建议使用 conda 环境:

  ```bash
  conda activate math
  pip install -r requirements.txt
  ```

- 依赖仅需 `numpy / pandas / matplotlib / seaborn / openpyxl / pulp`
  (Q1-Q4 实际代码只 import numpy/pandas/matplotlib 及标准库)
- 中文 Windows 控制台:先在 UTF-8 终端执行 `conda activate math` 再 `python <脚本>`,
  不要在 `conda run -n math` 下运行(有 GBK 编码问题)

## 4. 复现命令

每题先在对应目录下执行,输出写入该题 `results/`(绘图输出写入 `figures/`):

**Q1**(确定性,秒级):

```bash
cd Q1
python run_question1.py          # 输出 results/question1_result.json
python plot_question1.py         # 3 张配图
```

**Q2**(实体夹逼蒙特卡洛,运行耗时较长):

```bash
cd Q2
python run_solid_boundary_comparison.py   # 正式结果 solid_boundary_comparison.json
python solid_boundary_sensitivity.py      # 敏感性 solid_boundary_sensitivity.json
python plot_question2.py                  # 图4
python plot_question2_statistics.py       # 图7
python plot_question2_crossing_ratio.py   # 图8
```

**Q3**(M=10000 实体首次导通,运行耗时很长):

```bash
cd Q3
python run_question3.py          # question3_solid_curve.csv + question3_solid_summary.json
python plot_solid_question3.py   # 图1
python plot_question3_zoom_critical.py        # 图2
python plot_question3_geometry_bracket.py     # 图3
python plot_question3_sensitivity_critical.py # 图4
python plot_question3_early_stop.py           # 图5
```

**Q4**:

```bash
cd Q4
python run_question4.py                     # 搜索 + 独立复算(四件套)
python run_cost_scan.py                     # 中间:成本层粗筛
python run_refined_scan.py                  # 中间:共同随机数精筛(见 §5 种子说明)
python run_unified_a_audit.py               # 纯 A 退化严格复核
python run_strict_mixed_audit.py            # 严格实体复核(耗时较长)
python plot_question4.py                    # 图1/图2
python plot_question4_verify_ci.py          # 图3
python plot_question4_strict_mixed.py       # 图4
```

## 5. 随机种子约定(可复现)

| 题目 | 主运行种子 | 规模 | 覆盖方式 |
|------|-----------|------|---------|
| Q1 | 确定性计算,无随机 | — | — |
| Q2 | `20260808` + 分片/试验索引 | M=2000 试验/φ, K=64 边 | 环境变量 `SHUMO_Q2_SOLID_SEED` |
| Q3 | `20260808` + 试验索引 | M=10000, K=32 边;假设一对照 M=2000 | 环境变量 `SHUMO_Q3_BASE_SEED` |
| Q4 搜索 | `42` | M=100 | 环境变量 `SHUMO_Q4_BASE_SEED` |
| Q4 独立复算/严格复核 | `20260808` | M=4000 | 同上(脚本内固定) |

注意:`run_refined_scan.py` 默认种子为 `20261808`(与其余 20260808 不一致),如需复现
`question4_refined_scan.json` 建议显式设置 `SHUMO_Q4_REFINE_SEED=20260808`。

## 6. 论文对应关系

| 论文 | 正式结果数据 | 配图 |
|------|-------------|------|
| 问题1报告_修订版 | Q1/results/question1_result.json | 图1~3:组1/2/3 接触网络与导电路径 |
| 问题2报告_完整实体正式版 | Q2/results/solid_boundary_comparison.json | 图4 导通概率曲线、图7 几何统计量、图8 跨壁比例诊断 |
| 问题3报告 | Q3/results/question3_solid_curve.csv, question3_solid_summary.json | 图1~5:夹逼曲线/临界放大/几何示意/敏感性/早停诊断 |
| 问题4报告 | Q4/results/question4_result.csv 等四件套 + strict_mixed_* | 图1 热图、图2 可行边界、图3 置信区间、图4 严格复核 |

另有 2 张过程流程图(仅 png,未编号入论文):`Q2/figures/问题2_单样本蒙特卡洛连通性判定流程.png`(单样本 MC 连通性判定流程)、`Q3/figures/Q3_流程图.png`(解题流程),作为过程示意参考。

## 7. 附件与论文 docs/ 的关系

- `reports/` 为仓库 `docs/` 下 4 份最终版论文的快照;论文其余过程文档(题目原文、
  旧版报告、建模说明稿)不随附
- AI 使用全程留存材料(交互记录、图配套信息、AI 工具表、声明草稿)位于仓库
  `docs/appendix/`,按组委会要求另行说明,不在本附件内

## 8. 注意事项

- **Q1 依赖赛题原始数据**:`run_question1.py` 读取赛题附件的 `附件.xlsx`
  (三个工作表 = 三组介质 A 轴线端点坐标,按工作表索引 0/1/2 读取)。该文件为赛题
  组委会统一提供的原始数据,不随本附件分发;运行时请将其放至附件上级目录 `attachment/` 下
  (即与 `Q1/` 平级的 `attachment/附件.xlsx`)
- 附件内文件名含中文,Windows/Linux 均兼容;请勿重命名,以免与论文图注失联
- `Q4/results/README.md` 为工作区原样复制,其中部分口径说明已过时(如
  `question4_unified_a_*` 文件已不存在、strict_mixed 复核规模描述以
  `Q4/figures/readme.md` 的 M=4000 口径为准);数据文件本身均为最终版本
- Q4 配图取自论文定稿图池(2026-08-09 定稿版),与工作区 `Q4/figures/` 中早期同名文件内容不同
