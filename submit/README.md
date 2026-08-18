# 提交包说明

## 附件（attachment/）

| 文件 | 用途 |
|------|------|
| `附件1.xlsx` | 赛题数据：4 个算例的目标点坐标与巡检等级（I 级 3 次 / II 级 2 次 / III 级 1 次） |
| `附件2.xlsx` | 飞行管制区域（问题 3 使用） |
| `result1.xlsx` | 赛题提供的结果表样例（问题 1 格式） |
| `result2.xlsx` | 赛题提供的结果表样例（问题 2 格式） |
| `result3.xlsx` | 赛题提供的结果表样例（问题 3 格式） |

## 结果表（根目录）

| 文件 | 对应问题 |
|------|----------|
| `result1.xlsx` | 问题 1：4 个算例的多无人机巡检路径方案（每个算例一个 sheet，行 = 无人机，列 = 按序访问的巡检点） |
| `result2.xlsx` | 问题 2：考虑任务均衡后的路径方案（sheet 结构同 result1） |
| `result3.xlsx` | 问题 3：叠加飞行管制区域后的路径方案（sheet 结构同 result1） |

## 代码与产物目录

| 目录 | 内容 |
|------|------|
| `Q1/`、`Q2/`、`Q3/` | 各问题求解代码（含主程序 `solve_qX.py`） |
| `templates/` | 复用的建模模板模块 |
| `outputs/workbooks/` | 求解档案（JSON 归档）、结果表工作副本、校验报告 |
| `outputs/workbooks/q1_solution_Case*.json` | 问题 1 各算例解档案 |
| `outputs/workbooks/q2/strict/` | 问题 2 正式归档 |
| `outputs/workbooks/q3/strict/` | 问题 3 正式归档（最终版） |
| `outputs/workbooks/q3/timetables/` | 问题 3 各算例详细时刻表 |
| `outputs/workbooks/*_verification_report.*` | 结果校验报告 |
| `figures/`、`plots/` | 论文配图 |
| `requirements.txt` | Python 依赖 |
