# 问题1 采用 OR-Tools Routing 作主求解器

赛题 A 问题 1 是带单机 9 h 时限的 min-max 多旅行商问题（最大算例 182 个任务），需要高质量启发式解。我们决定安装并使用 Google OR-Tools Routing（9.15），首解策略 PARALLEL_CHEAPEST_INSERTION、局部搜索 GUIDED_LOCAL_SEARCH、两阶段词典序搜索（先最小可行 N，再压 Tmax）。

## Considered Options

- **OR-Tools Routing（采纳）**：GLS 是该问题规模下最强的现成开源启发式之一；建模笔记（`docs/问题1.md` §6）即按其能力撰写。
- **PuLP + CBC（已安装，弃用）**：min-max mTSP 的 MILP 含大量 0-1 变量与子回路消除，182 任务规模下 CBC 在比赛时限内既给不出好解，也证明不了不可行。
- **自写启发式（弃用）**：templates/ 仅有模拟退火 TSP 单机版，扩到多机 + 时限 + 禁止相邻约束工作量大、解质量预期更差。

## Consequences

- 新增依赖：requirements.txt 已追加 ortools（math 环境安装 9.15.6755）。
- **无随机种子**：OR-Tools 局部搜索不暴露 seed 接口，结果随时间预算与机器波动。复现策略以**解档案为锚点**：论文数字一律取 `outputs/workbooks/q1_solution_Case*.json`，`solve_q1.py --verify` 可在任何机器从档案 + 原始附件独立复算，不承诺重跑求解得到相同航线。
- 解为启发式解：仅当 N_feasible = N_LB 时宣称无人机数严格最少（下界相等即证明），Tmax 一律不宣称全局最优。
