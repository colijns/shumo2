# 问题3 重跑计划（Q2 更新后同步热启动）

- **版本**: v1（2026-08-18）
- **上游**: `Q3/PLAN.md`（v1 实现计划，本文件是其补充执行计划）
- **触发**: Q2 增强邻域正式预算运行完成（`enhanced_run_20260817_e2000/`），Q3 热启动需要同步更新
- **决策来源**: grill-with-docs 流程，组长于 2026-08-18 确认（R1–R4）

---

## 1. 事实核查结论（已用当前代码实跑验证）

| 说法 | 判定 | 证据 |
|---|---|---|
| 上次 Q3 运行可保留 | ✅ | `outputs/workbooks/q3/strict/Case{1-4}.json` 全部 verified=true |
| 上次用第一版 Q2 路线 | ✅ | 热启动源 `outputs/workbooks/q2/strict/`（sha 05b8b01b / f1ba2902 与 Q3 manifest 记录一致，至今未变）；新版在 `enhanced_run_20260817_e2000/q2/strict/` |
| Case1 用旧基线 | ✅ | manifest: Case1 source=q2-strict，起点 (31255, 807) → 搜索后 (31255, 102) |
| Case2/3 回退贪心 | ✅ 结论 / ⚠️ 原因 | manifest: Case2/3 source="none" fallback="greedy"。**原因不是档案没读到，而是 Q2 路线在 Q3 禁飞规则下非法**（新旧档案均 INFEASIBLE，如 Case2 穿越 Z2 15:00–17:00，代码注释佐证） |
| Q2 优化了一轮 | ✅ | 新版严格档案与用户数字逐项一致：Case1 31255/30623/632、Case2 27058/27017/41、Case3 29303/29031/272、Case4 30158/30101/57（第三位是 Delta/s） |
| 最新 Q2 路线可作热启动 | ⚠️ 仅 Case1 | 实测：Case2/3 新版路线仍非法；Case4 合法但更差（S_max 36931 持平旧版、delta 3817、wait 20438 s）；Case1 合法 (31255/632) 但已被 Q3 现有结果 (31255/102) 支配 |
| Q3 现有结果优于一切 Q2 热启动 | ✅ | Case1: 102 < 632 < 807；Case4: 49 远优于 36931 起点；Case2/3 无 Q2 热启动可用 |

**核心结论**：重跑的主要收益来自「Q3 现有完整路线作为初始解继续搜索」（首次实现初始解 1）；Q2 新路线作为次级兜底（首次可用）。`build_initial_solutions` 已按词典序排序 + `local_search` 从 pool[0] 起步（first-improvement），把 Q3 现有路线放入池即自动获得「词典序不劣于现有结果」的保证。

---

## 2. 已确认决策（追加到 Q3/PLAN.md §1）

| ID | 决策点 | 采纳值 |
|----|--------|--------|
| R1 | 初始解池来源 | 两级：Q3 自身已验证档案（`q3/strict/Case{1-4}.json`，最高优先）→ Q2 strict 档案 → Q1 基线 → 贪心；全部按词典序排序后择优，扰动基于池首 |
| R2 | Q2 新路线处理 | 仍纳入池（仅 Case1 合法）；因排序在 Q3 档案之后，仅当 Q3 档案缺失/损坏时起兜底作用 |
| R3 | 旧版留存 | 重跑前把 `outputs/workbooks/q3/` 整体归档到 `outputs/workbooks/q3_baseline_20260817/`（含重写归档 manifest 相对路径），新版照常写原位 |
| R4 | 运行预算 | 每 Case 1800 s，顺序 Case2 → Case3 → Case4 → Case1（可中断，已完成 Case 落盘后可单独跳跑） |

---

## 3. 代码改动（Q3/）

### 3.1 `Q3/search.py` — 初始解池两级来源

1. 新增 `_q3_archive_routes(problem, repository_root) -> list[tuple[int, ...]] | None`：
   - 读 `outputs/workbooks/q3/strict/{case}.json`（verified 档案；不存在/无 `task_routes` → None）
   - `eval_solution(problem, routes)` 为 None → None（禁飞下不可能，防御性）
2. 修改 `build_initial_solutions`：
   - 收集顺序：Q3 档案路线 → Q2 热启动路线（现有 `_hot_start_routes`）→ Q1 基线 → 贪心变体
   - Q3 档案可用时：`pool = [q3_solution] + _eval_perturbations(q3_routes, k-1)`；Q2 路线合法也 append 进池（兜底候选，排序后不改变池首）
   - Q3 档案不可用：回退现有逻辑（Q2 → Q1 → 贪心）
   - 保持 `pool.sort(key=lex_key)` 与 fail-closed
3. 修改 `Q3/solve_q3.py::_hot_start_block`：增加 Q3 档案优先级分支（source=`q3-archive`，记录 archive_path/sha/metrics），其后才是现有 Q2/Q1/greedy 分支；acceptance 基线随之变为 Q3 现有结果（其结果必然不劣，fail-closed 自动满足）

### 3.2 测试（`tests/q3/test_search.py`）

- 现有 `test_hot_start_loads_real_case1_archive_readonly` 断言宽松（pool 非空 / S_max>0 / 4 条路线 / 合法性），不受影响
- 新增：
  - `test_q3_archive_seeds_pool_first`：Case1 的 pool[0] 词典序 == `q3/strict/Case1.json` 的 metrics（验证两级来源与排序）
  - `test_q3_archive_missing_falls_back_to_q2`：构造临时 repository_root（有 q2/strict 无 q3/），pool[0] 回到 Q2 热启动
- 跑 `python -m pytest tests/q3/ -v` 全绿后进入运行阶段

---

## 4. 执行步骤

### 4.1 归档旧版（保留记录）
```powershell
# 复制 q3 目录族到归档位置
Copy-Item -Recurse outputs/workbooks/q3 outputs/workbooks/q3_baseline_20260817
# 重写归档 manifest 的相对路径字段（outputs\workbooks\q3\... -> outputs\workbooks\q3_baseline_20260817\...）
python -c "import json,pathlib; p=pathlib.Path('outputs/workbooks/q3_baseline_20260817/reports/run_manifest.json'); m=json.loads(p.read_text(encoding='utf-8')); [c.update(archive=c['archive'].replace('workbooks\\\\q3\\\\','workbooks\\\\q3_baseline_20260817\\\\'),timetable=c['timetable'].replace('workbooks\\\\q3\\\\','workbooks\\\\q3_baseline_20260817\\\\')) for c in m['cases'].values()]; p.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')"
```

### 4.2 测试验证改动
```bash
conda activate math  # 或全路径 python
python -m pytest tests/q3/ -v
```

### 4.3 重跑（后台，约 2 小时）
```bash
python -m Q3.solve_q3 --case Case2 --time-budget 1800   # 30 min
python -m Q3.solve_q3 --case Case3 --time-budget 1800
python -m Q3.solve_q3 --case Case4 --time-budget 1800
python -m Q3.solve_q3 --case Case1 --time-budget 1800
python -m Q3.solve_q3 --all --skip-solved               # 复用已解档案，生成 result3.xlsx
python -m Q3.solve_q3 --verify                          # 独立验证，必须通过
```
每 Case 独立落盘（strict + timetable + manifest），可中断、可跳跑。

### 4.4 结果对比与报告
- 新旧对比表（每 Case：S_max / Δ / ΣS_k / W / L，旧值取归档目录，新值取新档案）
- 更新 `docs/问题3报告.md` 结果章节；`docs/问题3报告_整理版.md`（未跟踪，用户整理稿）是否同步由组长决定

### 4.5 提交（分步）
1. `feat(q3): seed initial pool with own verified archive, keep q2 as fallback`（代码 + 测试 + PLAN.md 决策记录 + RERUN_PLAN.md）
2. `data(q3): archive baseline run to q3_baseline_20260817`
3. `results(q3): rerun with q3-seeded warm start, adopt lexicographic best`（结果 + 报告）

---

## 5. 风险与回滚

| 风险 | 应对 |
|---|---|
| Case2/3 从 Q3 现有起步仍无改善 | first-improvement 保证不劣（结果 ≥ 初始解词典序）；持平则如实记录 |
| 确定性搜索路径与上次不同 | seed=42 保持；从新起点（Q3 现有）出发，算子序列同但起点不同，可能找到更优 |
| 某 Case 运行中崩溃（fail closed） | 该 Case 不写盘，旧档已在归档目录，可单独重跑该 Case |
| 2 小时长任务 | 后台运行 + 日志监控；逐 Case 落盘可中断续跑 |
| 独立验证失败 | 不提交、不写 result3.xlsx；定位后重跑对应 Case |
