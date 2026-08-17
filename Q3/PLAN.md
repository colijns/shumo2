# 问题3 实现计划

- **版本**: v1（2026-08-17）
- **上游建模方案**: `docs/问题3.md`（口径权威；本文件与其冲突时，以建模方案为准）
- **决策来源**: grill-with-docs 流程，组长于 2026-08-17 确认"均按推荐方案"（Q1–Q9、N1–N3 共 12 项）
- **关联 ADR**: `docs/adr/0002-deterministic-local-search-for-q3.md`（弃用 OR-Tools/shapely/SA 的完整理由）
- **边界声明**: 本计划是**实现计划**，不含问题3数值结果；六段式报告另走 math-modeling-report skill，论文配图另走 competition-record skill。按本计划执行前不填写任何问题3数值。

---

## 1. 已确认决策记录

| ID | 决策点 | 采纳值 |
|----|--------|--------|
| Q1 | 建模口径 | 推荐口径：固定 4/2/5/4 机群、不设 9 h 截止、允许必要安全等待；车辆数/时限/目标词典序全部配置化，§7.3 备选口径只改配置不动代码 |
| Q2 | Case4-Z8（17:00–17:00） | 零时长：数据加载时记录警告并跳过，不生成任何时空约束 |
| Q3 | 安全裕度 | η = 0.01 km（附件2 半径 3.1–6.3 km，η 相对可忽略） |
| Q4 | 落盘位置 | 本文件 `Q3/PLAN.md` |
| Q5 | 计划粒度 | 代码手可直接照做：结构/数据结构/函数签名/伪代码/测试清单/验收/里程碑/模型合同 |
| Q6 | 几何内核 | 手写 numpy 闭式几何（math 环境无 shapely，且时变判定本就须手写） |
| Q7 | 上层搜索 | 确定性多起点局部搜索；不引入 SA、不引入 OR-Tools |
| Q8 | 运行预算 | 连续 3 轮全邻域扫描无改进 + 每 Case 30 min 硬上限（4 Case 合计 ≤ 2 h） |
| Q9 | 事件时刻缓存 | 不进 v1，正确性优先，列 v2 方向 |
| N1 | 代码组织 | 对齐 Q2 包结构（不可变 dataclass、原子落盘、契约哈希、独立校验器）+ `tests/q3/` |
| N2 | 热启动轨道 | 仅 strict 轨；缺失/校验失败回退 Q1 基线 → 贪心，manifest 记录回退来源 |
| N3 | OR-Tools | 不引入；几何与图搜索用 numpy + 手写 Dijkstra |

**已固定假设**（无异议）：Q3 档案 schema 版本化（`q3-solution-v1`）；输出目录族 `outputs/workbooks/q3/{strict,timetables,reports,checkpoints}/`；逐航段时间表按建模方案 §10 的 13 字段 CSV；result3.xlsx 用 pandas+openpyxl 原子写；独立校验器共享几何原语但独立重放时间轴。

---

## 2. 目标与边界

### 2.1 求解目标

固定各 Case 机群（Case1=4 / Case2=2 / Case3=5 / Case4=4），词典序最小化：

```
min lex ( S_max,  Δ,  Σ_k S_k,  W,  L )
```

- `S_max`：系统完成时间（max S_k，整数秒）
- `Δ = S_max − S_min`：工作负载极差（整数秒）
- `Σ_k S_k`：总任务时长（整数秒）
- `W`：总必要等待时间（整数秒）
- `L`：总飞行距离（km）

### 2.2 非目标（v1 不做）

- 9 h 截止与机群重求（§7.3 备选口径，仅留配置开关）
- 事件时刻缓存（建模方案 §9.3，v2 方向）
- SA/元启发式接受准则、OR-Tools 求解
- 六段式报告、论文配图

### 2.3 继承 Q1/Q2 的规则（建模方案 §2）

1. I/II/III 级巡检点展开为 3/2/1 个独立任务，单次服务 300 s；
2. 不同无人机同时到达同一点，分别计为一次巡检（无互斥）；
3. 同一无人机同一点相邻不可重复计（相邻相同 task 对应的 Point_ID 禁排），再次巡检前必须先离开；
4. 全部任务完成，`served == M`（覆盖 task_id 1..M 恰好一次）；
5. 速度 55 km/h，坐标 1 单位 = 0.1 km；
6. 全部无人机 8:00（t=0）自基地 (0,0) 出发，完成后返回基地；
7. 飞行、服务、等待均用整数秒；飞行单段时长 = ceil(距离/速度) 取整（继承 Q1/Q2）。

---

## 3. 数据流与目录

### 3.1 输入

| 输入 | 路径 | 用途 |
|------|------|------|
| 巡检任务 | `attachment/附件1.xlsx`（Case1~4 sheet） | 经 `Q2.q1_adapter.load_problem` 冻结（sha256 + 契约哈希），保证 task_id 展开顺序与 Q2 档案一致 |
| 禁飞区 | `attachment/附件2.xlsx`（Case1~4 sheet） | 6 字段：Zone_ID, Center_X, Center_Y, Radius, Start_Time, End_Time |
| 热启动 | `outputs/workbooks/q2/strict/Case{1..4}.json` | 问题2 通过独立校验的发布档案 |
| 回退链 | `outputs/workbooks/baseline_20260816/q1_solution_Case*.json` | strict 档案缺失/校验失败时 |

### 3.2 输出

| 输出 | 路径 | 格式 |
|------|------|------|
| Q3 档案 | `outputs/workbooks/q3/strict/Case{1..4}.json` | `q3-solution-v1`（§7.2） |
| 逐航段时间表 | `outputs/workbooks/q3/timetables/Case{1..4}_timetable.csv` | §7.3 的 13 字段，UTF-8 |
| 提交表格 | `outputs/workbooks/result3.xlsx` | Case1~4 四个 sheet，每行一架无人机，原始 Point_ID 序列（不填基地 0） |
| 运行清单 | `outputs/workbooks/q3/reports/run_manifest.json` | `q3-manifest-v1`：各 Case 档案 sha256、热启动来源、配置快照、校验状态 |
| 校验报告 | `outputs/workbooks/q3/reports/verify_report.{md,json}` | 独立重放结果（九条验证逐项） |
| 搜索中间态 | `outputs/workbooks/q3/checkpoints/` | 每次词典序改进的完整可行方案（原子写） |

### 3.3 数据流

```text
附件1.xlsx ─┐
附件2.xlsx ─┴─> load_problem (复用 Q2.q1_adapter，冻结+sha256) ─> ProblemData
Q2 strict 档案 (q2/strict/Case*.json) ─> 热启动 task_routes（sha256 校验）
Q1 基线 (baseline_20260816) ─> 回退链（manifest 记录）
              │
              ▼
   search.py 上层（多起点 + 局部搜索，词典序接受）
              │ 每个候选：受影响路线全量重算（未受影响路线复用缓存时间表）
              ▼
   safe_path.py 下层（earliest_safe_travel / earliest_safe_service_completion）
              │ 几何原语
              ▼
   geometry.py（手写 numpy：线段-圆、切线、可见图、Dijkstra）
              │
              ▼
   checkpoints/（改进即存）→ 最终词典序最优候选
              │
              ▼
   verify.py 独立重放校验（fail closed，未通过不得发布）
              │ 通过
              ▼
   strict 档案 + timetable CSV + result3.xlsx + manifest + 校验报告
```

---

## 4. 术语表（实现层唯一解释，与建模方案 §4–§7 对齐）

| 术语 | 精确定义 |
|------|----------|
| **禁飞区 z** | 闭圆盘 `(C_z, r_z)` × 生效区间 `[s_z, e_z]`（相对 8:00 的整数秒）；`C_z`、`r_z` 以 km 计（坐标 ×0.1） |
| **安全圆盘 Ω_z** | 半径 `r_z + η` 的闭圆盘 |
| **时空冲突** | 轨迹某点位于安全圆盘内（含边界）**且**该时刻落在 `[s_z, e_z]` 内；空间相交但时间不重叠、或时间重叠但空间不进入，均不构成冲突 |
| **必要等待** | 仅由禁飞约束迫使的等待；固定任务序列必须取最早可行时间表，禁止为改善 Δ 人为增加短路线等待 |
| **直接飞行 direct** | 直线航段全程无时空冲突 |
| **等待后直飞 wait_direct** | 在安全位置等待至直线航段可完整通过 |
| **绕飞 detour** | 对单个圆盘的切线—安全圆弧—切线路径 |
| **可见图路径 visibility** | 多圆并集障碍下的边界图最短安全路径 |
| **下层求解器** | `earliest_safe_travel`（E_{uv}(t)）与 `earliest_safe_service_completion`（Φ_{uv}(t)） |
| **上层求解器** | 任务分配与访问顺序优化（多起点局部搜索） |
| **重放 replay** | 由 task_routes + 求解函数重建完整时间表 |
| **独立校验** | 不调用下层求解函数的时空重放（只共享几何原语） |
| **词典序比较** | 五元组 (S_max, Δ, ΣS_k, W, L) 从左到右首次分出大小即定 |

---

## 5. 时间与数值约定（实现必须严格遵守）

- **时间轴**：t=0 对应 8:00，一切时刻为相对 8:00 的**整数秒**；禁飞时刻 "HH:MM" 换算为整数秒。
- **几何轴**：位置、距离、交点参数均为**连续实数**；判定"何时进入/离开圆盘"用连续区间，再与整数秒的生效区间做闭区间重叠判定。
- **飞行时长**：单段 `ceil(distance_km / 55.0 * 3600)` 整数秒（含绕飞弧长）。
- **裕量与容差**：
  - η = 0.01 km（安全圆盘加宽）；
  - ε_arc = 1e-6 km：绕飞圆弧构造半径 `r_z + η + ε_arc`，避免精确贴着安全圆边界（闭圆盘边界本身即冲突）；
  - ε_ver = 1e-9 km：校验器冲突判定用 `dist ≤ r_z + η − ε_ver`（保守方向，1e-6 ≫ 1e-9 保证求解器输出必过校验）。
- **词典序末项**：L 以 km 计，比较时四舍五入到 1e-6 km，容差内相等视为平局。
- **闭区间语义**：服务区间 `[A, A+300]`、等待区间 `[t_a, t_b]`、生效区间 `[s_z, e_z]` 均为闭区间；端点重合即冲突（η 与 ε 已吸收浮点误差）。

---

## 6. 包结构与文件清单

```
Q3/
├── __init__.py        # 导出领域值与模块面
├── domain.py          # 不可变领域值 + 常量 + 配置（<150 行）
├── io.py              # 附件解析、原子 JSON/CSV/xlsx 落盘、契约哈希（<200 行）
├── geometry.py        # 手写几何原语（<250 行）
├── safe_path.py       # 下层求解器 E/Φ（<300 行）
├── search.py          # 上层多起点局部搜索（<400 行）
├── verify.py          # 独立时空校验器（<350 行）
└── solve_q3.py        # CLI 入口：--case / --all / --verify / --promote（<150 行）

tests/q3/
├── conftest.py        # Mini 合成算例夹具（3~4 点 + 2 禁飞区，不依赖 attachment）
├── test_domain.py     # 常量、时间换算、Z8 警告、配置
├── test_geometry.py   # 几何原语全覆盖
├── test_safe_path.py  # 下层求解器场景测试
├── test_search.py     # 热启动、邻域合法性、词典序接受、确定性
├── test_verify.py     # 校验器正例 + 逐类违规注入
└── test_io.py         # 档案/CSV/xlsx 往返、原子写、契约哈希
```

**依赖方向**：`solve_q3 → search → safe_path → geometry / domain`；`verify → geometry / domain / io`；`io → domain`。**禁止** `verify` import `safe_path` 或 `search`（独立校验的硬约束）。

**复用 Q2（只读）**：`from Q2.q1_adapter import load_problem, FLEET_SIZE_BY_CASE`、`from Q2.domain import LEVEL_VISITS, SPEED_KMH, UNIT_KM, SERVICE_S`。任务展开与输入冻结不重复实现，保证 task_id 与 Q2 档案逐位一致。运行方式：仓库根 `conda activate math && python -m Q3.solve_q3 --case Case1`。

---

## 7. 核心数据结构

### 7.1 domain.py

```python
# 常量（继承 Q2.domain）
SPEED_KMH, UNIT_KM, SERVICE_S, LEVEL_VISITS  # 复用 Q2，不重定义
SAFETY_MARGIN_KM   = 0.01   # η
EPS_ARC_KM         = 1e-6   # 绕飞弧构造裕量
EPS_VER_KM         = 1e-9   # 校验容差
START_T_S          = 0      # 8:00

@dataclass(frozen=True)
class NoFlyZone:
    zone_id: str
    cx_km: float; cy_km: float
    radius_km: float
    start_s: int; end_s: int
    def active(self, t_s: int) -> bool: ...          # start_s <= t <= end_s
    def safe_radius(self) -> float: ...              # radius_km + SAFETY_MARGIN_KM

@dataclass(frozen=True)
class Task:
    task_id: int          # 1..M，展开顺序与 Q2 一致
    point_id: int         # 原始 Point_ID
    x_km: float; y_km: float
    level: str            # "I"|"II"|"III"

@dataclass(frozen=True)
class ProblemData:
    case: str
    tasks: tuple[Task, ...]          # M 个，task_id = index+1
    zones: tuple[NoFlyZone, ...]     # 已剔除零时长区（记入 warnings）
    fleet_size: int
    warnings: tuple[str, ...]        # 如 "Case4-Z8: zero-duration zone skipped"
    input_sha256: str

@dataclass(frozen=True)
class Config:
    fleet_size: dict[str, int]        # {"Case1":4,"Case2":2,"Case3":5,"Case4":4}
    eta_km: float = 0.01
    seed: int = 42
    time_budget_s_per_case: int = 1800
    stall_rounds: int = 3             # 连续无改进全邻域轮数
    nine_hour_cap_s: int | None = None   # §7.3 备选口径设为 32400
    lex_first_is_N: bool = False          # §7.3 备选口径设为 True
    allow_empty_routes: bool = False      # 默认与 Q1/Q2 一致：禁空路线

@dataclass(frozen=True)
class SegmentRecord:                 # 时间表一行 = 一个航段
    from_id: int                     # 0=基地，否则 task_id
    to_id: int
    path_type: str                   # "direct"|"wait_direct"|"detour"|"visibility"
    depart_s: int                    # 出发时刻（整数秒）
    arrive_s: int                    # 到达时刻（整数秒）
    wait_s: int                      # 出发前必要等待
    distance_km: float
    affected_zones: tuple[str, ...]  # 本次通行避让/等待涉及的 zone_id

@dataclass(frozen=True)
class UAVSchedule:
    uav_id: int
    task_route: tuple[int, ...]              # task_id 序列（不含基地）
    segments: tuple[SegmentRecord, ...]      # 含最后返航段
    service_intervals: tuple[tuple[int, int], ...]  # (start_s, end_s)，end=start+300
    S_k_s: int                               # 最终返航到达时刻
    flight_s: int; wait_s: int; distance_km: float

@dataclass(frozen=True)
class SolutionMetrics:               # 词典序五元组
    S_max_s: int; S_min_s: int; delta_s: int
    sum_T_s: int; total_wait_s: int; total_distance_km: float
    def lex_key(self) -> tuple: ...  # 按 §5 舍入规则

@dataclass(frozen=True)
class Solution:
    problem: ProblemData
    schedules: tuple[UAVSchedule, ...]
    metrics: SolutionMetrics
    def freeze(self) -> dict: ...    # q3-solution-v1 JSON
```

### 7.2 Q3 档案 schema（`q3-solution-v1`）

```json
{
  "schema_version": "q3-solution-v1",
  "case": "Case1",
  "fleet_size": 4,
  "hot_start": {
    "source": "q2-strict",
    "archive_path": "outputs/workbooks/q2/strict/Case1.json",
    "archive_sha256": "<64hex>",
    "fallback": null
  },
  "config": {"eta_km": 0.01, "seed": 42,
             "time_budget_s": 1800, "nine_hour_cap_s": null},
  "input": {"attachment_sha256": "<64hex>", "problem_contract_sha256": "<64hex>"},
  "task_routes": [[55, 22, 4, ...], [...], ...],
  "schedules": [
    {"uav_id": 0,
     "segments": [
       {"from_id": 0, "to_id": 55, "path_type": "direct",
        "depart_s": 0, "arrive_s": 842, "wait_s": 0,
        "distance_km": 12.86, "affected_zones": []}
     ],
     "service_intervals": [[842, 1142], ...],
     "S_k_s": 31255, "flight_s": 26065, "wait_s": 120, "distance_km": 310.2}
  ],
  "metrics": {"S_max_s": 31255, "S_min_s": 30448, "delta_s": 807,
              "sum_T_s": 123245, "total_wait_s": 480, "total_distance_km": 1240.9,
              "mean_T_s": {"numerator": 123245, "denominator": 4}}
}
```

### 7.3 逐航段时间表 CSV（13 字段，UTF-8，一行一航段）

```text
case,uav_id,segment_id,from_id,to_id,path_type,
depart_time,arrive_time,service_start,service_end,
wait_seconds,distance_km,affected_zones
```

- `from_id/to_id`：0=基地，否则 task_id（与档案一致）；`result3.xlsx` 用原始 Point_ID，写入时经 `ProblemData` 映射转换；
- 返航段（to_id=0）的 `service_start/service_end` 留空；
- `affected_zones` 为逗号分隔的 zone_id，空则空字符串。

### 7.4 关键实现规则（R1–R7）

- **R1 航段冲突**：对航段直线参数化 `P(t) = A + (t−t0)/q (B−A)`，解线段与各安全圆盘的相交参数区间（连续实数），换算成时刻区间后与 `[s_z, e_z]` 做闭区间重叠判定。
- **R2 服务冲突**：`[A, A+300]` 与"u 所在安全圆盘"的生效区间相交即冲突；u 是否位于圆内按 `dist ≤ r_z + η` 判定。
- **R3 等待安全**：等待位置 P 在整个等待区间内对每个生效禁飞区满足 `dist(P, C_z) > r_z + η`；**等待只允许发生在基地或已服务的巡检点**（当前节点 u）。
- **R4 弧段冲突**：绕飞圆弧（半径 `r_arc = r_z + η + ε_arc`）与其他圆盘的交集用圆-圆相交闭式解再裁剪到弧参数区间；对本圆盘自身：半径恒大于安全半径，永不冲突。
- **R5 时间轴连续**：每段 `arrive_s = depart_s + 飞行时长`；有等待时 `下一段 depart_s = 上段 arrive_s + wait_s`；服务 300 s 计入 `service_intervals`；最后返航 `arrive_s = S_k`。
- **R6 词典序**：按 §5 舍入规则生成 `lex_key`，所有方案（含跨 Case 复盘）统一比较。
- **R7 fail closed**：任何候选路径若全部候选策略均冲突，该候选路线判为不可行并拒绝（上层算子不得接受）；校验失败不得发布。

---

## 8. 算法规格

### 8.1 geometry.py — 几何原语（全部纯 numpy，无外部依赖）

```python
def segment_circle_interval(p, q, c, r) -> tuple[float, float] | None:
    """线段 [p,q] 与圆 (c,r) 的相交参数区间（t∈[0,1]，相对航段起点）。
    无交点返回 None。解一元二次方程：|p + t(q−p) − c|² = r²。"""

def tangent_points(p, c, r) -> tuple[Point, Point]:
    """点 p 在圆 (c,r) 外时，两条外切线在圆上的切点。"""

def detour_path(p, q, c, r) -> list[Point]:
    """单圆绕飞：p→T1 切线段 + T1→T2 安全圆弧（选短弧，弧半径 r）+ T2→q 切线段。"""

def arc_intersection_with_circle(c1, r1, theta1, theta2, c2, r2) -> list[float]:
    """圆心 c1、半径 r1 的圆弧（角区间）与圆 (c2,r2) 的交集角区间列表。"""

def visibility_path(p, q, disks: list[tuple[Point, float]]) -> list[Point] | None:
    """可见图最短安全路径。节点 = {p, q} ∪ 各圆外切点 ∪ 重叠圆对的两个边界交点；
    边合法 ⇔ 线段不与任何圆盘相交（margin=EPS_VER_KM）；Dijkstra 求最短路。
    多圆并集天然由逐圆判交实现（线段穿过并集 ⇔ 穿过至少一个圆盘）。"""

def path_length(points: list[Point]) -> float:
    """折线+圆弧路径总长（km）。圆弧部分按弧长公式。"""
```

**复杂度**：可见图节点数 O(n²)（n=障碍圆数≤8），边判交 O(n)，Dijkstra O(V²)；附件规模下每次调用 <1 ms。

### 8.2 safe_path.py — 下层求解器

```python
def spatiotemporal_safe(path: Path, depart_s: int, zones) -> bool:
    """沿 path 逐段累积整数秒得到各段 [t0, t1]；对每个在 [t0, t1] 内生效的禁飞区，
    按 R1/R4 求交集区间并做闭区间重叠判定；任一段冲突即 False。"""

def earliest_safe_travel(problem, u: Task|Base, v: Task|Base,
                         depart_s: int) -> SegmentRecord:
    """E_{uv}(t)。候选枚举，返回 arrive_s 最早者（tiebreak: wait_s 升序, distance 升序）：
    1) direct：直线无冲突 → 直接采纳；
    2) wait_direct：枚举安全等待时长 w（事件驱动候选集，见下），在 u 处等待后直飞；
    3) detour：对每个"安全圆盘与直线相交 且 生效区间与通行窗口重叠"的圆盘 z，
       构造 tangent—arc—tangent 路径，整体时空校验；
    4) visibility：障碍集 O = 在 [depart_s, depart_s+BOUND] 内任一时刻生效的圆盘
       （BOUND = direct_time*4 + 3600，保守上界；只影响代价不影响安全，
        因为避开未生效圆盘至多绕远，最终仍以整体时空校验为准）；
    全部候选冲突 → 返回 None（fail closed）。
    记录 path_type / wait_s / distance_km / affected_zones。"""

def earliest_safe_service_completion(problem, prev, u: Task,
                                     depart_s: int) -> tuple[SegmentRecord, int, int]:
    """Φ_{uv}(t) = 旅行 + 服务联合。步骤：
    1) travel = earliest_safe_travel(prev, u, depart_s)；A = travel.arrive_s；
    2) 若 [A, A+300] 与含 u 的生效禁飞区冲突（R2）：
       不能原地等待（u 在生效圆内等待不安全，R3）；
       改在 prev 处延迟出发：事件驱动延迟候选
       w ∈ {0} ∪ {e_z − A_direct(w) + 1}（对每个与当前服务窗冲突的圆盘 z，迭代至可行）；
    3) 仍不可行 → None（fail closed）；
    4) 返回 (travel, A, A+300)。"""

def event_delays(problem, zones, now_s) -> list[int]:
    """事件驱动等待候选：{0} ∪ 各冲突圆盘的 (e_z − now_s + 1)、
    (e_z − now_s + 1 + 300 提前量变体)；排序去重。v1 不做连续优化，仅枚举事件边界。"""

def eval_route(problem, route: tuple[int, ...]) -> UAVSchedule | None:
    """从 t=0、基地出发，逐任务调用 Φ 折半递推（prev 更新为当前任务、depart 更新为服务结束），
    最后 earliest_safe_travel(prev, base, depart) 得 S_k。
    任一环节 None → 整条路线不可行。"""

def eval_solution(problem, task_routes: list[tuple[int, ...]]) -> Solution | None:
    """逐路线调用 eval_route，聚合 SolutionMetrics。"""

def base_safety_assert(problem) -> None:
    """断言 t=0 时基地不在任何生效禁飞圆内（四 Case 均成立：Case2 基地在 Z3 内
    但 Z3 于 10:30 生效）。不成立即硬错误。"""
```

**事件驱动延迟说明**（`wait_direct` / `Φ` 共用）：等待时长只枚举"禁飞区边界时刻 − 当前相关到达时刻 + 1"等事件值，保证不遗漏"等到某区结束即可通过"的最早时机；不做连续优化是正确性优先的刻意选择，v2 可改为二分。

### 8.3 search.py — 上层求解器

```python
def build_initial_solutions(problem, rng_hashed, k: int) -> list[Solution]:
    """多起点池（按优先级）：
    1) 热启动：Q2 strict 档案 task_routes（sha256 校验）→ eval_solution；
    2) 回退链：Q1 基线路线（若 1 缺失/不可行）；
    3) 贪心：确定性最近邻构造 ×3 变体（近邻/按等级优先/随机贪心，子种子哈希派生）；
    4) 扰动：对 1) 施加 K 次 relocate/swap 扰动（每个 seed 派生）；
    全部不可行 → 报错退出（fail closed）。"""

def moves(solution, problem) -> Iterator[Solution]:
    """邻域算子（每次生成一个新候选，不可变）：
    - two_opt：单路线 2-opt（逆转子序列）；
    - relocate：跨路线搬移单任务到最佳插入位；
    - swap：跨路线交换两任务；
    - block：跨路线搬移连续任务块（长度 2~4）；
    - destroy_repair：破坏（移除 q=3~5 个任务，优先最长路线）+ 贪心最优位重插。
    全部经 legality 检查后 eval（受影响路线全量重算，其余复用缓存时间表）。"""

def legality(problem, task_routes) -> bool:
    """served==M（1..M 恰好一次）；相邻相同 Point_ID 禁排；
    非空路线（Config.allow_empty_routes=False）；
    §7.3 备选口径下 S_k ≤ 32400。"""

def local_search(problem, config) -> Solution:
    """主循环：
    best = 初始池最优（lex_key）
    while 未连续 stall_rounds 轮全邻域无改进 且 未超 time_budget_s_per_case:
        按算子顺序做一轮完整邻域扫描（two_opt → relocate → swap → block → destroy_repair）
        每发现改进：更新 best、checkpoint 原子落盘、重置 stall 计数
    return best"""

def solve_case(case: str, config: Config) -> Solution:
    """加载 ProblemData → 初始池 → local_search → 返回 best（校验在发布侧）。"""
```

**确定性要求**：全局 `seed=42`；一切随机性经 `hash((seed, case, stage, i))` 派生子种子，禁止使用全局 RNG 状态，保证任意机器复现。

### 8.4 verify.py — 独立时空校验器

```python
def replay_segment(problem, seg: SegmentRecord) -> list[Sample]:
    """按 path_type 重建几何（不信任档案中的 distance_km）：
    direct/wait_direct → 直线；detour → 重新计算切线与弧；visibility → 重新构图求解。
    输出稠密采样点（线段 50 点/段，弧 100 点/段）。"""

def verify_archive(problem, archive: dict) -> list[str]:
    """返回违规列表（空=通过）。九条验证（建模方案 §10）：
    1. served==M，task_id 覆盖 1..M 恰好一次（不重不漏）；
    2. 每条任务路线及实际航迹从基地出发并返回基地；
    3. 同一路线不存在相邻相同 Point_ID；
    4. 逐航段采样重放：直线/折线/圆弧任意时刻不进入生效禁飞区（R1/R4，ε_ver 判定）；
    5. 每个 300 s 服务区间与等待区间满足 R2/R3；
    6. 时间轴连续（R5 逐段等式校验）；
    7. S_max/S_min/Δ/总等待/总距离由采样与事件独立复算，与档案 metrics 一致；
    8. 档案 task_routes ↔ timetable CSV ↔ result3.xlsx 三者逐格一致；
    9. 配置开启 nine_hour_cap 时，每架 S_k ≤ 32400。"""

def verify_all(cases, config) -> None:
    """读 run_manifest.json 逐 Case 校验；任何违规 → 写 verify_report 并返回非零退出码。"""
```

**独立性声明**：`verify.py` 只 import `geometry` 的纯几何原语（线段-圆区间、切线、弧交点），**禁止** import `safe_path`/`search`；采样重放按路径类型独立重建，与求解器共享的仅是几何事实而非调度逻辑。

---

## 9. 配置

`domain.py::DEFAULT_CONFIG` 为 v1 默认（§7.1 的 Config）。**§7.3 备选口径切换表**（只改配置，不改代码）：

| 口径 | nine_hour_cap_s | lex_first_is_N | fleet_size | 备注 |
|------|-----------------|----------------|------------|------|
| 推荐（v1 默认） | None | False | 4/2/5/4 | 词典序 (S_max, Δ, ΣS_k, W, L) |
| 继承 9 h 重求机群 | 32400 | True | 从 Q1 下界逐档测试 | 目标首项改为 N；搜索失败不可证明较小机群不可行 |

---

## 10. 测试清单（tests/q3/，覆盖率 ≥80%）

| 文件 | 用例（对应建模方案 §9.1 优先级） |
|------|--------------------------------|
| test_domain.py | HH:MM→秒换算（含 17:00→61200）；Z8 零时长跳过后 warning 记录；半径 η 换算；Config 冻结不可变 |
| test_geometry.py | 线段-圆：无交/相切/穿越/线段在内/圆在内五类；切点计算；弧-圆交；可见图：直通、单圆绕、双圆重叠并集（模拟 Case2 Z1/Z2）、三圆链式遮挡；Dijkstra 最短路正确性 |
| test_safe_path.py | 直线无冲突直达；穿越生效区→detour；等待后直飞优于绕飞（构造场景断言 wait_direct 被选中）；点内服务冲突→Φ 在 prev 延迟出发至窗口外；等待位置在生效圆内→禁止原地等；Case2 式基地在圆内（10:30–13:00 不可返航落基地）；重叠圆盘并集绕行；ε_arc 路径过 ε_ver 校验 |
| test_search.py | strict 档案热启动装载（真实 Case1 档案，只读）；served==M 与相邻同点禁排的算子后置校验；词典序接受（构造 (S_max,Δ) 交叉优劣样例）；固定 seed 两次运行逐位一致；stall/预算停止触发 |
| test_verify.py | 合法 Mini 方案九条全过；逐类注入违规：航段穿生效区 / 服务窗冲突 / 等待进圆 / 时间轴断裂 / metrics 篡改 / xlsx 与档案不一致 / served≠M / 相邻同点 / 9h 超限——各断言对应违规被检出 |
| test_io.py | 档案 JSON 往返一致；原子写（目标文件不残留半截）；契约哈希不匹配拒绝；result3.xlsx 写读回逐格一致；timetable CSV 13 字段与 schema 校验 |

**Mini 合成算例**（conftest.py）：4 个巡检点 + 2 个禁飞区，手工可算的最早到达时刻作为断言真值，全部下层测试以真值断言，不搞自洽式验证。

---

## 11. 验收标准

发布前必须全部满足（= 建模方案 §10 九条 + 下列工程门槛）：

1. 九条验证项在 `verify.py` 全过（`solve_q3 --verify` 退出码 0）；
2. `tests/q3/` 全绿且覆盖率 ≥80%（`pytest tests/q3 --cov=Q3`）；
3. 四 Case 均在 30 min/Case 预算内产出词典序不劣于热启动基线的方案（热启动 eval 值作为基线存档于 manifest）；
4. 全部随机性由 seed=42 + 哈希派生，重复运行逐位一致；
5. 输出四件套（strict 档案 / timetable CSV / result3.xlsx / manifest）齐全且互相一致；
6. `attachment/result3.xlsx` 为空白模板，**不得**被覆盖；真实产出写入 `outputs/workbooks/result3.xlsx`。

---

## 12. 里程碑

| 里程碑 | 交付物 | 验收门 |
|--------|--------|--------|
| **M0 脚手架** | Q3 包骨架、domain.py、io.py、Mini 夹具、Z8 警告 | test_domain / test_io 绿 |
| **M1 几何** | geometry.py 全部原语 | test_geometry 绿（含重叠并集、三圆链） |
| **M2 下层** | safe_path.py（E/Φ/事件延迟） | test_safe_path 绿；**先做 Case2 时空正确性验证**（点内服务与重叠圆盘最多，建模方案 §9.2 优先级） |
| **M3 上层** | search.py（热启动→局部搜索→checkpoint） | Case1 端到端：产出词典序不劣于热启动的方案，seed 复现 |
| **M4 校验与输出** | verify.py、result3.xlsx、timetable、manifest | test_verify 绿；Case1 四件套一致性自检 |
| **M5 全量运行** | 四 Case 预算内求解 + 独立校验 + 归档 | §11 全部验收标准 |

**顺序约束**：M0→M1→M2→M3→M4 严格串行；M5 依赖 M4 全过。每个里程碑完成后跑全量 `pytest tests/ -q`（含 Q2 既有测试）确认未破坏接口。

---

## 13. 风险与回退

| 风险 | 缓解/回退 |
|------|-----------|
| Case2 复杂（139 任务/2 机/重叠圆/基地在 Z3 内） | M2 后优先 Case2 正确性验证；上层若收敛慢，先以热启动改进版交付，再继续优化 |
| 保守障碍窗口（BOUND）导致过度绕飞、T_max 虚高 | v1 接受（安全优先）；v2 缩小窗口至通行时间邻域 |
| 事件驱动延迟遗漏中间可行时机 | 事件候选集由圆盘 s/e 边界全覆盖生成；若复盘发现遗漏，v2 二分细化 |
| strict 档案 sha256 校验失败 | 回退链 Q1 基线→贪心，manifest 记录 `fallback` 字段 |
| Z8 口径变更（改为瞬时禁飞） | 仅 `verify` 增补一条分支 + io 解析调整，架构不动 |
| §7.3 备选口径启用 | 纯配置切换（§9 表），搜索失败不作为较小机群不可行证明 |
| 运行超预算 | checkpoint 机制保证随时可停可取（最优 checkpoint 即交付） |
| 时间表与档案不一致 | verify.py 第 8 条逐格比对，发布前强制 |

---

## 14. 模型合同（交付代码手的完整契约）

建模方案 §11 全文即为合同主体，此处增补实现层签名契约：

```text
【巡检输入】attachment/附件1.xlsx（经 Q2.q1_adapter.load_problem 冻结，task_id 与 Q2 一致）
【禁飞输入】attachment/附件2.xlsx（HH:MM → 相对 8:00 整数秒；零时长区警告跳过）
【起始时刻】08:00 记为相对时间 0，内部一律整数秒
【默认车辆】4/2/5/4（Config.fleet_size；§7.3 切换不改代码）
【任务规则】继承 Q1/Q2：展开 3/2/1、同时巡检、同点相邻禁排、served==M
【禁飞表示】闭圆盘 + η=0.01 km 安全裕度，仅在 [s,e] 内生效
【航段冲突】连续几何相交区间 ∩ 生效区间 ≠ ∅
【点内服务】[A, A+300] 与点所在圆盘生效区间不相交
【等待规则】仅安全位置必要等待；固定任务序列取最早可行时间表
【重叠圆盘】同时生效且相交按并集处理（逐圆判交天然等价）
【动态代价】E_{uv}(t)=earliest_safe_travel；Φ_{uv}(t)=earliest_safe_service_completion
【主目标】lex(S_max, Δ, ΣS_k, W, L)，整数秒 + 1e-6 km 舍入
【可选口径】nine_hour_cap_s=32400 且 lex_first_is_N=True 时目标首项为 N
【重点异常】Case2 基地在 Z3 内（10:30–13:00 禁起降停）；Case2 Z1/Z2 重叠；Case4 Z8=17:00–17:00
【热启动】outputs/workbooks/q2/strict/Case*.json（sha256 校验），回退 Q1 基线→贪心
【输出】outputs/workbooks/q3/{strict,timetables,reports,checkpoints} + outputs/workbooks/result3.xlsx
【必须验证】verify.py 九条全过、tests/q3 全绿且覆盖 ≥80%、seed 复现、四件套一致
【结论边界】未运行前不填写问题3数值结果；attachment/result3.xlsx 永不覆盖
```
