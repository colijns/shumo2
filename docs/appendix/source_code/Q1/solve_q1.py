#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题1：最少无人机数与最短完工时间（带单机 9h 时限的 min-max mTSP）。

模型与流程见 docs/问题1.md（模型合同）。要点：
- I/II/III 级巡检点展开为 3/2/1 个独立任务，全部必访；
- 距离 = 0.1 × 欧氏坐标（km），飞行时间逐段向上取整到秒，单次巡检 300 s；
- 单机工作时间 = 飞行 + 巡检 ≤ 32400 s（9 h），所有无人机自基地 (0,0) 出发并返回；
- 同一无人机的相邻任务不得属于同一巡检点（必须先离开再返回）。禁排采用双编码：
  弧惩罚（超过单机上限，容量维度等效硬禁止）+ NextVar 传播约束；首解用
  ALL_UNPERFORMED（空路线起步）+ 节点丢弃惩罚，由 GLS 在传播检查下插入全部任务——
  构造式首解策略在紧可行域下会失败（见 Q1/logs 与会话记录中的对照实验）；
- 两阶段词典序：先自 N_LB 起求最小可行 N（每档 120 s），固定 N 后压 Tmax（300 s）；
- 仅当 N_feasible == N_LB 时标记 PROVEN_BY_LOWER_BOUND，否则 BEST_FEASIBLE_NOT_PROVEN。

复现性说明：OR-Tools 局部搜索无随机种子接口，结果随时间预算与机器状态波动；
论文数字以 outputs/workbooks/q1_solution_Case*.json 解档案为准，
`--verify` 可在任何机器上从档案 + 原始附件独立复算全部指标，不调用求解器。

用法（math 环境，先 conda activate math）：
    python solve_q1.py              # 全量求解 Case1~Case4
    python solve_q1.py --smoke      # Case4 冒烟测试（30 s，核对约束数与可行性）
    python solve_q1.py --resume     # 跳过已有完整解档案的算例
    python solve_q1.py --verify     # 不求解，仅从解档案重放验证并出报告
"""

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "templates"))
from common.io_utils import load_table  # noqa: E402

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402

# ---------------- 建模合同常量 ----------------
SPEED_KMH = 55.0            # 巡航速度 km/h
UNIT_KM = 0.1               # 坐标 1 单位 = 0.1 km
SERVICE_S = 300             # 单次巡检 5 min
CAP_S = 32400               # 单机 9 h
PHASE1_S = 120              # 阶段1：每个候选 N 的搜索预算
PHASE2_S = 300              # 阶段2：固定 N 压 Tmax 的搜索预算
N_SEARCH_CAP = 6            # 阶段1 自 N_LB 向上最多尝试 +6
SMOKE_S = 30                # 冒烟测试预算
CASES = ["Case1", "Case2", "Case3", "Case4"]
LEVEL_VISITS = {"I": 3, "II": 2, "III": 1}
# 同点相邻禁排约束期望条数（6*n_I + 2*n_II，按 2026-08 附件核对）。
# 硬断言用于 fail-fast 拦截 all-pairs 误实现（该误实现会产生 ~33k 条约束）。
EXPECTED_ADJ = {"Case1": 50, "Case2": 100, "Case3": 100, "Case4": 130}
ADJ_WARN_THRESHOLD = 5000
# 禁止相邻的双编码：① 弧惩罚（>单机上限 → 容量约束等效硬禁止，且不影响首解构造）；
# ② NextVar != 约束（对局部搜索移动做传播检查）。实测构造式首解策略 + NextVar
# 在紧可行域下会构造失败（Case1 N=4 出现无解假象），故首解改用 ALL_UNPERFORMED：
# 初始解为空路线（恒可构造），配合节点丢弃惩罚由 GLS 在传播检查下逐步插入任务。
BAN_PENALTY_S = 10_000_000
DROP_PENALTY = 100_000_000
FIRST_SOLUTION_CHAIN = [
    routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
]
FS_NAMES = {
    routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED: "ALL_UNPERFORMED",
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION: "PARALLEL_CHEAPEST_INSERTION",
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC: "PATH_CHEAPEST_ARC",
}
ATTACH_XLSX = ROOT / "attachment" / "附件1.xlsx"
OUT_DIR = ROOT / "outputs" / "workbooks"
LOG_DIR = HERE / "logs"

logger = logging.getLogger("q1")


class InfeasiblePointError(Exception):
    """单点可达性校验不通过：往返飞行 + 单次巡检超过 9 h。"""


# ---------------- 日志 ----------------
def setup_logging(ts: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"solve_q1_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------- 数据层 ----------------
def load_case(name: str) -> dict:
    df = load_table(ATTACH_XLSX, sheet_name=name)
    points = []
    for row in df.itertuples(index=False):
        pid, x, y, lvl = int(row.Point_ID), float(row.X_Coordinate), float(row.Y_Coordinate), str(row.Inspection_Level).strip().upper()
        if lvl not in LEVEL_VISITS:
            raise ValueError(f"{name}: 未知的巡检等级 {lvl!r}（Point_ID={pid}）")
        points.append({"pid": pid, "x": x, "y": y, "level": lvl})
    tasks = []
    for p in points:
        for _ in range(LEVEL_VISITS[p["level"]]):
            tasks.append({"task_id": len(tasks) + 1, "pid": p["pid"], "x": p["x"], "y": p["y"]})
    lv = Counter(p["level"] for p in points)
    logger.info("[%s] 点数=%d（I=%d II=%d III=%d），展开任务数 M=%d",
                name, len(points), lv["I"], lv["II"], lv["III"], len(tasks))
    return {"name": name, "points": points, "tasks": tasks}


# ---------------- 矩阵层 ----------------
def build_matrices(case: dict):
    """节点 0 = 基地 (0,0)，节点 1..M = 任务。返回 (dist_km, time_s 嵌套 int 列表)。"""
    coords = [(0.0, 0.0)] + [(t["x"], t["y"]) for t in case["tasks"]]
    a = np.asarray(coords) * UNIT_KM
    diff = a[:, None, :] - a[None, :, :]
    dist_km = np.sqrt((diff ** 2).sum(-1))
    time_s = np.ceil(dist_km / SPEED_KMH * 3600.0).astype(np.int64).tolist()
    return dist_km, time_s


# ---------------- 下界层 ----------------
def compute_lower_bounds(case: dict) -> dict:
    M = len(case["tasks"])
    # 下界1：纯服务量（单机 9h 至多 108 个任务）
    lb_service = math.ceil(M * SERVICE_S / CAP_S)
    # 下界2：总服务时间 + MST 飞行距离
    pc = np.array([[p["x"], p["y"]] for p in case["points"]] + [[0.0, 0.0]]) * UNIT_KM
    d = np.sqrt(((pc[:, None, :] - pc[None, :, :]) ** 2).sum(-1))
    mst_km = float(minimum_spanning_tree(csr_matrix(d)).sum())
    lb_mst = math.ceil((M / 12.0 + mst_km / SPEED_KMH) / 9.0)
    # 单点可达性（用向上取整后的保守秒数）
    for p in case["points"]:
        d0i = UNIT_KM * math.hypot(p["x"], p["y"])
        c0i = math.ceil(3600.0 * d0i / SPEED_KMH)
        if 2 * c0i + SERVICE_S > CAP_S:
            raise InfeasiblePointError(
                f"Point_ID={p['pid']} 往返 {2 * c0i} s + 巡检 {SERVICE_S} s 超过单机上限 {CAP_S} s")
    lb = max(lb_service, lb_mst)
    logger.info("[%s] 下界：服务量=%d，MST=%.3f km → MST下界=%d，N_LB=%d",
                case["name"], lb_service, mst_km, lb_mst, lb)
    return {"n_lb": lb, "lb_service": lb_service, "lb_mst": lb_mst, "mst_km": mst_km}


# ---------------- 构造式可行解（确定性，用于热启动与可行性证明） ----------------
def _leg_s_xy(pa, pb):
    d = UNIT_KM * math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    return math.ceil(3600.0 * d / SPEED_KMH)


def _ctx_maps(case):
    tasks_of = {}
    for t in case["tasks"]:
        tasks_of.setdefault(t["pid"], []).append(t["task_id"])
    pid_of = {t["task_id"]: t["pid"] for t in case["tasks"]}
    coords = {p["pid"]: (p["x"], p["y"]) for p in case["points"]}
    return tasks_of, pid_of, coords


def _task_nn_route(chunk, tasks_of, pid_of, coords):
    """簇内任务级最近邻构造（同点禁排：下一任务不得来自同一巡检点）。

    自然产生 "p → 邻近点 → p" 的廉价再访模式，代价约 2×邻距，
    远优于把同点副本隔开一整轮的做法。"""
    depot = (0.0, 0.0)
    remaining = [t for p in chunk for t in tasks_of[p["pid"]]]
    cur, cur_pid, seq = depot, None, []
    while remaining:
        best, best_d = None, None
        for t in remaining:
            if pid_of[t] == cur_pid:
                continue
            d = _leg_s_xy(cur, coords[pid_of[t]])
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best is None:
            # 剩余任务全在当前点：把副本回插到序列中两侧均非同点的最廉价位置
            if not seq:
                return None
            stuck = remaining[0]
            best_pos, best_cost = None, None
            for i in range(len(seq)):
                left = seq[i - 1] if i > 0 else None
                right = seq[i]
                if left is not None and pid_of[left] == cur_pid:
                    continue
                if pid_of[right] == cur_pid:
                    continue
                a = coords[pid_of[left]] if left is not None else depot
                b = coords[pid_of[right]]
                c = coords[cur_pid]
                cost = _leg_s_xy(a, c) + _leg_s_xy(c, b) - _leg_s_xy(a, b)
                if best_cost is None or cost < best_cost:
                    best_cost, best_pos = cost, i
            if best_pos is None:
                return None
            seq.insert(best_pos, stuck)
            remaining.remove(stuck)
            continue
        seq.append(best)
        remaining.remove(best)
        cur, cur_pid = coords[pid_of[best]], pid_of[best]
    return seq


def _two_opt_ban(seq, pid_of, coords):
    """带同点禁排检查的 2-opt（反转段的两端接缝不得出现相同 Point_ID）。"""
    depot = (0.0, 0.0)
    pos = {t: coords[pid_of[t]] for t in seq}
    for _ in range(100):
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                if j - i == 1:
                    continue
                a, b = (pos[seq[i - 1]] if i > 0 else depot), pos[seq[i]]
                c, d = pos[seq[j]], (pos[seq[j + 1]] if j + 1 < len(seq) else depot)
                delta = (_leg_s_xy(a, c) + _leg_s_xy(b, d)) - (_leg_s_xy(a, b) + _leg_s_xy(c, d))
                if delta < -1e-9:
                    if i > 0 and pid_of[seq[i - 1]] == pid_of[seq[j]]:
                        continue
                    if j + 1 < len(seq) and pid_of[seq[i]] == pid_of[seq[j + 1]]:
                        continue
                    seq[i:j + 1] = reversed(seq[i:j + 1])
                    improved = True
        if not improved:
            break
    return seq


def _route_work(seq, pid_of, coords):
    depot = (0.0, 0.0)
    pc = [depot] + [coords[pid_of[t]] for t in seq] + [depot]
    return sum(_leg_s_xy(a, b) for a, b in zip(pc, pc[1:])) + SERVICE_S * len(seq)


def _chunk_seq(chunk, tasks_of, pid_of, coords):
    seq = _task_nn_route(chunk, tasks_of, pid_of, coords)
    if seq is None:
        return None
    return _two_opt_ban(seq, pid_of, coords)


def _rebalance_chunks(chunks, tasks_of, pid_of, coords, max_rounds=20, n_cand=6):
    """改进式再平衡：每轮从最超载簇枚举 (边界点, 目标簇) 移动，只接受能降低
    最大簇工作时间的移动（避免震荡）；全部簇满足上限时返回任务路线列表。
    候选点取离其他簇质心最近的 n_cand 个（边界点），轮数封顶控制耗时。"""
    n_uav = len(chunks)
    centroids = [(sum(p["x"] for p in ch) / len(ch), sum(p["y"] for p in ch)) if ch else (0.0, 0.0)
                 for ch in chunks]
    for _ in range(max_rounds):
        seqs = [_chunk_seq(ch, tasks_of, pid_of, coords) for ch in chunks]
        if any(s is None for s in seqs):
            return None
        works = [_route_work(s, pid_of, coords) for s in seqs]
        if all(w <= CAP_S for w in works):
            return seqs
        k = max(range(n_uav), key=lambda i: works[i])
        others = [works[i] for i in range(n_uav) if i != k]
        border = sorted(chunks[k], key=lambda p: min(math.hypot(p["x"] - centroids[j][0], p["y"] - centroids[j][1])
                                                     for j in range(n_uav) if j != k))
        best_score, best_move = max(works), None
        for p in border[:n_cand]:
            for k2 in range(n_uav):
                if k2 == k:
                    continue
                src = [q for q in chunks[k] if q is not p]
                dst = chunks[k2] + [p]
                if not src:  # 保持每簇非空
                    continue
                s2, s3 = _chunk_seq(src, tasks_of, pid_of, coords), _chunk_seq(dst, tasks_of, pid_of, coords)
                if s2 is None or s3 is None:
                    continue
                w2, w3 = _route_work(s2, pid_of, coords), _route_work(s3, pid_of, coords)
                score = max([w2, w3] + others)
                if score < best_score - 1e-9:
                    best_score, best_move = score, (p, k2)
        if best_move is None:
            return None
        p, k2 = best_move
        chunks[k].remove(p)
        chunks[k2].append(p)
        centroids[k] = (sum(q["x"] for q in chunks[k]) / len(chunks[k]), sum(q["y"] for q in chunks[k]) / len(chunks[k]))
        centroids[k2] = (sum(q["x"] for q in chunks[k2]) / len(chunks[k2]), sum(q["y"] for q in chunks[k2]) / len(chunks[k2]))
    return None


def sweep_routes(case: dict, n_uav: int):
    """构造法一：极角扫过分扇区 + 改进式再平衡。失败返回 None（不代表该 N 不可行）。"""
    pts = sorted(case["points"], key=lambda p: math.atan2(p["y"], p["x"]))
    tasks_of, pid_of, coords = _ctx_maps(case)
    weights, prev = [], (0.0, 0.0)
    for p in pts:
        weights.append(SERVICE_S * len(tasks_of[p["pid"]]) + _leg_s_xy(prev, (p["x"], p["y"])))
        prev = (p["x"], p["y"])

    cuts, acc, target = [0], 0.0, sum(weights) / n_uav
    for i, w in enumerate(weights):
        acc += w
        if len(cuts) < n_uav and acc >= target * len(cuts) and i + 1 < len(pts):
            cuts.append(i + 1)
    cuts.append(len(pts))
    chunks = [pts[cuts[k]:cuts[k + 1]] for k in range(n_uav)]
    if any(not c for c in chunks):
        return None
    return _rebalance_chunks(chunks, tasks_of, pid_of, coords)


def kmeans_routes(case: dict, n_uav: int):
    """构造法二：容量感知 k-means 聚类（Lloyd 迭代）+ 改进式再平衡。

    服务容量按 (1-travel_frac)×CAP 预留飞行时间，travel_frac 依次尝试 0.30/0.38/0.46；
    聚类中心由极角扇区质心确定性初始化。失败返回 None。"""
    tasks_of, pid_of, coords = _ctx_maps(case)
    pts = case["points"]
    n = len(pts)
    if n_uav <= 0:
        return None

    by_angle = sorted(pts, key=lambda p: math.atan2(p["y"], p["x"]))
    centers = []
    for k in range(n_uav):
        seg = by_angle[k * n // n_uav:(k + 1) * n // n_uav] or by_angle[-1:]
        centers.append((sum(p["x"] for p in seg) / len(seg), sum(p["y"] for p in seg) / len(seg)))
    w = {p["pid"]: SERVICE_S * len(tasks_of[p["pid"]]) for p in pts}

    for travel_frac in (0.30, 0.38, 0.46):
        cap = CAP_S * (1.0 - travel_frac)
        assign = {p["pid"]: 0 for p in pts}
        failed = False
        for _ in range(30):  # Lloyd 迭代
            rem = [cap] * n_uav
            order = sorted(pts, key=lambda p: min(math.hypot(p["x"] - c[0], p["y"] - c[1]) for c in centers))
            for p in order:
                for c in sorted(range(n_uav), key=lambda i: math.hypot(p["x"] - centers[i][0], p["y"] - centers[i][1])):
                    if rem[c] >= w[p["pid"]]:
                        assign[p["pid"]] = c
                        rem[c] -= w[p["pid"]]
                        break
                else:
                    failed = True
                    break
            if failed:
                break
            new_centers = []
            for c in range(n_uav):
                members = [p for p in pts if assign[p["pid"]] == c]
                if not members:
                    far = max(pts, key=lambda p: min(math.hypot(p["x"] - cc[0], p["y"] - cc[1]) for cc in centers))
                    members = [far]
                    assign[far["pid"]] = c
                new_centers.append((sum(p["x"] for p in members) / len(members),
                                    sum(p["y"] for p in members) / len(members)))
            if new_centers == centers:
                break
            centers = new_centers
        if failed:
            continue
        chunks = [[p for p in pts if assign[p["pid"]] == c] for c in range(n_uav)]
        if any(not c for c in chunks):
            continue
        routes = _rebalance_chunks(chunks, tasks_of, pid_of, coords)
        if routes is not None:
            return routes
    return None


def solve_n(case: dict, time_s, n_uav: int, budget_s: int, want_curve: bool = False, init_routes=None):
    """建模型并搜索。返回 (result|None, curve)；result 含 routes 与 served（已服务任务数）。

    可行性判据：served == M（有丢弃惩罚时求解器可能返回带丢弃的"解"，不算可行）。
    init_routes：任务节点序列的热启动解（如 sweep 构造），经约束校验后用作初始解。"""
    M = len(case["tasks"])
    same = set()
    for a in range(1, M + 1):
        for b in range(1, M + 1):
            if a != b and case["tasks"][a - 1]["pid"] == case["tasks"][b - 1]["pid"]:
                same.add((a, b))
    if len(same) != EXPECTED_ADJ[case["name"]]:
        raise RuntimeError(
            f"[{case['name']}] 禁止相邻任务对 {len(same)} != 期望 {EXPECTED_ADJ[case['name']]}："
            f"疑似把约束加成了全体任务对（all-pairs），立即终止")
    if len(same) > ADJ_WARN_THRESHOLD:
        logger.warning("[%s] 禁止相邻任务对 %d 超过告警阈值 %d", case["name"], len(same), ADJ_WARN_THRESHOLD)

    t0 = time.perf_counter()
    manager = pywrapcp.RoutingIndexManager(M + 1, n_uav, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(fi, ti):
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        c = time_s[f][t] + (SERVICE_S if 1 <= t <= M else 0)
        if (f, t) in same:
            c += BAN_PENALTY_S  # 超过单机上限 → 容量维度等效硬禁止
        return c

    tcb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(tcb)
    routing.AddDimension(tcb, 0, CAP_S, True, "Time")
    dim = routing.GetDimensionOrDie("Time")
    dim.SetGlobalSpanCostCoefficient(1)  # 主目标 = 最长路线时间（起点全固定为 0）

    # 节点丢弃惩罚：初始解（ALL_UNPERFORMED）为空路线，靠 GLS 插入全部任务
    for t in range(1, M + 1):
        routing.AddDisjunction([manager.NodeToIndex(t)], DROP_PENALTY)

    # 同 Point_ID 任务禁止相邻（先离开再返回才能再次巡检）——传播层硬约束
    solver = routing.solver()
    by_pid = {}
    for node in range(1, M + 1):
        by_pid.setdefault(case["tasks"][node - 1]["pid"], []).append(node)
    n_con = 0
    for nodes in by_pid.values():
        idxs = [manager.NodeToIndex(n) for n in nodes]
        for iu in idxs:
            for iv in idxs:
                if iu != iv:
                    solver.Add(routing.NextVar(iu) != iv)
                    n_con += 1
    assert n_con == len(same)
    if n_con != EXPECTED_ADJ[case["name"]]:
        raise RuntimeError(
            f"[{case['name']}] 相邻禁排约束数 {n_con} != 期望 {EXPECTED_ADJ[case['name']]}："
            f"疑似把约束加成了全体任务对（all-pairs），立即终止")
    if n_con > ADJ_WARN_THRESHOLD:
        logger.warning("[%s] 约束数 %d 超过告警阈值 %d", case["name"], n_con, ADJ_WARN_THRESHOLD)
    build_s = time.perf_counter() - t0

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(budget_s * 1000))

    curve = []
    t_start = time.perf_counter()
    if want_curve:
        def _on_solution():
            curve.append((round(time.perf_counter() - t_start, 3), int(routing.CostVar().Max())))
        try:
            routing.AddAtSolutionCallback(_on_solution)
        except AttributeError:  # 旧版本无该 API，降级为起止两点
            logger.info("[%s] 当前 OR-Tools 无 AtSolutionCallback，收敛曲线降级", case["name"])

    solution, strategy_name = None, None
    if init_routes is not None:
        initial = routing.ReadAssignmentFromRoutes(init_routes, True)
        if initial is not None:
            sol = routing.SolveFromAssignmentWithParameters(initial, params)
            if sol is not None:
                solution, strategy_name = sol, "WARM_START(provided)"
        else:
            logger.info("[%s] N=%d 热启动解未通过约束校验，回退首解策略链", case["name"], n_uav)
    if solution is None:
        for fs in FIRST_SOLUTION_CHAIN:
            params.first_solution_strategy = fs
            sol = routing.SolveWithParameters(params)
            if sol is not None:
                solution, strategy_name = sol, FS_NAMES[fs]
                break
            logger.info("[%s] N=%d 首解策略 %s 未返回解，尝试下一策略",
                        case["name"], n_uav, FS_NAMES[fs])
    elapsed = time.perf_counter() - t_start
    if solution is None:
        return None, curve, build_s, n_con, elapsed

    routes = []
    for v in range(n_uav):
        idx, seq = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                seq.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(seq)
    result = {
        "routes": routes,
        "served": int(sum(len(r) for r in routes)),
        "n_con": n_con,
        "build_s": build_s,
        "elapsed": elapsed,
        "strategy": strategy_name,
        "objective": int(solution.ObjectiveValue()),
    }
    return result, curve, build_s, n_con, elapsed


# ---------------- 评价层 ----------------
def evaluate(case: dict, dist_km, time_s, routes):
    stats = []
    for seq in routes:
        path = [0] + seq + [0]
        fly = int(sum(time_s[a][b] for a, b in zip(path, path[1:])))
        dk = float(sum(dist_km[a][b] for a, b in zip(path, path[1:])))
        work = fly + SERVICE_S * len(seq)
        stats.append({
            "n_tasks": len(seq),
            "fly_s": fly,
            "dist_km": round(dk, 6),
            "work_s": work,
            "work_h": work / 3600.0,
            "task_seq": [int(t) for t in seq],
            "point_seq": [int(case["tasks"][t - 1]["pid"]) for t in seq],
        })
    return stats


def span_of(stats):
    active = [s for s in stats if s["n_tasks"] > 0]
    return max(s["work_s"] for s in active), min(s["work_s"] for s in active), len(active)


# ---------------- 两阶段驱动 ----------------
def solve_case(case: dict) -> dict:
    name = case["name"]
    dist_km, time_s = build_matrices(case)
    lb = compute_lower_bounds(case)

    # 阶段1：自 N_LB 递增找首个可行 N（判据：全部任务被服务）；
    # 每档先做确定性构造（sweep → kmeans，成功即证明可行并作热启动），失败再交给 GLS 插入
    n_try, phase1 = lb["n_lb"], None
    while n_try <= lb["n_lb"] + N_SEARCH_CAP:
        ctor = sweep_routes(case, n_try)
        ctor_name = "sweep"
        if ctor is None:
            ctor = kmeans_routes(case, n_try)
            ctor_name = "kmeans"
        if ctor is not None:
            logger.info("[%s] 阶段1 N=%d %s构造成功（可行性构造证明）", name, n_try, ctor_name)
        else:
            logger.info("[%s] 阶段1 N=%d sweep/kmeans 构造失败，转 GLS 插入", name, n_try)
        res, _, build_s, n_con, elapsed = solve_n(case, time_s, n_try, PHASE1_S, init_routes=ctor)
        if res is not None and res["served"] == len(case["tasks"]):
            logger.info("[%s] 阶段1 N=%d 预算=%ds 建模=%.2fs 约束=%d → 可行 served=%d/%d（耗时 %.1fs）",
                        name, n_try, PHASE1_S, build_s, n_con, res["served"], len(case["tasks"]), elapsed)
            phase1 = (n_try, res)
            break
        served = res["served"] if res is not None else 0
        logger.info("[%s] 阶段1 N=%d 预算=%ds 建模=%.2fs 约束=%d → 未找到全服务解（served=%d/%d，耗时 %.1fs）",
                    name, n_try, PHASE1_S, build_s, n_con, served, len(case["tasks"]), elapsed)
        n_try += 1
    if phase1 is None:
        return {"case": name, "status": "NO_FEASIBLE_FOUND",
                "n_lb": lb["n_lb"], "detail": f"自 N_LB={lb['n_lb']} 尝试至 +{N_SEARCH_CAP} 均未找到可行解", "complete": False}

    # 阶段2：固定 N 压 Tmax；若解中出现空航线则尝试收缩 N（收缩后重新优化）
    n_uav, fallback = phase1
    best = None
    while True:
        warm = (best or fallback)["routes"]
        res, curve, build_s, n_con, elapsed = solve_n(case, time_s, n_uav, PHASE2_S, want_curve=True,
                                                      init_routes=warm)
        if res is not None and res["served"] == len(case["tasks"]):
            best = res
        elif best is None:
            logger.warning("[%s] 阶段2未找到全服务解（异常，阶段1已有可行解），回退阶段1解", name)
            best = fallback
        n_eff = sum(1 for r in best["routes"] if r)
        if n_eff < n_uav:
            logger.info("[%s] 阶段2解含空航线（有效 %d/%d），尝试收缩至 N=%d", name, n_eff, n_uav, n_eff)
            cand, _, _, _, _ = solve_n(case, time_s, n_eff, PHASE1_S)
            if cand is None:
                logger.info("[%s] N=%d 收缩失败，保留当前 N=%d 解", name, n_eff, n_uav)
                break
            n_uav, fallback, best = n_eff, cand, None
            continue
        logger.info("[%s] 阶段2 N=%d 预算=%ds 策略=%s 目标=%d（耗时 %.1fs，收敛点 %d 个）",
                    name, n_uav, PHASE2_S, best["strategy"], best["objective"], elapsed, len(curve))
        break

    stats = evaluate(case, dist_km, time_s, best["routes"])
    tmax_s, tmin_s, n_active = span_of(stats)
    if n_active < n_uav:
        logger.warning("[%s] 最终解仍有空航线（%d/%d 有效）", name, n_active, n_uav)
    optimality = "PROVEN_BY_LOWER_BOUND" if n_uav == lb["n_lb"] else "BEST_FEASIBLE_NOT_PROVEN"
    logger.info("[%s] 结果：N=%d（N_LB=%d，%s）Tmax=%.4f h Tmin=%.4f h",
                name, n_uav, lb["n_lb"], optimality, tmax_s / 3600.0, tmin_s / 3600.0)

    archive = {
        "case": name,
        "status": "SOLVED",
        "complete": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "N": n_uav,
        "Tmax_s": tmax_s,
        "Tmin_s": tmin_s,
        "Tmax_h": tmax_s / 3600.0,
        "Tmin_h": tmin_s / 3600.0,
        "optimality": optimality,
        "lower_bounds": lb,
        "adj_constraints": best["n_con"],
        "model_build_s": round(best["build_s"], 3),
        "first_solution_strategy": best["strategy"],
        "solver_objective": best["objective"],
        "params": {"speed_kmh": SPEED_KMH, "unit_km": UNIT_KM, "service_s": SERVICE_S,
                   "cap_s": CAP_S, "phase1_s": PHASE1_S, "phase2_s": PHASE2_S,
                   "metaheuristic": "GUIDED_LOCAL_SEARCH"},
        "uavs": [{"uav_id": i + 1, **s} for i, s in enumerate(stats)],
    }
    apath = OUT_DIR / f"q1_solution_{name}.json"
    apath.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[%s] 解档案已写入 %s", name, apath)

    if curve:
        cpath = LOG_DIR / f"convergence_{name}.csv"
        pd.DataFrame(curve, columns=["elapsed_s", "objective"]).to_csv(cpath, index=False)
    return archive


# ---------------- 独立校验层（纯重放，不用 OR-Tools） ----------------
def _leg_seconds(pa, pb):
    d = UNIT_KM * math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    return math.ceil(3600.0 * d / SPEED_KMH)


def verify_case(case: dict, arch: dict):
    """从解档案的 Point_ID 序列 + 附件原始数据独立复算，输出结构化违规清单。"""
    name, viol = case["name"], []
    coords = {p["pid"]: (p["x"], p["y"]) for p in case["points"]}
    depot = (0.0, 0.0)
    uavs = arch.get("uavs", [])

    for u in uavs:
        seq = u.get("point_seq", [])
        if not seq:
            viol.append({"case": name, "uav": u.get("uav_id"), "type": "EMPTY_ROUTE", "detail": "路线为空"})
        for a, b in zip(seq, seq[1:]):
            if a == b:
                viol.append({"case": name, "uav": u.get("uav_id"), "type": "ADJACENT_DUPLICATE",
                             "detail": f"相邻 Point_ID 重复：{a} → {b}"})

    cnt = Counter(pid for u in uavs for pid in u.get("point_seq", []))
    for p in case["points"]:
        exp, act = LEVEL_VISITS[p["level"]], cnt.get(p["pid"], 0)
        if act < exp:
            viol.append({"case": name, "uav": None, "type": "MISSING_VISIT",
                         "detail": f"Point_ID={p['pid']} 期望 {exp} 次巡检，实际 {act} 次"})
        elif act > exp:
            viol.append({"case": name, "uav": None, "type": "EXTRA_VISIT",
                         "detail": f"Point_ID={p['pid']} 期望 {exp} 次巡检，实际 {act} 次"})

    recompute = []
    for u in uavs:
        seq = u.get("point_seq", [])
        path = [depot] + [coords[pid] for pid in seq] + [depot]
        s = sum(_leg_seconds(a, b) for a, b in zip(path, path[1:])) + SERVICE_S * len(seq)
        recompute.append(s)
        if s > CAP_S:
            viol.append({"case": name, "uav": u.get("uav_id"), "type": "TIME_OVERFLOW",
                         "detail": f"复算工作时间 {s} s 超上限 {CAP_S} s（超出 {s - CAP_S} s）"})
        if s != u.get("work_s"):
            viol.append({"case": name, "uav": u.get("uav_id"), "type": "MISMATCH",
                         "detail": f"复算 work_s={s}，档案记录 work_s={u.get('work_s')}"})

    active = [(s, u) for s, u in zip(recompute, uavs) if u.get("point_seq")]
    if active:
        rt_max, rt_min = max(s for s, _ in active), min(s for s, _ in active)
        if arch.get("Tmax_s") != rt_max or arch.get("Tmin_s") != rt_min:
            viol.append({"case": name, "uav": None, "type": "MISMATCH",
                         "detail": f"复算 Tmax/Tmin={rt_max}/{rt_min} s，档案记录 {arch.get('Tmax_s')}/{arch.get('Tmin_s')} s"})
        if len(active) != arch.get("N"):
            viol.append({"case": name, "uav": None, "type": "MISMATCH",
                         "detail": f"有效无人机数 {len(active)} != 档案 N={arch.get('N')}"})
    return viol


def check_result1_xlsx(archives):
    """回读 result1.xlsx，与解档案逐格核对。"""
    path = OUT_DIR / "result1.xlsx"
    if not path.exists():
        return [{"case": None, "uav": None, "type": "RESULT_XLSX_MISSING", "detail": str(path)}]
    viol = []
    for name in CASES:
        arch = archives.get(name)
        if not arch or arch.get("status") != "SOLVED":
            continue
        df = load_table(path, sheet_name=name)
        for i, u in enumerate(arch["uavs"]):
            row = df.iloc[i]
            if int(row["UAV ID"]) != u["uav_id"]:
                viol.append({"case": name, "uav": u["uav_id"], "type": "RESULT_XLSX_MISMATCH",
                             "detail": f"行 {i + 1} UAV ID={row['UAV ID']} != {u['uav_id']}"})
                continue
            cells = [int(v) for v in row.tolist()[1:] if pd.notna(v)]
            if cells != u["point_seq"]:
                viol.append({"case": name, "uav": u["uav_id"], "type": "RESULT_XLSX_MISMATCH",
                             "detail": f"表中序列 {cells} != 档案序列 {u['point_seq']}"})
    return viol


# ---------------- 输出层 ----------------
def write_result1(archives):
    """官方模板格式：列 UAV ID / 1th Inspection Point / 2th …（多 sheet 需 ExcelWriter，
    common/io_utils.export_result 仅支持单表，故此处直接用 pandas）。"""
    path = OUT_DIR / "result1.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in CASES:
            arch = archives.get(name)
            if not arch or arch.get("status") != "SOLVED":
                continue
            seqs = [u["point_seq"] for u in arch["uavs"] if u["point_seq"]]
            width = max(len(s) for s in seqs)
            rows = [[i] + s + [None] * (width - len(s)) for i, s in enumerate(seqs, 1)]
            cols = ["UAV ID"] + [f"{k}th Inspection Point" for k in range(1, width + 1)]
            pd.DataFrame(rows, columns=cols).to_excel(xw, sheet_name=name, index=False)
    logger.info("result1.xlsx 已写入 %s", path)
    return path


def write_summary(archives):
    path = OUT_DIR / "summary_q1.xlsx"
    t2 = []
    for name in CASES:
        arch = archives.get(name, {})
        if arch.get("status") == "SOLVED":
            t2.append([name, arch["N"], round(arch["Tmax_h"], 4), round(arch["Tmin_h"], 4), arch["optimality"]])
        else:
            t2.append([name, None, None, None, arch.get("status", "NOT_RUN")])
    df2 = pd.DataFrame(t2, columns=["测试算例", "无人机数量 N", "单架最长工作时间 Tmax (h)", "单架最短工作时间 Tmin (h)", "最优性标记"])
    detail = []
    for name in CASES:
        arch = archives.get(name, {})
        for u in arch.get("uavs", []):
            detail.append([name, u["uav_id"], u["n_tasks"], u["fly_s"], u["dist_km"], round(u["work_h"], 4),
                           " → ".join(map(str, u["point_seq"]))])
    dfd = pd.DataFrame(detail, columns=["算例", "UAV ID", "任务数", "飞行时间 (s)", "飞行距离 (km)", "工作时间 (h)", "访问序列 (Point_ID)"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df2.to_excel(xw, sheet_name="表2", index=False)
        dfd.to_excel(xw, sheet_name="routes_detail", index=False)
    logger.info("summary_q1.xlsx 已写入 %s", path)
    return path


def write_reports(violations):
    jpath, mpath = OUT_DIR / "q1_verification_report.json", OUT_DIR / "q1_verification_report.md"
    jpath.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "n_violations": len(violations), "violations": violations},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 问题1 校验报告", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"违规总数：**{len(violations)}**", ""]
    if violations:
        lines += ["| 算例 | 无人机 | 类型 | 详情 |", "|---|---|---|---|"]
        lines += [f"| {v['case']} | {v['uav']} | {v['type']} | {v['detail']} |" for v in violations]
    else:
        lines += ["六项检查（任务不重不漏 / 巡检次数 / 相邻不重复 / 路线闭合非空 / 单机 ≤9h / 指标复算一致）全部通过。"]
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("校验报告已写入 %s（违规 %d 条）", mpath, len(violations))
    return violations


# ---------------- 运行模式 ----------------
def run_smoke():
    case = load_case("Case4")
    _, time_s = build_matrices(case)
    lb = compute_lower_bounds(case)
    res, curve, build_s, n_con, elapsed = solve_n(case, time_s, lb["n_lb"], SMOKE_S)
    M = len(case["tasks"])
    if res is None:
        verdict = "求解器未返回解"
    elif res["served"] == M:
        verdict = f"全服务可行 served={res['served']}/{M}"
    else:
        verdict = f"未完成插入 served={res['served']}/{M}（预算内未找到全服务解，不能断言不可行）"
    logger.info("[SMOKE Case4] 建模耗时 %.2fs，约束数 %d（期望 %d），N=%d 预算 %ds → %s，搜索耗时 %.1fs",
                build_s, n_con, EXPECTED_ADJ["Case4"], lb["n_lb"], SMOKE_S, verdict, elapsed)
    return 0


def run_full(resume: bool) -> int:
    archives = {}
    for name in CASES:
        apath = OUT_DIR / f"q1_solution_{name}.json"
        if resume and apath.exists():
            arch = json.loads(apath.read_text(encoding="utf-8"))
            if arch.get("complete"):
                logger.info("[%s] --resume：解档案完整，跳过求解", name)
                archives[name] = arch
                continue
        case = load_case(name)
        try:
            archives[name] = solve_case(case)
        except InfeasiblePointError as e:
            logger.error("[%s] 数据不可行：%s", name, e)
            archives[name] = {"case": name, "status": "INFEASIBLE_DATA", "error": str(e), "complete": False}
            apath.write_text(json.dumps(archives[name], ensure_ascii=False, indent=2), encoding="utf-8")

    write_result1(archives)
    write_summary(archives)

    viol = []
    for name in CASES:
        arch = archives.get(name)
        if arch and arch.get("status") == "SOLVED":
            viol += verify_case(load_case(name), arch)
    viol += check_result1_xlsx(archives)
    write_reports(viol)

    logger.info("表2（论文誊抄用）：")
    for name in CASES:
        arch = archives.get(name, {})
        if arch.get("status") == "SOLVED":
            logger.info("  %s：N=%d，Tmax=%.4f h，Tmin=%.4f h，%s",
                        name, arch["N"], arch["Tmax_h"], arch["Tmin_h"], arch["optimality"])
        else:
            logger.info("  %s：%s", name, arch.get("status", "NOT_RUN"))
    return 1 if viol else 0


def run_verify() -> int:
    archives, viol = {}, []
    for name in CASES:
        apath = OUT_DIR / f"q1_solution_{name}.json"
        if not apath.exists():
            logger.error("缺少解档案：%s（先运行全量求解）", apath)
            return 2
        archives[name] = json.loads(apath.read_text(encoding="utf-8"))
    for name in CASES:
        arch = archives[name]
        if arch.get("status") == "SOLVED":
            logger.info("[%s] 重放校验：N=%d Tmax=%.4f h Tmin=%.4f h", name, arch["N"], arch["Tmax_h"], arch["Tmin_h"])
            viol += verify_case(load_case(name), arch)
        else:
            logger.info("[%s] 状态 %s，跳过指标复算", name, arch.get("status"))
    viol += check_result1_xlsx(archives)
    write_reports(viol)
    return 1 if viol else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="问题1：min-max mTSP 两阶段求解")
    ap.add_argument("--verify", action="store_true", help="不求解，仅从解档案重放验证")
    ap.add_argument("--resume", action="store_true", help="跳过已有完整解档案的算例")
    ap.add_argument("--smoke", action="store_true", help="Case4 冒烟测试（30s）")
    args = ap.parse_args()
    setup_logging(time.strftime("%Y%m%d_%H%M%S"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify:
        return run_verify()
    if args.smoke:
        return run_smoke()
    return run_full(args.resume)


if __name__ == "__main__":
    sys.exit(main())
