import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import solve_q1
from solve_q1 import (
    CAP_S, CASES, EXPECTED_ADJ, LEVEL_VISITS, LOG_DIR, OUT_DIR, SERVICE_S,
    build_matrices, check_result1_xlsx, compute_lower_bounds, evaluate,
    kmeans_routes, load_case, solve_n, sweep_routes, verify_case,
    write_reports, write_result1, write_summary,
)
from ortools.sat.python import cp_model
TIGHT_N = {"Case1": 3, "Case3": 4}
COMPRESS_N = {"Case1": 4, "Case2": 2, "Case3": 5, "Case4": 4}
C3_TARGET_S = 29102
SEED_DEFAULT = 42
WORKERS_DEFAULT = 8
CPK_DIR = OUT_DIR
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
logger = logging.getLogger("tight")
def setup_logging(ts: str, suffix: str = "") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"tight_{suffix}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    for h in (fh, sh):
        logger.addHandler(h)
    q1 = logging.getLogger("q1")
    q1.handlers.clear()
    q1.addHandler(fh)
    q1.addHandler(sh)
    q1.setLevel(logging.INFO)
    q1.propagate = False
def cp_path(case: str, n: int, track: str) -> Path:
    prefix = "tight_compress" if track == "compress" else "tight_checkpoint"
    return CPK_DIR / f"{prefix}_{case}_N{n}_{track}.json"
def write_atomic(path: Path, obj: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
def better(a: dict, b: dict) -> bool:
    if a["served"] != b["served"]:
        return a["served"] > b["served"]
    if a["Tmax_s"] != b["Tmax_s"]:
        return a["Tmax_s"] < b["Tmax_s"]
    return a["sum_T_s"] < b["sum_T_s"]
def verified_stats(case, dist_km, time_s, routes, name):
    non_empty = [r for r in routes if r]
    stats = evaluate(case, dist_km, time_s, non_empty)
    active = [(s["work_s"], s) for s in stats if s["n_tasks"] > 0]
    if not active:
        return stats, None, [{"case": name, "uav": None, "type": "ALL_EMPTY", "detail": "全部路线为空"}]
    tmax, tmin = max(w for w, _ in active), min(w for w, _ in active)
    arch = {
        "case": name, "status": "SOLVED", "N": len(active),
        "Tmax_s": tmax, "Tmin_s": tmin,
        "uavs": [{"uav_id": i + 1, **s} for i, s in enumerate(stats)],
    }
    viol = verify_case(case, arch)
    return stats, arch, viol
def _ctx(case: dict):
    pid_of = {t["task_id"]: t["pid"] for t in case["tasks"]}
    coords = {p["pid"]: (p["x"], p["y"]) for p in case["points"]}
    return pid_of, coords
def merge_warm_starts(case: dict, time_s, base_routes, n_target: int, seed: int):
    pid_of, coords = _ctx(case)
    rng = random.Random(seed + 1000 * n_target + len(base_routes))
    cands, seen = [], set()
    for k in range(len(base_routes)):
        keep = [list(r) for r in base_routes[:k] + base_routes[k + 1:]]
        moved = list(base_routes[k])
        orders = [
            ("seq", list(moved)),
            ("near", sorted(moved, key=lambda t: math.hypot(*coords[pid_of[t]]))),
        ]
        shuf = list(moved)
        rng.shuffle(shuf)
        orders.append(("shuffle", shuf))
        for oname, ordered in orders:
            routes = [list(r) for r in keep]
            ok = True
            for t in ordered:
                best_pos, best_cost = None, None
                for ri, r in enumerate(routes):
                    for pos in range(len(r) + 1):
                        prev = r[pos - 1] if pos > 0 else 0
                        nxt = r[pos] if pos < len(r) else 0
                        if prev and pid_of.get(prev) == pid_of[t]:
                            continue
                        if nxt and pid_of.get(nxt) == pid_of[t]:
                            continue
                        cost = time_s[prev][t] + time_s[t][nxt] - time_s[prev][nxt]
                        if best_cost is None or cost < best_cost:
                            best_cost, best_pos = cost, (ri, pos)
                if best_pos is None:
                    ok = False
                    break
                ri, pos = best_pos
                routes[ri].insert(pos, t)
            if ok:
                key = tuple(tuple(r) for r in routes)
                if key not in seen:
                    seen.add(key)
                    cands.append((routes, f"merge_del{k}_{oname}"))
    return cands
def build_candidates(case: dict, time_s, n: int, seed: int, baseline_arch=None):
    M = len(case["tasks"])
    cands = []
    if baseline_arch is not None:
        base_routes = [u["task_seq"] for u in baseline_arch["uavs"] if u["task_seq"]]
        if len(base_routes) == n + 1:
            cands += merge_warm_starts(case, time_s, base_routes, n, seed)
        elif len(base_routes) == n and baseline_arch.get("status") == "SOLVED":
            cands.append(([list(r) for r in base_routes], "baseline_warm"))
    for ctor, cname in ((sweep_routes, "sweep"), (kmeans_routes, "kmeans")):
        try:
            routes = ctor(case, n)
        except Exception as e:
            logger.info("构造器 %s N=%d 异常：%s", cname, n, e)
            routes = None
        if routes is not None and sum(len(r) for r in routes) == M:
            cands.append((routes, f"ctor_{cname}"))
    return cands
def build_cp_model(case: dict, time_s, n: int):
    M = len(case["tasks"])
    pid_of = {t["task_id"]: t["pid"] for t in case["tasks"]}
    model = cp_model.CpModel()
    t = {}
    for i in range(1, M + 1):
        ub = CAP_S - time_s[i][0]
        t[i] = model.NewIntVar(0, ub, f"t_{i}")
    arcs, arc_vars, arc_from, banned = [], {}, {}, 0
    def add_arc(u, v):
        lit = model.NewBoolVar(f"x_{u}_{v}")
        arcs.append((u, v, lit))
        arc_vars[(u, v)] = lit
        arc_from.setdefault(u, []).append((v, lit))
        return lit
    for j in range(1, M + 1):
        lit = add_arc(0, j)
        model.Add(t[j] == time_s[0][j] + SERVICE_S).OnlyEnforceIf(lit)
    for i in range(1, M + 1):
        for j in range(1, M + 1):
            if i == j:
                continue
            if pid_of[i] == pid_of[j]:
                banned += 1
                continue
            lit = add_arc(i, j)
            model.Add(t[j] == t[i] + time_s[i][j] + SERVICE_S).OnlyEnforceIf(lit)
    for i in range(1, M + 1):
        add_arc(i, 0)
    if case["name"] in EXPECTED_ADJ:
        if banned != EXPECTED_ADJ[case["name"]]:
            raise RuntimeError(
                f"[{case['name']}] CP-SAT 剔除禁排弧数 {banned} != 期望 {EXPECTED_ADJ[case['name']]}："
                f"疑似误实现（all-pairs 或漏约束）")
    model.AddMultipleCircuit(arcs)
    out_of_depot = [arc_vars[(0, j)] for j in range(1, M + 1)]
    model.Add(sum(out_of_depot) <= n)
    return model, {"M": M, "n": n, "t": t, "arc_vars": arc_vars, "arc_from": arc_from,
                   "banned": banned, "n_arcs": len(arcs)}
def add_route_hint(model, ctx, routes) -> bool:
    for seq in routes:
        if not seq:
            continue
        prev, cum = 0, 0
        for nd in seq:
            lit = ctx["arc_vars"].get((prev, nd))
            if lit is None:
                return False
            model.AddHint(lit, 1)
            cum += _arc_cost(ctx, prev, nd)
            model.AddHint(ctx["t"][nd], cum)
            prev = nd
        lit = ctx["arc_vars"].get((prev, 0))
        if lit is None:
            return False
        model.AddHint(lit, 1)
    return True
def _arc_cost(ctx, u, v):
    return ctx["time_s"][u][v] + (SERVICE_S if v > 0 else 0)
def extract_routes(solver, ctx):
    routes, tour_sums = [], []
    for j in range(1, ctx["M"] + 1):
        if solver.Value(ctx["arc_vars"][(0, j)]) == 1:
            seq, cur = [], j
            while True:
                seq.append(cur)
                nxt = None
                for v, lit in ctx["arc_from"][cur]:
                    if solver.Value(lit) == 1:
                        nxt = v
                        break
                if nxt is None or nxt == 0:
                    break
                cur = nxt
            routes.append(seq)
            tour_sums.append(int(solver.Value(ctx["t"][seq[-1]]) + ctx["time_s"][seq[-1]][0]))
    return routes, tour_sums
def run_track_cpsat(case, dist_km, time_s, n: int, budget: int, seed: int, workers: int,
                    baseline_arch=None):
    name, M = case["name"], len(case["tasks"])
    lb = compute_lower_bounds(case)
    model, ctx = build_cp_model(case, time_s, n)
    ctx["time_s"] = time_s
    logger.info("[%s] CP-SAT N=%d 建模完成：弧=%d 禁排剔除=%d", name, n, ctx["n_arcs"], ctx["banned"])
    cands = build_candidates(case, time_s, n, seed, baseline_arch)
    witness = None
    best_hint, best_hint_work = None, None
    for routes, prov in cands:
        stats, _, viol = verified_stats(case, dist_km, time_s, routes, name)
        works = [s["work_s"] for s in stats]
        if sum(len(r) for r in routes) == M and all(w <= CAP_S for w in works) and not viol:
            witness = (routes, prov)
        if best_hint_work is None or max(works) < best_hint_work:
            best_hint, best_hint_work = routes, max(works)
    if best_hint is not None:
        if add_route_hint(model, ctx, best_hint):
            logger.info("[%s] CP-SAT hint 已注入（%d 条路线，最长 %.0f s）", name, len(best_hint), best_hint_work)
        else:
            logger.info("[%s] CP-SAT hint 注入失败（含禁排弧），无 hint 求解", name)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    logger.info("[%s] CP-SAT 阶段F：status=%s 耗时 %.1f s", name, status.name, elapsed)
    kind, routes, stats, viol = None, None, None, None
    solver_info = {
        "solver": "cp-sat", "status": status.name,
        "elapsed_s": round(elapsed, 3), "budget_s": budget, "workers": workers,
        "seed": seed, "wall_time_s": round(solver.WallTime(), 3), "hint": (best_hint is not None),
    }
    if status == cp_model.INFEASIBLE:
        if witness is not None:
            kind = "model_error"
            logger.error("[%s] CP-SAT 报 INFEASIBLE，但本地存在可行见证 %s（served=%d）——模型编码错误，证明作废",
                         name, witness[1], sum(len(r) for r in witness[0]))
        else:
            kind = "infeasibility_proof"
            logger.info("[%s] CP-SAT 证明 N=%d 不可行（无矛盾见证，矛盾校验通过）", name, n)
    elif status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        routes, tour_sums = extract_routes(solver, ctx)
        stats, arch, viol = verified_stats(case, dist_km, time_s, routes, name)
        if viol:
            kind = "model_error"
            logger.error("[%s] CP-SAT 解提取后独立校验 %d 违规：%s", name, len(viol), viol[:2])
        else:
            for k, s in enumerate(stats):
                if s["work_s"] != tour_sums[k]:
                    kind = "model_error"
                    logger.error("[%s] 交叉断言失败：车%d 复算 work_s=%d != solver 回路值=%d",
                                 name, k + 1, s["work_s"], tour_sums[k])
                    break
            if kind is None:
                kind = "feasible"
                if len(routes) < n:
                    logger.warning("[%s] CP-SAT 解仅 %d/%d 条回路（空车）——更小 N 可能可行", name, len(routes), n)
                logger.info("[%s] CP-SAT 阶段F 得解：served=%d Tmax=%.0f s", name, sum(len(r) for r in routes),
                            max(s["work_s"] for s in stats))
                solver_info["stage"] = "F"
    else:
        kind = "unknown" if status != cp_model.MODEL_INVALID else "model_error"
        logger.info("[%s] CP-SAT 未决：%s（不构成不可行结论）", name, kind)
    if kind == "feasible" and elapsed <= 600 and budget - elapsed > 60:
        tmax_var = model.NewIntVar(0, CAP_S, "Tmax")
        for i in range(1, M + 1):
            model.Add(tmax_var >= ctx["t"][i] + time_s[i][0])
        model.Minimize(tmax_var)
        if add_route_hint(model, ctx, routes):
            solver2 = cp_model.CpSolver()
            solver2.parameters.max_time_in_seconds = int(budget - elapsed)
            solver2.parameters.num_search_workers = workers
            solver2.parameters.random_seed = seed
            t0 = time.perf_counter()
            status2 = solver2.Solve(model)
            e2 = time.perf_counter() - t0
            logger.info("[%s] CP-SAT 阶段O：status=%s 耗时 %.1f s", name, status2.name, e2)
            if status2 in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                routes2, tour_sums2 = extract_routes(solver2, ctx)
                stats2, arch2, viol2 = verified_stats(case, dist_km, time_s, routes2, name)
                if not viol2:
                    routes, stats, viol = routes2, stats2, viol2
                    solver_info.update({
                        "stage": "O", "status": status2.name,
                        "elapsed_s": round(e2, 3),
                        "objective": int(solver2.ObjectiveValue()) if status2 == cp_model.OPTIMAL else None,
                    })
                    logger.info("[%s] CP-SAT 阶段O 改进：Tmax=%.0f s", name, max(s["work_s"] for s in stats))
        else:
            logger.info("[%s] 阶段O 跳过（hint 注入失败）", name)
    checkpoint = {
        "case": name, "n": n, "track": "cpsat", "kind": kind, "M": M,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_lb": lb["n_lb"],
        "served": sum(len(r) for r in routes) if routes is not None else None,
        "routes": routes, "stats": stats,
        "Tmax_s": max(s["work_s"] for s in stats) if stats else None,
        "sum_T_s": sum(s["work_s"] for s in stats) if stats else None,
        "verification": {"n_violations": len(viol)} if viol else {"n_violations": 0},
        "solver": solver_info,
        "witness_at_infeasible": witness[1] if (kind == "model_error" and witness) else None,
    }
    write_atomic(cp_path(name, n, "cpsat"), checkpoint)
    logger.info("[%s] CP-SAT checkpoint 已写入（kind=%s）", name, kind)
    return checkpoint
def write_cp(cand, name, n, track, M, lb, kind):
    obj = {
        "case": name, "n": n, "track": track, "kind": kind, "M": M,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_lb": lb["n_lb"], "served": cand["served"], "routes": cand["routes"],
        "stats": cand["stats"], "Tmax_s": cand["Tmax_s"], "sum_T_s": cand["sum_T_s"],
        "verification": {"n_violations": len(cand["viol"]), "violations": cand["viol"]},
        "provenance": cand["provenance"],
        "solver": {"strategy": cand["solver"].get("strategy"), "objective": cand["solver"].get("objective")},
    }
    write_atomic(cp_path(name, n, track), obj)
def run_track_ortools(case, dist_km, time_s, n: int, budget: int, seed: int,
                      screen_s: int = 60, chunk_s: int = 300):
    name, M = case["name"], len(case["tasks"])
    lb = compute_lower_bounds(case)
    apath = OUT_DIR / f"q1_solution_{name}.json"
    baseline_arch = json.loads(apath.read_text(encoding="utf-8")) if apath.exists() else None
    cands = build_candidates(case, time_s, n, seed, baseline_arch)
    logger.info("[%s] 轨道A N=%d 候选起点 %d 个", name, n, len(cands))
    best = None
    s_screen = min(screen_s, max(10, budget // max(len(cands), 1)))
    deadline = time.perf_counter() + budget
    curve = []
    for routes, prov in cands:
        if time.perf_counter() + 5 > deadline:
            break
        res, _, _, _, _ = solve_n(case, time_s, n, s_screen, init_routes=routes)
        if res is None:
            continue
        stats, arch, viol = verified_stats(case, dist_km, time_s, res["routes"], name)
        cand = {"served": res["served"],
                "Tmax_s": max(s["work_s"] for s in stats) if stats else 0,
                "sum_T_s": sum(s["work_s"] for s in stats) if stats else 0,
                "routes": res["routes"], "stats": stats, "viol": viol,
                "provenance": prov, "solver": res}
        if best is None or better(cand, best):
            best = cand
            write_cp(best, name, n, "ortools", M, lb, "partial" if best["served"] < M else "feasible")
            logger.info("[%s] 轨道A 筛选改进（%s）：served=%d/%d Tmax=%.0f s", name, prov,
                        best["served"], M, best["Tmax_s"])
    while time.perf_counter() + 10 < deadline:
        warm = best["routes"] if best is not None else None
        rem = min(chunk_s, max(30, int(deadline - time.perf_counter())))
        res, crv, _, _, _ = solve_n(case, time_s, n, rem, want_curve=True, init_routes=warm)
        curve += [(round(t, 1), v) for t, v in crv]
        if res is None:
            continue
        stats, arch, viol = verified_stats(case, dist_km, time_s, res["routes"], name)
        cand = {"served": res["served"],
                "Tmax_s": max(s["work_s"] for s in stats) if stats else 0,
                "sum_T_s": sum(s["work_s"] for s in stats) if stats else 0,
                "routes": res["routes"], "stats": stats, "viol": viol,
                "provenance": "refine", "solver": res}
        if best is None or better(cand, best):
            best = cand
            write_cp(best, name, n, "ortools", M, lb, "partial" if best["served"] < M else "feasible")
            logger.info("[%s] 轨道A 精修改进：served=%d/%d Tmax=%.0f s", name, best["served"], M, best["Tmax_s"])
    if curve:
        cpath = LOG_DIR / f"tight_curve_{name}_N{n}.csv"
        pd.DataFrame(curve, columns=["elapsed_s", "objective"]).to_csv(cpath, index=False)
    final_kind = "feasible" if (best is not None and best["served"] == M and not best["viol"]) else "partial"
    if best is not None:
        write_cp(best, name, n, "ortools", M, lb, final_kind)
        logger.info("[%s] 轨道A 结束：served=%d/%d Tmax=%.0f s（kind=%s）", name, best["served"], M,
                    best["Tmax_s"], final_kind)
    else:
        logger.info("[%s] 轨道A 结束：无任何解", name)
    return best
def run_track_compress(case, dist_km, time_s, n: int, budget: int, seed: int, chunk_s: int = 300):
    name, M = case["name"], len(case["tasks"])
    lb = compute_lower_bounds(case)
    apath = OUT_DIR / f"q1_solution_{name}.json"
    baseline = json.loads(apath.read_text(encoding="utf-8"))
    base_routes = [u["task_seq"] for u in baseline["uavs"] if u["task_seq"]]
    cands = [(base_routes, "baseline_warm")] + build_candidates(case, time_s, n, seed, baseline_arch=baseline)
    deadline = time.perf_counter() + budget
    best = None
    for routes, prov in cands:
        if time.perf_counter() + 5 > deadline:
            break
        stats, arch, viol = verified_stats(case, dist_km, time_s, routes, name)
        if not viol and sum(len(r) for r in routes) == M:
            cand = {"served": M, "Tmax_s": arch["Tmax_s"], "sum_T_s": sum(s["work_s"] for s in stats),
                    "routes": routes, "stats": stats, "viol": viol, "provenance": prov, "solver": {}}
            if best is None or better(cand, best):
                best = cand
    logger.info("[%s] 压缩轨道 N=%d 起点构造：best Tmax=%.0f s（基线 %.0f s）", name, n,
                best["Tmax_s"] if best else -1, baseline["Tmax_s"])
    while time.perf_counter() + 10 < deadline:
        rem = min(chunk_s, max(30, int(deadline - time.perf_counter())))
        res, _, _, _, _ = solve_n(case, time_s, n, rem, init_routes=best["routes"] if best else base_routes)
        if res is None or res["served"] != M:
            continue
        stats, arch, viol = verified_stats(case, dist_km, time_s, res["routes"], name)
        if viol:
            continue
        cand = {"served": M, "Tmax_s": arch["Tmax_s"], "sum_T_s": sum(s["work_s"] for s in stats),
                "routes": res["routes"], "stats": stats, "viol": viol,
                "provenance": "compress_refine", "solver": res}
        if better(cand, best):
            best = cand
            write_cp(best, name, n, "compress", M, lb, "feasible")
            logger.info("[%s] 压缩改进：Tmax=%.0f s ΣT=%.0f s", name, best["Tmax_s"], best["sum_T_s"])
    if best is not None:
        write_cp(best, name, n, "compress", M, lb, "feasible")
        target_note = ""
        if name == "Case3":
            target_note = "（靶值 %d s，%s）" % (C3_TARGET_S, "达成" if best["Tmax_s"] <= C3_TARGET_S else "未达")
        logger.info("[%s] 压缩轨道结束：Tmax=%.0f s ΣT=%.0f s%s", name, best["Tmax_s"], best["sum_T_s"], target_note)
    return best
def _synthetic(name, spec):
    points = [{"pid": p[0], "x": p[1], "y": p[2], "level": p[3]} for p in spec]
    tasks = []
    for p in points:
        for _ in range(LEVEL_VISITS[p["level"]]):
            tasks.append({"task_id": len(tasks) + 1, "pid": p["pid"], "x": p["x"], "y": p["y"]})
    return {"name": name, "points": points, "tasks": tasks}
def run_smoke() -> int:
    fails = []
    def check(cond, label, extra=""):
        if cond:
            logger.info("SMOKE PASS: %s", label)
        else:
            logger.error("SMOKE FAIL: %s %s", label, extra)
            fails.append(label)
    for name, n in TIGHT_N.items():
        case = load_case(name)
        _, time_s = build_matrices(case)
        M = len(case["tasks"])
        banned = EXPECTED_ADJ[name]
        model, ctx = build_cp_model(case, time_s, n)
        expect_arcs = M * (M - 1) - banned + 2 * M
        check(ctx["n_arcs"] == expect_arcs, f"S3 弧数 {name} N={n}", f"{ctx['n_arcs']} != {expect_arcs}")
    syn_a = _synthetic("SynthA", [(1, 100, 0, "III"), (2, 0, 100, "III")])
    da, ta = build_matrices(syn_a)
    cpa = run_track_cpsat(syn_a, da, ta, 1, 60, SEED_DEFAULT, 4)
    ok_a = cpa["kind"] == "feasible" and cpa["routes"] is not None and sum(len(r) for r in cpa["routes"]) == 2
    if ok_a:
        w = max(s["work_s"] for s in cpa["stats"])
        expect = ta[0][1] + SERVICE_S + ta[1][2] + SERVICE_S + ta[2][0]
        check(w == expect, "S4a 合成小例 N=1 不同点", f"work={w} != {expect}")
    else:
        check(False, "S4a 合成小例 N=1 不同点", f"kind={cpa['kind']}")
    syn_b = _synthetic("SynthB", [(1, 100, 0, "II")])
    db, tb = build_matrices(syn_b)
    cpb1 = run_track_cpsat(syn_b, db, tb, 1, 60, SEED_DEFAULT, 4)
    check(cpb1["kind"] == "infeasibility_proof", "S4b 同点 N=1 必须 INFEASIBLE", f"kind={cpb1['kind']}")
    cpb2 = run_track_cpsat(syn_b, db, tb, 2, 60, SEED_DEFAULT, 4)
    check(cpb2["kind"] == "feasible", "S4c 同点 N=2 必须 FEASIBLE", f"kind={cpb2['kind']}")
    case1 = load_case("Case1")
    dist1, time1 = build_matrices(case1)
    arch1 = json.loads((OUT_DIR / "q1_solution_Case1.json").read_text(encoding="utf-8"))
    routes1 = [u["task_seq"] for u in arch1["uavs"] if u["task_seq"]]
    stats1, a1, viol1 = verified_stats(case1, dist1, time1, routes1, "Case1")
    check(len(viol1) == 0 and stats1 is not None and a1["N"] == 4, "S5 基线管线", f"viol={len(viol1)}")
    base1 = json.loads((OUT_DIR / "q1_solution_Case1.json").read_text(encoding="utf-8"))
    cp1 = run_track_cpsat(case1, dist1, time1, 4, 600, SEED_DEFAULT, WORKERS_DEFAULT, baseline_arch=base1)
    check(cp1["kind"] == "feasible" and cp1["served"] == len(case1["tasks"]),
          "S2 Case1 N=4 CP-SAT", f"kind={cp1['kind']} served={cp1['served']}")
    case2 = load_case("Case2")
    dist2, time2 = build_matrices(case2)
    base2 = json.loads((OUT_DIR / "q1_solution_Case2.json").read_text(encoding="utf-8"))
    cp2 = run_track_cpsat(case2, dist2, time2, 2, 300, SEED_DEFAULT, WORKERS_DEFAULT, baseline_arch=base2)
    check(cp2["kind"] == "feasible" and cp2["served"] == len(case2["tasks"]),
          "S1 Case2 N=2 CP-SAT", f"kind={cp2['kind']} served={cp2['served']}")
    logger.info("SMOKE 汇总：%s", "全部 PASS" if not fails else f"{len(fails)} 项 FAIL：{fails}")
    return 1 if fails else 0
def run_orchestrate() -> int:
    py = sys.executable
    script = str(HERE / "tight_search.py")
    jobs = []
    for case, n in TIGHT_N.items():
        jobs += [(case, n, "ortools"), (case, n, "cpsat")]
    for case, n in COMPRESS_N.items():
        jobs.append((case, n, "compress"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for case, n, track in jobs:
        lf = open(LOG_DIR / f"tight_{case}_N{n}_{track}_{ts}.log", "w", encoding="utf-8")
        cmd = [py, script, "--case", case, "--n", str(n), "--track", track,
               "--budget", "1800", "--seed", str(SEED_DEFAULT)]
        if track == "cpsat":
            cmd += ["--workers", str(WORKERS_DEFAULT)]
        logger.info("拉起：%s", " ".join(cmd))
        procs.append((f"{case}_N{n}_{track}", subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf))
    bad = []
    for label, proc, lf in procs:
        rc = proc.wait()
        lf.close()
        if rc != 0:
            bad.append((label, rc))
            logger.error("子进程 %s 退出码 %d", label, rc)
        else:
            logger.info("子进程 %s 完成", label)
    logger.info("编排结束：%d/%d 成功", len(procs) - len(bad), len(procs))
    return 1 if bad else 0
def run_promote() -> int:
    CPK_DIR.mkdir(parents=True, exist_ok=True)
    archives = {}
    infeasible_proofs = {name: 0 for name in CASES}
    notes = []
    for name in CASES:
        case = load_case(name)
        dist_km, time_s = build_matrices(case)
        lb = compute_lower_bounds(case)
        M = len(case["tasks"])
        candidates = []
        apath = OUT_DIR / f"q1_solution_{name}.json"
        base = json.loads(apath.read_text(encoding="utf-8"))
        base_routes = [u["task_seq"] for u in base["uavs"] if u["task_seq"]]
        stats, _, viol = verified_stats(case, dist_km, time_s, base_routes, name)
        if viol:
            logger.error("[%s] 基线档案重验证 %d 违规（异常），基线弃用", name, len(viol))
        else:
            candidates.append({
                "N": base["N"], "Tmax_s": base["Tmax_s"],
                "sum_T_s": sum(s["work_s"] for s in stats),
                "routes": base_routes, "stats": stats, "viol": [],
                "source": "baseline", "solver": {"strategy": base.get("first_solution_strategy")},
            })
        for cpf in list(CPK_DIR.glob(f"tight_checkpoint_{name}_N*.json")) + \
                   list(CPK_DIR.glob(f"tight_compress_{name}_N*.json")):
            cp = json.loads(cpf.read_text(encoding="utf-8"))
            if cp.get("kind") == "infeasibility_proof":
                if cp.get("solver", {}).get("status") == "INFEASIBLE":
                    infeasible_proofs[name] = max(infeasible_proofs[name], cp["n"])
                continue
            if cp.get("kind") != "feasible" or cp.get("routes") is None:
                continue
            stats, _, viol = verified_stats(case, dist_km, time_s, cp["routes"], name)
            if viol:
                logger.warning("[%s] checkpoint %s 重验证 %d 违规，弃用", name, cpf.name, len(viol))
                continue
            candidates.append({
                "N": cp["n"], "Tmax_s": max(s["work_s"] for s in stats),
                "sum_T_s": sum(s["work_s"] for s in stats),
                "routes": cp["routes"], "stats": stats, "viol": [],
                "source": f"checkpoint:{cpf.name}", "solver": cp.get("solver", {}),
            })
        for cand in candidates:
            if cand["N"] <= infeasible_proofs[name]:
                logger.error("[%s] CP-SAT 曾证 N=%d 不可行，但存在 N=%d 可行候选（%s）——证明作废，不用于下界",
                             name, infeasible_proofs[name], cand["N"], cand["source"])
                infeasible_proofs[name] = cand["N"] - 1
        candidates.sort(key=lambda c: (c["N"], c["Tmax_s"], c["sum_T_s"]))
        best = candidates[0]
        lb_eff = max(lb["n_lb"], infeasible_proofs[name] + 1)
        optimality = "PROVEN_BY_LOWER_BOUND" if best["N"] == lb_eff else "BEST_FEASIBLE_NOT_PROVEN"
        if best["N"] == lb_eff and best["source"] != "baseline":
            logger.info("[%s] 下界提升生效：lb_eff=%d，选中 N=%d -> %s", name, lb_eff, best["N"], optimality)
        n_active = len([r for r in best["routes"] if r])
        if n_active < best["N"]:
            notes.append(f"{name}：选中解含空车（有效 {n_active}/{best['N']}），更小 N 可能可行")
            logger.warning("[%s] %s", name, notes[-1])
        stats = [s for s in best["stats"]]
        tmax_s = max(s["work_s"] for s in stats)
        tmin_s = min(s["work_s"] for s in stats)
        arch = {
            "case": name, "status": "SOLVED", "complete": True,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N": best["N"], "Tmax_s": tmax_s, "Tmin_s": tmin_s,
            "Tmax_h": tmax_s / 3600.0, "Tmin_h": tmin_s / 3600.0,
            "optimality": optimality,
            "lower_bounds": dict(lb, lb_eff=lb_eff,
                                 lb_source=("computed+infeasibility_proof" if infeasible_proofs[name] >= lb["n_lb"]
                                            else "computed"),
                                 infeasible_proof_n=infeasible_proofs[name]),
            "adj_constraints": EXPECTED_ADJ.get(name, 0),
            "first_solution_strategy": best["solver"].get("strategy", "mixed"),
            "solver_objective": best["solver"].get("objective"),
            "params": {"speed_kmh": solve_q1.SPEED_KMH, "unit_km": solve_q1.UNIT_KM,
                       "service_s": SERVICE_S, "cap_s": CAP_S, "note": "tight_search promote"},
            "uavs": [{"uav_id": i + 1, **s} for i, s in enumerate(stats)],
            "provenance": {
                "source": "tight_search.promote", "selected_from": best["source"],
                "baseline_was": {"N": base["N"], "Tmax_h": base["Tmax_h"]},
                "candidates_considered": len(candidates),
                "infeasible_proofs": infeasible_proofs[name],
            },
        }
        archives[name] = arch
        logger.info("[%s] promote 选中 %s：N=%d Tmax=%.4f h（基线 N=%d %.4f h）%s", name, best["source"],
                    best["N"], tmax_s / 3600.0, base["N"], base["Tmax_h"], optimality)
    write_result1(archives)
    write_summary(archives)
    viol = []
    for name in CASES:
        viol += verify_case(load_case(name), archives[name])
    viol += check_result1_xlsx(archives)
    write_reports(viol)
    for n_ in notes:
        logger.warning("注意：%s", n_)
    logger.info("promote 完成：违规 %d 条", len(viol))
    return 1 if viol else 0
def main() -> int:
    ap = argparse.ArgumentParser(description="问题1 紧档搜索与压缩")
    ap.add_argument("--case", choices=CASES + ["SynthA", "SynthB"], default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--track", choices=["ortools", "cpsat", "compress"], default=None)
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--screen-s", type=int, default=60)
    ap.add_argument("--chunk-s", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--orchestrate", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.case or 'main'}_{args.n or 0}_{args.track or 'cmd'}"
    setup_logging(ts, suffix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        return run_smoke()
    if args.orchestrate:
        return run_orchestrate()
    if args.promote:
        return run_promote()
    if args.case is None or args.n is None or args.track is None:
        logger.error("需要 --case/--n/--track，或使用 --smoke/--orchestrate/--promote")
        return 2
    case = load_case(args.case)
    dist_km, time_s = build_matrices(case)
    baseline_arch = None
    if args.track == "cpsat":
        apath = OUT_DIR / f"q1_solution_{args.case}.json"
        if apath.exists():
            baseline_arch = json.loads(apath.read_text(encoding="utf-8"))
        run_track_cpsat(case, dist_km, time_s, args.n, args.budget, args.seed, args.workers,
                        baseline_arch=baseline_arch)
    elif args.track == "ortools":
        run_track_ortools(case, dist_km, time_s, args.n, args.budget, args.seed, args.screen_s, args.chunk_s)
    else:
        run_track_compress(case, dist_km, time_s, args.n, args.budget, args.seed, args.chunk_s)
    return 0
if __name__ == "__main__":
    sys.exit(main())
