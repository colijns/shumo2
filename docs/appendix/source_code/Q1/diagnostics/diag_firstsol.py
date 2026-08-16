# -*- coding: utf-8 -*-
"""诊断 v7：阶段1可行性搜索用 min-total（span系数=0）替代 min-max，测紧 N 档。"""
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1 import load_case, build_matrices, compute_lower_bounds  # noqa: E402

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: E402

FS = routing_enums_pb2.FirstSolutionStrategy
BAN = 10_000_000
DROP = 100_000_000


def run(case, time_s, N, span_coeff, budget):
    M = len(case["tasks"])
    same = set()
    for a in range(1, M + 1):
        for b in range(1, M + 1):
            if a != b and case["tasks"][a - 1]["pid"] == case["tasks"][b - 1]["pid"]:
                same.add((a, b))
    manager = pywrapcp.RoutingIndexManager(M + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(fi, ti):
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        c = time_s[f][t] + (300 if 1 <= t <= M else 0)
        return c + BAN if (f, t) in same else c

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, 32400, True, "Time")
    if span_coeff:
        routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(span_coeff)
    for t in range(1, M + 1):
        routing.AddDisjunction([manager.NodeToIndex(t)], DROP)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = FS.ALL_UNPERFORMED
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(budget)
    t0 = _time.perf_counter()
    sol = routing.SolveWithParameters(p)
    dt = _time.perf_counter() - t0
    served, worst = 0, 0
    if sol is not None:
        for v in range(N):
            idx, seq = routing.Start(v), []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node:
                    seq.append(node)
                idx = sol.Value(routing.NextVar(idx))
            served += len(seq)
            path = [0] + seq + [0]
            work = sum(time_s[a][b] for a, b in zip(path, path[1:])) + 300 * len(seq)
            worst = max(worst, work)
    print(f"{case['name']} N={N} span_coeff={span_coeff}: served={served}/{M} "
          f"Tmax={worst / 3600:.4f}h ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    tests = [("Case1", 3), ("Case3", 3), ("Case3", 4), ("Case4", 3), ("Case4", 4)]
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for cname, N in tests:
        case = load_case(cname)
        _, time_s = build_matrices(case)
        run(case, time_s, N, 0, budget)
