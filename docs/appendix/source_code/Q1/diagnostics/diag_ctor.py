# -*- coding: utf-8 -*-
"""诊断 v9：验证重写后的 sweep/kmeans 双构造器（四算例 × N 候选）。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1 import (load_case, build_matrices, compute_lower_bounds, sweep_routes,
                      kmeans_routes, evaluate, span_of, LEVEL_VISITS)  # noqa: E402

for name in ["Case1", "Case2", "Case3", "Case4"]:
    case = load_case(name)
    dist_km, time_s = build_matrices(case)
    lb = compute_lower_bounds(case)
    for N in range(lb["n_lb"], lb["n_lb"] + 3):
        for cname, ctor in (("sweep", sweep_routes), ("kmeans", kmeans_routes)):
            routes = ctor(case, N)
            if routes is None:
                print(f"{name} N={N} {cname}: 失败", flush=True)
                continue
            stats = evaluate(case, dist_km, time_s, routes)
            tmax, tmin, _ = span_of(stats)
            pids = [[case["tasks"][t - 1]["pid"] for t in r] for r in routes]
            bad_adj = sum(1 for r in pids for a, b in zip(r, r[1:]) if a == b)
            cnt = Counter(p for r in pids for p in r)
            ok = all(cnt[p["pid"]] == LEVEL_VISITS[p["level"]] for p in case["points"])
            print(f"{name} N={N} {cname}: 成功 Tmax={tmax/3600:.4f}h Tmin={tmin/3600:.4f}h "
                  f"相邻同点={bad_adj} 次数正确={ok}", flush=True)
