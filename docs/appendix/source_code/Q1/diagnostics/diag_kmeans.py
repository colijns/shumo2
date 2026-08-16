# -*- coding: utf-8 -*-
"""诊断 v8：插桩 kmeans 构造过程（Case1, N=3），打印每轮各簇负荷。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1 import (load_case, _ctx_maps, _expand_rounds, _two_opt_ban,
                      _route_work, _chunks_to_routes, _chunk_work, CAP_S, SERVICE_S)  # noqa: E402

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "Case1")
n_uav = int(sys.argv[2]) if len(sys.argv) > 2 else 3
tasks_of, pid_of, coords = _ctx_maps(case)
pts = case["points"]
n = len(pts)

by_angle = sorted(pts, key=lambda p: math.atan2(p["y"], p["x"]))
centers = []
for k in range(n_uav):
    seg = by_angle[k * n // n_uav:(k + 1) * n // n_uav] or by_angle[-1:]
    centers.append((sum(p["x"] for p in seg) / len(seg), sum(p["y"] for p in seg) / len(seg)))
w = {p["pid"]: SERVICE_S * len(tasks_of[p["pid"]]) for p in pts}

travel_frac = 0.30
cap = CAP_S * (1.0 - travel_frac)
assign = {p["pid"]: 0 for p in pts}
failed = False
for it in range(30):
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
        print("Lloyd 分配失败（容量不足）")
        sys.exit(1)
    new_centers = []
    for c in range(n_uav):
        members = [p for p in pts if assign[p["pid"]] == c]
        new_centers.append((sum(p["x"] for p in members) / len(members), sum(p["y"] for p in members) / len(members)))
    if new_centers == centers:
        print(f"Lloyd 第{it}轮收敛")
        break
    centers = new_centers

chunks = [[p for p in pts if assign[p["pid"]] == c] for c in range(n_uav)]
for it in range(20):
    works = [_chunk_work(ch, tasks_of, pid_of, coords) for ch in chunks]
    sizes = [len(ch) for ch in chunks]
    print(f"再平衡第{it}轮: 簇大小={sizes} 工作时间(h)={[round(x / 3600, 3) for x in works]}")
    if all(x <= CAP_S for x in works):
        routes = _chunks_to_routes(chunks, tasks_of, pid_of, coords)
        print("全部满足上限, routes =", "OK" if routes else "None(展开失败)")
        break
    k = max(range(n_uav), key=lambda i: works[i])
    cand = sorted(chunks[k], key=lambda p: -math.hypot(p["x"] - centers[k][0], p["y"] - centers[k][1]))
    moved = False
    for p in cand:
        for k2 in sorted(range(n_uav), key=lambda i: math.hypot(p["x"] - centers[i][0], p["y"] - centers[i][1])):
            if k2 != k and chunks[k2]:
                print(f"  移动 Point {p['pid']} 从簇{k}到簇{k2}")
                chunks[k].remove(p)
                chunks[k2].append(p)
                moved = True
                break
        if moved:
            break
    if not moved or not chunks[k]:
        print("无法继续移动，放弃")
        break
