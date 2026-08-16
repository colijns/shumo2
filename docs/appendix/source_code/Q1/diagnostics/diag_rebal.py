# -*- coding: utf-8 -*-
"""诊断 v10：插桩新构造管线（Case3, N=4），打印每轮各簇负荷。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1 import (load_case, _ctx_maps, _chunk_seq, _route_work, CAP_S, SERVICE_S)  # noqa: E402
import math  # noqa: E402

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "Case3")
n_uav = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tasks_of, pid_of, coords = _ctx_maps(case)
pts = case["points"]
n = len(pts)

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
    for _ in range(30):
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
            new_centers.append((sum(p["x"] for p in members) / len(members), sum(p["y"] for p in members) / len(members)))
        if new_centers == centers:
            break
        centers = new_centers
    if failed:
        print(f"travel_frac={travel_frac}: Lloyd 分配失败")
        continue
    chunks = [[p for p in pts if assign[p["pid"]] == c] for c in range(n_uav)]
    print(f"travel_frac={travel_frac}: 初始簇大小={[len(c) for c in chunks]}")
    for it in range(30):
        seqs = [_chunk_seq(ch, tasks_of, pid_of, coords) for ch in chunks]
        if any(s is None for s in seqs):
            print(f"  第{it}轮: 构造失败（NN/回插失败）")
            break
        works = [_route_work(s, pid_of, coords) for s in seqs]
        print(f"  第{it}轮: 工作时间(h)={[round(x / 3600, 3) for x in works]} 上限={CAP_S / 3600}")
        if all(x <= CAP_S for x in works):
            print("  ✓ 全部满足上限")
            break
        k = max(range(n_uav), key=lambda i: works[i])
        others = [works[i] for i in range(n_uav) if i != k]
        best_score, best_move = max(works), None
        for p in chunks[k]:
            for k2 in range(n_uav):
                if k2 == k:
                    continue
                src = [q for q in chunks[k] if q is not p]
                dst = chunks[k2] + [p]
                if not src:
                    continue
                s2, s3 = _chunk_seq(src, tasks_of, pid_of, coords), _chunk_seq(dst, tasks_of, pid_of, coords)
                if s2 is None or s3 is None:
                    continue
                w2, w3 = _route_work(s2, pid_of, coords), _route_work(s3, pid_of, coords)
                score = max([w2, w3] + others)
                if score < best_score - 1e-9:
                    best_score, best_move = score, (p, k2)
        if best_move is None:
            print("  无改进移动可用，陷入平台")
            break
        p, k2 = best_move
        print(f"  移动 Point {p['pid']}: 簇{k}→簇{k2} (最大负荷 {max(works)/3600:.3f}h → {best_score/3600:.3f}h)")
        chunks[k].remove(p)
        chunks[k2].append(p)
