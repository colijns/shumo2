# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import os
import sys

import numpy as np

_Q2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2')
_Q1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q1')
for _p in (_Q2, _Q1):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geometry as geo

from core import UnionFind

def crossing_count(c, u, h):
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(c)
    if n == 0:
        return 0, 0
    e = h[:, None] * np.abs(u) + geo.R * np.sqrt(
        np.maximum(1.0 - u * u, 0.0))
    cross = np.any(np.abs(c) + e > geo.HALF_L, axis=1)
    return int(cross.sum()), n

def first_contact_n(c, u, h, same_source=False, delta=geo.DELTA):
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n_cyl = len(c)
    if n_cyl == 0:
        return {'n_contact': None, 'conductive': False, 'n_fragments': 0,
                'n_edges': 0, 'n_gjk': 0, 'n_cross': 0, 'n_same_contacts': 0}

    frag = geo.clip_batch(c, u, h)
    p1s, p2s, cyl_idx = frag['p1s'], frag['p2s'], frag['cyl_idx']
    nf = len(p1s)
    dL, dR = geo.electrode_dist(p1s, p2s)

    cyl_frag_idx = []
    for i in range(n_cyl):
        idxs = np.nonzero(cyl_idx == i)[0]
        if len(idxs):
            cyl_frag_idx.append(idxs)

    uf = UnionFind(nf + 2)
    s_node, t_node = nf, nf + 1
    lo = np.minimum(p1s, p2s) - geo.R
    hi = np.maximum(p1s, p2s) + geo.R

    added = 0
    n_edges = n_gjk = 0
    n_same_contacts = 0

    for i, idxs in enumerate(cyl_frag_idx):

        for f in idxs:
            f = int(f)
            if dL[f] <= delta:
                uf.union(s_node, f)
            if dR[f] <= delta:
                uf.union(t_node, f)

        if same_source:
            for a, b in zip(idxs[:-1], idxs[1:]):
                uf.union(int(a), int(b))

        for m in range(len(idxs)):
            fi = int(idxs[m])
            for n2 in range(m + 1, len(idxs)):
                fj = int(idxs[n2])
                if geo.segment_distance_batch(
                        p1s[fi:fi + 1], p2s[fi:fi + 1],
                        p1s[fj:fj + 1], p2s[fj:fj + 1])[0] <= geo.AXIS_THRESH:
                    dg = geo._fragment_gjk(p1s[fi], p2s[fi], p1s[fj], p2s[fj])
                    n_gjk += 1
                    if dg <= delta + 1e-6:
                        uf.union(fi, fj)
                        n_edges += 1
                        if cyl_idx[fi] == cyl_idx[fj]:
                            n_same_contacts += 1

        if added:
            new_lo = lo[idxs]
            new_hi = hi[idxs]
            ok = np.ones((len(idxs), added), dtype=bool)
            for a in range(3):
                ok &= (new_lo[:, None, a] - hi[None, :added, a] <= geo.AXIS_THRESH)
                ok &= (lo[None, :added, a] - new_hi[:, None, a] <= geo.AXIS_THRESH)
            ii, jj = np.nonzero(ok)
            if len(ii):
                d = geo.segment_distance_batch(
                    p1s[idxs[ii]], p2s[idxs[ii]],
                    p1s[jj], p2s[jj])
                cand = d <= geo.AXIS_THRESH
                for m in np.nonzero(cand)[0]:
                    fi, fj = int(idxs[ii[m]]), int(jj[m])
                    dg = geo._fragment_gjk(p1s[fi], p2s[fi], p1s[fj], p2s[fj])
                    n_gjk += 1
                    if dg <= delta + 1e-6:
                        uf.union(fi, fj)
                        n_edges += 1

        added += len(idxs)

        if uf.connected(s_node, t_node):
            n_cross, _ = crossing_count(c[:i + 1], u[:i + 1], h[:i + 1])
            return {'n_contact': i + 1, 'conductive': True,
                    'n_fragments': nf, 'n_edges': n_edges, 'n_gjk': n_gjk,
                    'n_cross': n_cross, 'n_same_contacts': n_same_contacts}

    n_cross, _ = crossing_count(c, u, h)
    return {'n_contact': None, 'conductive': False,
            'n_fragments': nf, 'n_edges': n_edges, 'n_gjk': n_gjk,
            'n_cross': n_cross, 'n_same_contacts': n_same_contacts}

def simulate_trials(n_max, m=2000, seed=42, same_source=False):
    rng = np.random.default_rng(seed)
    contacts = []
    total_cross = 0
    total_same = 0
    for _ in range(m):
        c, u, h = geo.generate_cylinders(n_max, rng)
        res = first_contact_n(c, u, h, same_source=same_source)
        contacts.append(res['n_contact'])
        total_cross += res['n_cross']
        total_same += res['n_same_contacts']
    stats = {
        'n_cross': total_cross,
        'n_same_contacts': total_same,
        'n_trials': m,
        'n_never': sum(1 for v in contacts if v is None),
    }
    return contacts, stats

if __name__ == '__main__':

    import time
    t0 = time.perf_counter()
    contacts, stats = simulate_trials(900, m=20, seed=42)
    arr = np.array([v if v is not None else 900 for v in contacts], dtype=int)
    print(f'M=20: 首次导通数量 = {arr.tolist()}')
    print(f'导通 {20 - stats["n_never"]}/20, 未导通 {stats["n_never"]}')
    print(f'跨壁圆柱平均 = {stats["n_cross"] / stats["n_trials"]:.1f} 根/试验')
    print(f'耗时 {time.perf_counter() - t0:.1f}s')
