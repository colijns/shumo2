# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题3 核心：首次导通数量 N_c（嵌套共同样本增量蒙特卡洛）。

建模口径（与 docs/问题3_新版.md 同步）：
- 一次试验先生成 N_max 根带固定顺序的完整介质序列，逐根加入；
- 每加入一根圆柱，先边界截断（PBC 平移），片段作为独立几何节点加入
  接触网络；同源片段不预合并，只有实际表面距离 ≤ 1.8 nm 才建边
  （假设二，与 Q2 geometry 新口径一致）；
- 记录首次形成左右导电通路时的完整介质数量 N_c = min{N: Y_s(N)=1}；
- 增量并查集早停：已导通则更高数量必然导通，直接停止该次试验；
- same_source=True 为敏感性口径（假设一）：同一根圆柱的截断片段
  自动电连续（并查集预联合），跨壁圆柱可桥接左右电极。

复用 Q2/geometry.py（只读）：clip_batch、electrode_dist、aabb 预筛、
segment_distance_batch、_fragment_gjk、generate_cylinders、V_A/V_BOX、
DELTA/R/AXIS_THRESH；复用 Q1/core.py 的 UnionFind。
"""

import os
import sys

import numpy as np

_Q2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2')
_Q1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q1')
for _p in (_Q2, _Q1):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geometry as geo  # noqa: E402

from core import UnionFind  # noqa: E402


def crossing_count(c, u, h):
    """精确跨壁判定（docs/问题3_新版.md §7）：存在某轴 |c_k|+e_k > L/2。

    e_{i,k} = h_i|u_{i,k}| + r√(1-u_{i,k}²)（平端圆柱精确投影半宽）。
    返回 (n_cross, n_cyl)。
    """
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
    """单次试验：增量逐根加入完整介质，返回首次导通数量 N_c。

    参数：
        c, u, h: 完整介质序列（N_max 根，固定生成顺序）。
        same_source: True 走敏感性口径（同源片段自动电连续）。
    返回 dict：
        {n_contact, conductive, n_fragments, n_edges, n_gjk, n_cross,
         n_same_contacts}；n_contact=None 表示 N_max 内未导通。
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n_cyl = len(c)
    if n_cyl == 0:
        return {'n_contact': None, 'conductive': False, 'n_fragments': 0,
                'n_edges': 0, 'n_gjk': 0, 'n_cross': 0, 'n_same_contacts': 0}

    # 一次性边界截断（clip 无跨圆柱依赖），再按 cyl_idx 逐根加入
    frag = geo.clip_batch(c, u, h)
    p1s, p2s, cyl_idx = frag['p1s'], frag['p2s'], frag['cyl_idx']
    nf = len(p1s)
    dL, dR = geo.electrode_dist(p1s, p2s)

    # 每根圆柱的片段索引（cyl_idx 单调分组）
    cyl_frag_idx = []
    for i in range(n_cyl):
        idxs = np.nonzero(cyl_idx == i)[0]
        if len(idxs):
            cyl_frag_idx.append(idxs)

    uf = UnionFind(nf + 2)
    s_node, t_node = nf, nf + 1
    lo = np.minimum(p1s, p2s) - geo.R
    hi = np.maximum(p1s, p2s) + geo.R

    added = 0              # 已加入片段数
    n_edges = n_gjk = 0
    n_same_contacts = 0

    for i, idxs in enumerate(cyl_frag_idx):
        # 电极边：片段到左右带电面 ≤ delta 分别连 S/T
        for f in idxs:
            f = int(f)
            if dL[f] <= delta:
                uf.union(s_node, f)
            if dR[f] <= delta:
                uf.union(t_node, f)

        # 同源片段自动电连续（敏感性口径，假设一）
        if same_source:
            for a, b in zip(idxs[:-1], idxs[1:]):
                uf.union(int(a), int(b))

        # 同圆柱片段内部（无同源预合并时仍按实际距离判连，k ≤ 4 直接循环）
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

        # 新片段 vs 全部已有片段（含本圆柱更早片段）：AABB 预筛 → 胶囊 → GJK
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
    """M 次独立试验的首次导通数量样本。

    每次试验：固定生成顺序的 N_max 根完整介质 → 增量早停。
    返回 (n_contact_list, stats_dict)。
    """
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
    # 冒烟：N_max=900，M=20，固定 seed
    import time
    t0 = time.perf_counter()
    contacts, stats = simulate_trials(900, m=20, seed=42)
    arr = np.array([v if v is not None else 900 for v in contacts], dtype=int)
    print(f'M=20: 首次导通数量 = {arr.tolist()}')
    print(f'导通 {20 - stats["n_never"]}/20, 未导通 {stats["n_never"]}')
    print(f'跨壁圆柱平均 = {stats["n_cross"] / stats["n_trials"]:.1f} 根/试验')
    print(f'耗时 {time.perf_counter() - t0:.1f}s')
