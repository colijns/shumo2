# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题2 几何内核：介质A 随机生成、边界截断（PBC 环面语义）、片段级导通判定。

建模口径（与 docs/问题2.md 同步）：
- 边界截断规则物理上等价于周期性边界条件：圆柱从左边界穿出、经平移从
  右边界进入，仍是同一根连续导体（空间为环面），电子可沿圆柱在左右两侧
  之间流动。因此同一原始圆柱的所有截断片段在图论上预先合并为一个连通组。
- 介质A 是平端圆柱（底面半径 r），非胶囊：胶囊距离 max(0, 轴距-2R)
  仅是表面距离的下界（胶囊 ⊃ 平端圆柱），只作安全拒绝预筛；建边一律
  GJK 精算（Q1 core.gjk_distance 同为平端圆柱模型，支撑
  h|v·u| + r√(1-(v·u)²)，与电极公式一致）。
- 电极距离：片段（短圆柱）x 向真实包络 x_min = c_x - h|u_x| - r√(1-u_x²)，
  与 Q1 core.electrode_contact 公式一致。

复用 Q1/core.py（只读）：常量 R/DELTA/L/HALF_L、UnionFind、标量
segment_distance、gjk_distance。不复用 analyze_group（整圆柱单节点 + 27 镜像
语义与片段级 + 同源合并不同构）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q1'))
from core import R, DELTA, L, HALF_L, UnionFind, segment_distance, gjk_distance  # noqa: E402

CYL_LEN = 5000.0            # 介质A 圆柱长度 (nm)
CYL_HALF_LEN = CYL_LEN / 2.0
V_A = np.pi * R ** 2 * CYL_LEN          # 单根介质A 体积 ≈ 1.4137e7 nm³
V_BOX = L ** 3                          # 微构体体积 = 1e12 nm³
AXIS_THRESH = 2.0 * R + DELTA           # 61.8：胶囊初筛阈值（轴距 > 61.8 必不连）
TOL = 1e-9                              # 平面交点/端点容差
BOX_TOL = 1e-6                          # 平移后 box 内断言容差


def generate_cylinders(n, rng):
    """随机生成 n 根介质A。返回 (c, u, h)。

    c: (n,3) 中心，三坐标独立均匀 U(-5000, 5000)；
    u: (n,3) 单位方向，三维各向同性（z=cosθ~U(-1,1), φ~U(0,2π)，
       u=(√(1-z²)cosφ, √(1-z²)sinφ, z)）；
    h: (n,) 半长，全 2500。
    """
    c = rng.uniform(-HALF_L, HALF_L, size=(n, 3))
    z = rng.uniform(-1.0, 1.0, size=(n,))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n,))
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    u = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)
    h = np.full(n, CYL_HALF_LEN, dtype=float)
    return c, u, h


def _crossing_ts(p1, p2):
    """轴线段 [p1,p2] 与 6 个边界平面的全部交点参数 t（排序去重）。

    每轴最多一次穿越（轴投影长 ≤ 5000 < L），总交点 ≤ 3。
    端点恰在平面上（t≈0 或 1）视为不穿越。
    """
    d = p2 - p1
    ts = []
    for a in range(3):
        if abs(d[a]) < 1e-12:
            continue
        for plane in (-HALF_L, HALF_L):
            if (p1[a] - plane) * (p2[a] - plane) < 0.0:  # 严格异侧
                t = (plane - p1[a]) / d[a]
                if TOL < t < 1.0 - TOL:
                    ts.append(t)
    ts.sort()
    # 去重：同一参数下可能同时穿越两轴（对角越界）
    out = []
    for t in ts:
        if not out or t - out[-1] > 1e-9:
            out.append(t)
    return out


def clip_cylinder(p1, p2):
    """单根轴线按边界平面交点切分 + 越界子段平移回 box。

    p1, p2 为 (3,) 轴线段端点（长度 5000）。返回 [(seg_p1, seg_p2), ...]，
    每片段端点已平移回 box 内（保方向、保长度）。

    平移规则：对每个子段取中点，按中点所在轴越界方向平移恰好一个 L
    （方向 = -sign(越界)·L）。端点坐标界 |p_a| ≤ 7500，单次平移必落回界内；
    不能合并相邻越界子段整体平移（会在角部案例产生越界片段）。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    length = np.linalg.norm(d)
    boundaries = [0.0] + _crossing_ts(p1, p2) + [1.0]
    frags = []
    for t1, t2 in zip(boundaries[:-1], boundaries[1:]):
        q1 = p1 + t1 * d
        q2 = p1 + t2 * d
        mid = 0.5 * (q1 + q2)
        shift = np.zeros(3)
        for a in range(3):
            if mid[a] > HALF_L:
                shift[a] = -L
            elif mid[a] < -HALF_L:
                shift[a] = L
        f1 = q1 + shift
        f2 = q2 + shift
        # 后置断言：平移后全部回到 box 内（容差）
        assert np.all(f1 >= -HALF_L - BOX_TOL) and np.all(f1 <= HALF_L + BOX_TOL), \
            f"片段端点越界: {f1}"
        assert np.all(f2 >= -HALF_L - BOX_TOL) and np.all(f2 <= HALF_L + BOX_TOL), \
            f"片段端点越界: {f2}"
        frags.append((f1, f2))
    # 长度守恒 sanity
    total = sum(np.linalg.norm(b - a) for a, b in frags)
    assert abs(total - length) < 1e-6, f"片段长度和不守恒: {total} vs {length}"
    return frags


def clip_batch(c, u, h):
    """整批截断。返回 {'p1s': (m,3), 'p2s': (m,3), 'cyl_idx': (m,) int}。

    cyl_idx: 每片段所属的原始圆柱编号（用于同源图论合并）。
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    p1s, p2s, idx = [], [], []
    for i in range(len(c)):
        a = c[i] - h[i] * u[i]
        b = c[i] + h[i] * u[i]
        for f1, f2 in clip_cylinder(a, b):
            p1s.append(f1)
            p2s.append(f2)
            idx.append(i)
    return {
        'p1s': np.asarray(p1s, dtype=float).reshape(-1, 3),
        'p2s': np.asarray(p2s, dtype=float).reshape(-1, 3),
        'cyl_idx': np.asarray(idx, dtype=int),
    }


def electrode_dist(p1s, p2s):
    """片段（短圆柱）到左右带电面 x=±5000 的最短表面距离（向量化）。

    介质A 是平端圆柱（底面半径 r），x 向投影半宽
    e_x = h|u_x| + r√(1-u_x²)（端面圆盘外沿在 x 的凸出；轴平行 x 时
    半径不凸出，e_x = h）。与 Q1 core.electrode_contact 公式一致。
    返回 (dL, dR)，clamp ≥ 0（圆柱穿过电极面即接触，距离 0）。
    """
    axis = p2s - p1s
    half = 0.5 * np.linalg.norm(axis, axis=1)
    ux = axis[:, 0] / np.maximum(2.0 * half, 1e-30)
    cx = 0.5 * (p1s[:, 0] + p2s[:, 0])
    ex = half * np.abs(ux) + R * np.sqrt(np.maximum(1.0 - ux * ux, 0.0))
    dL = np.maximum(cx - ex + HALF_L, 0.0)
    dR = np.maximum(HALF_L - (cx + ex), 0.0)
    return dL, dR


def aabb_candidates(p1s, p2s, thresh=AXIS_THRESH):
    """AABB 预筛：三轴盒间隙均 ≤ thresh 的 (i, j) 对（i < j）。

    lo = min(p1,p2) - R，hi = max(p1,p2) + R 为圆柱包络盒（超集，保守不漏）。
    """
    n = len(p1s)
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    lo = np.minimum(p1s, p2s) - R
    hi = np.maximum(p1s, p2s) + R
    ok = np.ones((n, n), dtype=bool)
    for a in range(3):
        ok &= (lo[:, None, a] - hi[None, :, a] <= thresh)
        ok &= (lo[None, :, a] - hi[:, None, a] <= thresh)
    ok &= np.triu(np.ones((n, n), dtype=bool), 1)
    i, j = np.nonzero(ok)
    return i, j


def segment_distance_batch(a1, b1, a2, b2):
    """向量化 Ericson 5.1.9 线段距离（与 Q1 标量 segment_distance 逐分支一致）。

    镜像 Q1 analyze_group 的批量骨架（去掉周期平移 a2k）。平行退化
    (denom<1e-24) 走 s=0 路径，与标量版候选 min 浮点级等价。
    返回 (m,) 轴线段距离。
    """
    uvec = b1 - a1
    vvec = b2 - a2
    w = a1 - a2
    uu = np.einsum('ij,ij->i', uvec, uvec)
    uv = np.einsum('ij,ij->i', uvec, vvec)
    vv = np.einsum('ij,ij->i', vvec, vvec)
    uw = np.einsum('ij,ij->i', uvec, w)
    vw = np.einsum('ij,ij->i', vvec, w)
    denom = uu * vv - uv * uv
    s_par = np.where(denom < 1e-24, 0.0,
                     np.clip((uv * vw - vv * uw) / np.maximum(denom, 1e-24),
                             0.0, 1.0))
    t_par = (uv * s_par + vw) / np.maximum(vv, 1e-24)
    # t < 0：clamp t=0，s = clamp(-uw/uu)
    m_lo = t_par < 0.0
    s_par = np.where(m_lo, np.clip(-uw / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
    t_par = np.where(m_lo, 0.0, t_par)
    # t > 1：clamp t=1，s = clamp((uv-uw)/uu)
    m_hi = t_par > 1.0
    s_par = np.where(m_hi, np.clip((uv - uw) / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
    t_par = np.where(m_hi, 1.0, t_par)
    p1 = a1 + s_par[:, None] * uvec
    p2 = a2 + t_par[:, None] * vvec
    return np.linalg.norm(p1 - p2, axis=1)


def _fragment_gjk(p1, p2, q1, q2, r=R, tol=1e-10):
    """两片段（短圆柱）的 GJK 精确表面距离。相交为 0。

    单行封装 Q1 core.gjk_distance（与 Q1 判定逐位一致）：
    中心差初始化、相对容差判据、max_iter=64。
    """
    c1 = 0.5 * (p1 + p2)
    c2 = 0.5 * (q1 + q2)
    axis1 = p2 - p1
    axis2 = q2 - q1
    n1 = np.linalg.norm(axis1)
    n2 = np.linalg.norm(axis2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return gjk_distance(c1, axis1 / n1, n1 / 2.0,
                        c2, axis2 / n2, n2 / 2.0, r, tol)


def sample_conductive(c, u, h, delta=DELTA):
    """单次生成的微构体导通判定（一个样本 = 一个 Y）。

    管线：
        1. clip_batch 截断成片段；
        2. 电极边：片段到左右带电面距离 ≤ delta 分别连 S/T，任一侧为空短路；
        3. UnionFind：先按 cyl_idx 合并同源片段（PBC 环面物理连续），再连电极边；
        4. AABB 预筛 + 批量轴距胶囊初筛；
        5. 建边：axis_dist ≤ 2R+delta 的候选对走 GJK 精算（平端圆柱精确
           表面距离 ≤ delta 建边）。注意胶囊距离 max(0, 轴距-2R) 对平端
           圆柱仅是下界（胶囊 ⊃ 平端圆柱），只能作安全拒绝预筛，不能直接
           判连——所有候选必须 GJK 精算（Q4 球-圆柱混合同样依赖 GJK）；
        6. connected(S,T) → conductive。

    返回 dict: {conductive, n_fragments, n_edges, n_left, n_right, n_gjk}。
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n_cyl = len(c)
    if n_cyl == 0:
        return {'conductive': False, 'n_fragments': 0, 'n_edges': 0,
                'n_left': 0, 'n_right': 0, 'n_gjk': 0}

    frag = clip_batch(c, u, h)
    p1s, p2s, cyl_idx = frag['p1s'], frag['p2s'], frag['cyl_idx']
    nf = len(p1s)

    dL, dR = electrode_dist(p1s, p2s)
    left = np.nonzero(dL <= delta)[0]
    right = np.nonzero(dR <= delta)[0]
    if len(left) == 0 or len(right) == 0:
        return {'conductive': False, 'n_fragments': nf, 'n_edges': 0,
                'n_left': int(len(left)), 'n_right': int(len(right)), 'n_gjk': 0}

    uf = UnionFind(nf + 2)
    s_node, t_node = nf, nf + 1

    # 同源片段图论合并（PBC 环面：同一圆柱的片段物理连续）
    order = np.argsort(cyl_idx, kind='stable')
    for a, b in zip(order[:-1], order[1:]):
        if cyl_idx[a] == cyl_idx[b]:
            uf.union(int(a), int(b))

    # 电极边
    for i in left:
        uf.union(s_node, int(i))
    for i in right:
        uf.union(t_node, int(i))

    # 片段对距离：AABB 预筛 → 批量轴距胶囊初筛 → GJK 精算建边
    i_pairs, j_pairs = aabb_candidates(p1s, p2s)
    n_edges = 0
    n_gjk = 0
    if len(i_pairs):
        d = segment_distance_batch(p1s[i_pairs], p2s[i_pairs],
                                   p1s[j_pairs], p2s[j_pairs])
        # 胶囊距离 max(0, 轴距-2R) 是平端圆柱表面距离的下界（安全拒绝）：
        # capsule > delta ⟹ 真实距离 > delta，免 GJK
        cand = d <= AXIS_THRESH
        for m in np.nonzero(cand)[0]:
            i, j = int(i_pairs[m]), int(j_pairs[m])
            dg = _fragment_gjk(p1s[i], p2s[i], p1s[j], p2s[j])
            n_gjk += 1
            if dg <= delta + 1e-6:
                uf.union(i, j)
                n_edges += 1

    conductive = uf.connected(s_node, t_node)
    return {'conductive': bool(conductive), 'n_fragments': nf,
            'n_edges': n_edges, 'n_left': int(len(left)),
            'n_right': int(len(right)), 'n_gjk': n_gjk}


if __name__ == '__main__':
    # 简单自检：单根跨壁圆柱在 PBC 环面上应导通（触左右电极 + 同源合并）
    rng = np.random.default_rng(42)
    c, u, h = generate_cylinders(3, rng)
    print('生成 3 根圆柱: centers =\n', c)
    for i in range(3):
        a = c[i] - h[i] * u[i]
        b = c[i] + h[i] * u[i]
        frags = clip_cylinder(a, b)
        print(f'圆柱 {i}: 片段数 = {len(frags)}, 总长 = '
              f'{sum(np.linalg.norm(f2 - f1) for f1, f2 in frags):.1f}')

    # 单根跨壁圆柱：p1=(-6000,0,0) → p2=(-1000,0,0)
    from core import build_cylinders
    ends = np.array([[[-6000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0]]])
    cc, uu, hh = build_cylinders(ends)
    res = sample_conductive(cc, uu, hh)
    print('单根跨壁圆柱 (PBC 环面):', res)
    assert res['conductive'], '跨壁圆柱应导通（PBC 环面语义）'
    print('自检通过')
