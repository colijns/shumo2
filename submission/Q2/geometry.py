# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31



import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q1'))
from core import R, DELTA, L, HALF_L, UnionFind, segment_distance, gjk_distance

CYL_LEN = 5000.0
CYL_HALF_LEN = CYL_LEN / 2.0
V_A = np.pi * R ** 2 * CYL_LEN
V_BOX = L ** 3
AXIS_THRESH = 2.0 * R + DELTA
TOL = 1e-9
BOX_TOL = 1e-6


def generate_cylinders(n, rng):
    c = rng.uniform(-HALF_L, HALF_L, size=(n, 3))
    z = rng.uniform(-1.0, 1.0, size=(n,))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n,))
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    u = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)
    h = np.full(n, CYL_HALF_LEN, dtype=float)
    return c, u, h


def cylinder_projection_halfwidth(u, h, r=R):
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    return (h[..., None] * np.abs(u)
            + float(r) * np.sqrt(np.maximum(1.0 - u * u, 0.0)))


def axis_crossing_flags(c, u, h, half_l=HALF_L):
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.any(np.abs(c) + h[..., None] * np.abs(u) > half_l, axis=1)


def solid_crossing_flags(c, u, h, r=R, half_l=HALF_L):
    c = np.asarray(c, dtype=float)
    extent = cylinder_projection_halfwidth(u, h, r)
    return np.any(np.abs(c) + extent > half_l, axis=1)


def _crossing_ts(p1, p2):
    d = p2 - p1
    ts = []
    for a in range(3):
        if abs(d[a]) < 1e-12:
            continue
        for plane in (-HALF_L, HALF_L):
            if (p1[a] - plane) * (p2[a] - plane) < 0.0:
                t = (plane - p1[a]) / d[a]
                if TOL < t < 1.0 - TOL:
                    ts.append(t)
    ts.sort()

    out = []
    for t in ts:
        if not out or t - out[-1] > 1e-9:
            out.append(t)
    return out


def clip_cylinder(p1, p2):
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

        assert np.all(f1 >= -HALF_L - BOX_TOL) and np.all(f1 <= HALF_L + BOX_TOL), \
            f"片段端点越界: {f1}"
        assert np.all(f2 >= -HALF_L - BOX_TOL) and np.all(f2 <= HALF_L + BOX_TOL), \
            f"片段端点越界: {f2}"
        frags.append((f1, f2))

    total = sum(np.linalg.norm(b - a) for a, b in frags)
    assert abs(total - length) < 1e-6, f"片段长度和不守恒: {total} vs {length}"
    return frags


def clip_batch(c, u, h):
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
    axis = p2s - p1s
    half = 0.5 * np.linalg.norm(axis, axis=1)
    ux = axis[:, 0] / np.maximum(2.0 * half, 1e-30)
    cx = 0.5 * (p1s[:, 0] + p2s[:, 0])
    ex = half * np.abs(ux) + R * np.sqrt(np.maximum(1.0 - ux * ux, 0.0))
    dL = np.maximum(cx - ex + HALF_L, 0.0)
    dR = np.maximum(HALF_L - (cx + ex), 0.0)
    return dL, dR


def aabb_candidates(p1s, p2s, thresh=AXIS_THRESH):
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

    m_lo = t_par < 0.0
    s_par = np.where(m_lo, np.clip(-uw / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
    t_par = np.where(m_lo, 0.0, t_par)

    m_hi = t_par > 1.0
    s_par = np.where(m_hi, np.clip((uv - uw) / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
    t_par = np.where(m_hi, 1.0, t_par)
    p1 = a1 + s_par[:, None] * uvec
    p2 = a2 + t_par[:, None] * vvec
    return np.linalg.norm(p1 - p2, axis=1)


def _fragment_gjk(p1, p2, q1, q2, r=R, tol=1e-10):
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
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n_cyl = len(c)
    if n_cyl == 0:
        return {'conductive': False, 'n_fragments': 0, 'n_crossing': 0,
                'n_edges': 0,
                'n_left': 0, 'n_right': 0, 'n_gjk': 0}

    frag = clip_batch(c, u, h)
    p1s, p2s, cyl_idx = frag['p1s'], frag['p2s'], frag['cyl_idx']
    nf = len(p1s)
    fragment_counts = np.bincount(cyl_idx, minlength=n_cyl)
    n_crossing = int(np.count_nonzero(fragment_counts > 1))

    dL, dR = electrode_dist(p1s, p2s)
    left = np.nonzero(dL <= delta)[0]
    right = np.nonzero(dR <= delta)[0]
    if len(left) == 0 or len(right) == 0:
        return {'conductive': False, 'n_fragments': nf,
                'n_crossing': n_crossing, 'n_edges': 0,
                'n_left': int(len(left)), 'n_right': int(len(right)), 'n_gjk': 0}

    uf = UnionFind(nf + 2)
    s_node, t_node = nf, nf + 1


    for i in left:
        uf.union(s_node, int(i))
    for i in right:
        uf.union(t_node, int(i))


    i_pairs, j_pairs = aabb_candidates(p1s, p2s)
    n_edges = 0
    n_gjk = 0
    if len(i_pairs):
        d = segment_distance_batch(p1s[i_pairs], p2s[i_pairs],
                                   p1s[j_pairs], p2s[j_pairs])


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
            'n_crossing': n_crossing, 'n_edges': n_edges,
            'n_left': int(len(left)),
            'n_right': int(len(right)), 'n_gjk': n_gjk}


if __name__ == '__main__':

    rng = np.random.default_rng(42)
    c, u, h = generate_cylinders(3, rng)
    print('生成 3 根圆柱: centers =\n', c)
    for i in range(3):
        a = c[i] - h[i] * u[i]
        b = c[i] + h[i] * u[i]
        frags = clip_cylinder(a, b)
        print(f'圆柱 {i}: 片段数 = {len(frags)}, 总长 = '
              f'{sum(np.linalg.norm(f2 - f1) for f1, f2 in frags):.1f}')


    from core import build_cylinders
    ends = np.array([[[-6000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0]]])
    cc, uu, hh = build_cylinders(ends)
    res = sample_conductive(cc, uu, hh)
    print('单根跨壁圆柱（片段独立）:', res)
    assert not res['conductive'], '同源片段不应无条件短接左右电极'
    print('自检通过')
