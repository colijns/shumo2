# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31



import os
import sys
from itertools import product

import numpy as np

_Q2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2')
_Q3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q3')
_Q1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q1')
for _p in (_Q2, _Q3, _Q1):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import geometry as geo
from core import UnionFind
from first_passage import crossing_count as crossing_count_cylinders

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    cKDTree = None
    _HAS_SCIPY = False

R_A = geo.R
R_B = 200.0
DELTA = geo.DELTA
L = geo.L
HALF_L = geo.HALF_L

CYL_LEN = 5000.0
V_A_NM3 = np.pi * R_A ** 2 * CYL_LEN
V_B_NM3 = 4.0 / 3.0 * np.pi * R_B ** 3
V_A_UM3 = V_A_NM3 / 1e9
V_B_UM3 = V_B_NM3 / 1e9
C_A_PER_UM3 = 1.05
C_B_PER_UM3 = 0.05
c_A = C_A_PER_UM3 * V_A_UM3
c_B = C_B_PER_UM3 * V_B_UM3

THRESH_AB = R_A + R_B + DELTA
THRESH_BB = 2.0 * R_B + DELTA
TOL_CONTACT = 1e-6

def _bb_pairs_grid(coords, thresh):
    n = len(coords)
    if n < 2 or thresh < 0.0:
        return np.empty((0, 2), dtype=np.int64)

    radius = float(thresh) + TOL_CONTACT
    if radius <= 0.0:

        cell_size = np.finfo(float).eps
    else:
        cell_size = radius
    cell_ids = np.floor(coords / cell_size).astype(np.int64)
    cells = {}
    for index, cell in enumerate(cell_ids):
        cells.setdefault(tuple(cell), []).append(index)

    offsets = tuple(product((-1, 0, 1), repeat=3))
    radius_sq = radius * radius
    pairs = []
    for i, (point, cell) in enumerate(zip(coords, cell_ids)):
        cell_tuple = tuple(cell)
        for offset in offsets:
            neighbour = (cell_tuple[0] + offset[0],
                         cell_tuple[1] + offset[1],
                         cell_tuple[2] + offset[2])
            for j in cells.get(neighbour, ()):
                if j <= i:
                    continue
                difference = point - coords[j]
                if float(np.dot(difference, difference)) <= radius_sq:
                    pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)

    pairs.sort()
    return np.asarray(pairs, dtype=np.int64)


def bb_pairs(coords, thresh=THRESH_BB):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    if _HAS_SCIPY:
        pairs = cKDTree(coords).query_pairs(
            float(thresh) + TOL_CONTACT, output_type='ndarray')
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return _bb_pairs_grid(coords, float(thresh))


def generate_balls(n, rng):
    return rng.uniform(-HALF_L, HALF_L, size=(n, 3))


def clip_balls(cs):
    cs = np.asarray(cs, dtype=float)
    n = len(cs)
    if n == 0:
        return {'c_img': np.empty((0, 3), dtype=float),
                'ball_idx': np.empty(0, dtype=int)}
    imgs = []
    idxs = []
    for index, center in enumerate(cs):
        choices = []
        for coordinate in range(3):
            axis_choices = [0]
            if center[coordinate] - R_B < -HALF_L:
                axis_choices.append(1)
            if center[coordinate] + R_B > HALF_L:
                axis_choices.append(-1)
            choices.append(axis_choices)
        for shift_index in product(*choices):
            imgs.append(center + L * np.asarray(shift_index, dtype=float))
            idxs.append(index)
    return {'c_img': np.asarray(imgs, dtype=float).reshape(-1, 3),
            'ball_idx': np.asarray(idxs, dtype=int)}


def ball_crossing_ratio(cs):
    cs = np.asarray(cs, dtype=float)
    if len(cs) == 0:
        return 0.0
    cross = np.any(np.abs(cs) + R_B > HALF_L, axis=1)
    return float(cross.mean())


def sphere_electrode_dist(cs):
    cs = np.asarray(cs, dtype=float)


    dy = np.maximum(np.abs(cs[:, 1]) - HALF_L, 0.0)
    dz = np.maximum(np.abs(cs[:, 2]) - HALF_L, 0.0)
    center_left = np.sqrt((cs[:, 0] + HALF_L) ** 2 + dy ** 2 + dz ** 2)
    center_right = np.sqrt((cs[:, 0] - HALF_L) ** 2 + dy ** 2 + dz ** 2)
    dL = np.maximum(center_left - R_B, 0.0)
    dR = np.maximum(center_right - R_B, 0.0)
    return dL, dR


def cost(na, nb):
    return c_A * na + c_B * nb


def _segment_distance_params_batch(a1, b1, a2, b2):
    uvec = b1 - a1
    vvec = b2 - a2
    w = a1 - a2
    uu = np.einsum('ij,ij->i', uvec, uvec)
    uv = np.einsum('ij,ij->i', uvec, vvec)
    vv = np.einsum('ij,ij->i', vvec, vvec)
    uw = np.einsum('ij,ij->i', uvec, w)
    vw = np.einsum('ij,ij->i', vvec, w)
    denom = uu * vv - uv * uv
    s = np.where(
        denom < 1e-24,
        0.0,
        np.clip((uv * vw - vv * uw) / np.maximum(denom, 1e-24),
                0.0, 1.0),
    )
    t = (uv * s + vw) / np.maximum(vv, 1e-24)
    low = t < 0.0
    s = np.where(
        low, np.clip(-uw / np.maximum(uu, 1e-24), 0.0, 1.0), s)
    t = np.where(low, 0.0, t)
    high = t > 1.0
    s = np.where(
        high, np.clip((uv - uw) / np.maximum(uu, 1e-24), 0.0, 1.0), s)
    t = np.where(high, 1.0, t)
    p = a1 + s[:, None] * uvec
    q = a2 + t[:, None] * vvec
    return np.linalg.norm(p - q, axis=1), s, t


def _aa_edges_all(p1s, p2s, a_lo, a_hi):
    n = len(p1s)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    ok = np.ones((n, n), dtype=bool)
    for a in range(3):
        ok &= (a_lo[:, None, a] - a_hi[None, :, a] <= geo.AXIS_THRESH)
        ok &= (a_lo[None, :, a] - a_hi[:, None, a] <= geo.AXIS_THRESH)
    ok &= np.triu(np.ones((n, n), dtype=bool), 1)
    ii, jj = np.nonzero(ok)
    if not len(ii):
        return np.empty((0, 2), dtype=np.int64)
    d, s, t = _segment_distance_params_batch(
        p1s[ii], p2s[ii], p1s[jj], p2s[jj])
    candidate = np.nonzero(d <= geo.AXIS_THRESH + TOL_CONTACT)[0]
    out = []
    for m in candidate:
        fi, fj = int(ii[m]), int(jj[m])



        interior = (1e-10 < s[m] < 1.0 - 1e-10
                    and 1e-10 < t[m] < 1.0 - 1e-10)
        if interior or geo._fragment_gjk(
                p1s[fi], p2s[fi], p1s[fj], p2s[fj]) \
                <= DELTA + TOL_CONTACT:
            out.append((fi, fj))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def point_finite_cylinder_distance_batch(p1s, p2s, points, radius=R_A):
    p1s = np.asarray(p1s, dtype=float)
    p2s = np.asarray(p2s, dtype=float)
    points = np.asarray(points, dtype=float)
    axis = p2s - p1s
    lengths = np.linalg.norm(axis, axis=1)
    unit = axis / np.maximum(lengths[:, None], 1e-30)
    center = 0.5 * (p1s + p2s)
    half = 0.5 * lengths
    relative = points - center
    axial_coordinate = np.einsum('ij,ij->i', relative, unit)
    radial_vector = relative - axial_coordinate[:, None] * unit
    radial_distance = np.linalg.norm(radial_vector, axis=1)
    axial_excess = np.maximum(np.abs(axial_coordinate) - half, 0.0)
    radial_excess = np.maximum(radial_distance - float(radius), 0.0)
    return np.hypot(axial_excess, radial_excess)


def _ab_edges_all(p1s, p2s, a_lo, a_hi, c_img, b_lo, b_hi):
    n_a, n_b = len(p1s), len(c_img)
    if n_a == 0 or n_b == 0:
        return np.empty((0, 2), dtype=np.int64)
    ok = np.ones((n_b, n_a), dtype=bool)
    for a in range(3):
        ok &= (b_lo[:, None, a] - a_hi[None, :, a] <= THRESH_AB)
        ok &= (a_lo[None, :, a] - b_hi[:, None, a] <= THRESH_AB)
    bii, fjj = np.nonzero(ok)
    if not len(bii):
        return np.empty((0, 2), dtype=np.int64)
    axis_distance = geo.segment_distance_batch(
        p1s[fjj], p2s[fjj], c_img[bii], c_img[bii])
    candidate = axis_distance <= THRESH_AB + TOL_CONTACT
    candidate_indices = np.nonzero(candidate)[0]
    exact_distance = point_finite_cylinder_distance_batch(
        p1s[fjj[candidate_indices]], p2s[fjj[candidate_indices]],
        c_img[bii[candidate_indices]], radius=R_A)
    m = candidate_indices[
        exact_distance <= R_B + DELTA + TOL_CONTACT]
    return np.stack([fjj[m], bii[m]], axis=1)


def prepare_trial(c, u, h, ball_cs, nb_max):
    frag = geo.clip_batch(c, u, h)
    p1s, p2s, cyl_idx = frag['p1s'], frag['p2s'], frag['cyl_idx']
    dL_A, dR_A = geo.electrode_dist(p1s, p2s)
    a_lo = np.minimum(p1s, p2s) - R_A
    a_hi = np.maximum(p1s, p2s) + R_A
    bf = clip_balls(ball_cs[:nb_max])
    c_img, ball_idx = bf['c_img'], bf['ball_idx']
    dL_B, dR_B = sphere_electrode_dist(c_img)
    b_lo = c_img - R_B
    b_hi = c_img + R_B
    return {
        'p1s': p1s, 'p2s': p2s, 'cyl_idx': cyl_idx,
        'dL_A': dL_A, 'dR_A': dR_A, 'a_lo': a_lo, 'a_hi': a_hi,
        'c_img': c_img, 'ball_idx': ball_idx,
        'dL_B': dL_B, 'dR_B': dR_B, 'b_lo': b_lo, 'b_hi': b_hi,
        'aa_edges': _aa_edges_all(p1s, p2s, a_lo, a_hi),
        'ab_edges': _ab_edges_all(p1s, p2s, a_lo, a_hi, c_img, b_lo, b_hi),
        'bb_edges': bb_pairs(c_img, THRESH_BB),
    }


def sample_prefix(pr, n_a, n_b, delta=DELTA):
    if n_a + n_b == 0:
        return False
    iA = np.nonzero(pr['cyl_idx'] < n_a)[0]
    iB = np.nonzero(pr['ball_idx'] < n_b)[0]
    nA, nB = len(iA), len(iB)
    if nA == 0 and nB == 0:
        return False

    left_a = np.nonzero(pr['dL_A'][iA] <= delta)[0]
    right_a = np.nonzero(pr['dR_A'][iA] <= delta)[0]
    left_b = np.nonzero(pr['dL_B'][iB] <= delta)[0]
    right_b = np.nonzero(pr['dR_B'][iB] <= delta)[0]
    if len(left_a) + len(left_b) == 0 or len(right_a) + len(right_b) == 0:
        return False

    uf = UnionFind(nA + nB + 2)
    s_node, t_node = nA + nB, nA + nB + 1
    for i in left_a:
        uf.union(s_node, int(i))
    for i in right_a:
        uf.union(t_node, int(i))
    for j in left_b:
        uf.union(s_node, nA + int(j))
    for j in right_b:
        uf.union(t_node, nA + int(j))
    if uf.connected(s_node, t_node):
        return True





    if nA >= 2 and len(pr['aa_edges']):
        ei = pr['aa_edges'][:, 0]
        ej = pr['aa_edges'][:, 1]
        keep = (pr['cyl_idx'][ei] < n_a) & (pr['cyl_idx'][ej] < n_a)
        li = np.searchsorted(iA, ei[keep])
        lj = np.searchsorted(iA, ej[keep])
        for m in range(len(li)):
            uf.union(int(li[m]), int(lj[m]))
        if uf.connected(s_node, t_node):
            return True


    if nA and nB and len(pr['ab_edges']):
        ef = pr['ab_edges'][:, 0]
        eb = pr['ab_edges'][:, 1]
        keep = ((pr['cyl_idx'][ef] < n_a) & (pr['ball_idx'][eb] < n_b))
        lf = np.searchsorted(iA, ef[keep])
        lb = np.searchsorted(iB, eb[keep])
        for m in range(len(lf)):
            uf.union(nA + int(lb[m]), int(lf[m]))
        if uf.connected(s_node, t_node):
            return True


    if nB >= 2 and len(pr['bb_edges']):
        ea = pr['bb_edges'][:, 0]
        eb = pr['bb_edges'][:, 1]
        keep = (pr['ball_idx'][ea] < n_b) & (pr['ball_idx'][eb] < n_b)
        la = np.searchsorted(iB, ea[keep])
        lb = np.searchsorted(iB, eb[keep])
        for m in range(len(la)):
            uf.union(nA + int(la[m]), nA + int(lb[m]))
    return uf.connected(s_node, t_node)


def sample_mixed(c, u, h, ball_cs, delta=DELTA):
    pr = prepare_trial(c, u, h, ball_cs, len(ball_cs))
    return {'conductive': bool(sample_prefix(pr, len(c), len(ball_cs),
                                             delta=delta)),
            'n_cylinders': int(len(c)), 'n_balls': int(len(ball_cs))}


def first_contact_nb(ball_cs, delta=DELTA):
    n_b = len(ball_cs)
    if n_b == 0:
        return None
    bf = clip_balls(ball_cs)
    c_img, ball_idx = bf['c_img'], bf['ball_idx']
    dL, dR = sphere_electrode_dist(c_img)
    nf = len(c_img)
    uf = UnionFind(nf + 2)
    s_node, t_node = nf, nf + 1

    pairs = bb_pairs(c_img, THRESH_BB)
    by_ball = [[] for _ in range(n_b)]
    for a, b in pairs:
        by_ball[int(max(ball_idx[int(a)], ball_idx[int(b)]))].append((int(a), int(b)))
    groups = [np.nonzero(ball_idx == i)[0] for i in range(n_b)]

    for i in range(n_b):
        for f in groups[i]:
            f = int(f)
            if dL[f] <= delta:
                uf.union(s_node, f)
            if dR[f] <= delta:
                uf.union(t_node, f)
        for a, b in by_ball[i]:
            uf.union(a, b)
        if uf.connected(s_node, t_node):
            return i + 1
    return None


if __name__ == '__main__':

    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    c, u, h = geo.generate_cylinders(50, rng)
    ball_cs = generate_balls(80, rng)
    print(f'单件成本: c_A = {c_A:.6e} 元, c_B = {c_B:.6e} 元')
    print(f'(50, 80) 混合导通: {sample_mixed(c, u, h, ball_cs)}')
    nc = first_contact_nb(generate_balls(200, np.random.default_rng(7)))
    print(f'纯B 首通(200球): {nc}')
    print(f'耗时 {time.perf_counter() - t0:.2f}s')
