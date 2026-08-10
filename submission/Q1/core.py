# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import os

import numpy as np

R = 30.0

DELTA = float(os.environ.get("SHUMO_DELTA", "1.8"))
L = 10000.0
HALF_L = 5000.0

def support_cylinder(v, c, u, h, r=R):
    w = v - (v @ u) * u
    nw = np.linalg.norm(w)
    dot = v @ u
    if abs(dot) < 1e-14:
        t_coeff = 0.0
    else:
        t_coeff = h * (1.0 if dot > 0.0 else -1.0)
    if nw > 1e-12:
        return c + t_coeff * u + r * w / nw
    return c + t_coeff * u

def _support_any(v, c, u, h, r):
    nv = np.linalg.norm(v)
    if h <= 0.0:
        if nv < 1e-14:
            return c
        return c + r * v / nv
    return support_cylinder(v, c, u, h, r)

class GJKError(RuntimeError):
    pass

def _closest_point_to_origin(pts):
    best_p = None
    best_idx = None
    best_norm2 = np.inf
    n = len(pts)
    tol2 = 1e-20

    for i in range(n):
        p = pts[i]
        n2 = p @ p
        if n2 < best_norm2:
            best_norm2, best_p, best_idx = n2, p, [i]

    for i in range(n):
        for j in range(i + 1, n):
            a, b = pts[i], pts[j]
            d = b - a
            denom = d @ d
            if denom < 1e-24:
                continue
            t = min(max(-(a @ d) / denom, 0.0), 1.0)
            p = a + t * d
            n2 = p @ p
            if n2 < best_norm2:
                best_norm2, best_p, best_idx = n2, p, [i, j]

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                e1, e2 = b - a, c - a
                nrm = np.cross(e1, e2)
                denom = nrm @ nrm
                if denom < 1e-24:
                    continue
                t = (nrm @ a) / denom
                p = t * nrm

                det = e1 @ e1 * (e2 @ e2) - (e1 @ e2) ** 2
                if det < 1e-24:
                    continue
                w = p - a
                lam1 = ((w @ e1) * (e2 @ e2) - (w @ e2) * (e1 @ e2)) / det
                lam2 = ((w @ e2) * (e1 @ e1) - (w @ e1) * (e1 @ e2)) / det
                lam0 = 1.0 - lam1 - lam2
                if lam0 >= -1e-12 and lam1 >= -1e-12 and lam2 >= -1e-12:
                    n2 = p @ p
                    if n2 < best_norm2:
                        best_norm2, best_p, best_idx = n2, p, [i, j, k]

    if n == 4 and best_norm2 > tol2:
        p0, p1, p2, p3 = pts

        M = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
        try:
            b = np.linalg.solve(M, -p0)
        except np.linalg.LinAlgError:
            b = None
        if b is not None:
            lam = np.concatenate([[1.0 - b.sum()], b])
            if np.all(lam >= -1e-12):
                return np.zeros(3), [0, 1, 2, 3]

    return best_p, best_idx

def gjk_distance(c1, u1, h1, c2, u2, h2, r=R, tol=1e-10, max_iter=64):
    def support(v):
        return _support_any(v, c1, u1, h1, r) - _support_any(-v, c2, u2, h2, r)

    v = c2 - c1
    if np.linalg.norm(v) < 1e-12:
        v = np.array([1.0, 0.0, 0.0])
    W = []
    for _ in range(max_iter):
        wm = support(-v)
        if v @ wm >= v @ v - tol * max(1.0, v @ v):
            return np.linalg.norm(v)
        wp = support(v)
        if wp @ v < v @ v - tol * max(1.0, v @ v):
            w = wp
        else:
            w = wm
        W.append(w)
        v_new, keep = _closest_point_to_origin(W)
        W = [W[i] for i in keep]
        if np.linalg.norm(v_new) <= tol:
            return 0.0
        if np.linalg.norm(v_new - v) <= tol * 1e-3:

            return 0.0
        v = v_new
    raise GJKError(f"GJK 未收敛: it={max_iter}, dist={np.linalg.norm(v)}")

def segment_distance(a1, b1, a2, b2):
    u = b1 - a1
    v = b2 - a2
    w = a1 - a2
    a = u @ u
    b = u @ v
    c = v @ v
    d = u @ w
    e = v @ w
    denom = a * c - b * b
    if denom < 1e-24:
        t = min(max(e / c if c > 1e-24 else 0.0, 0.0), 1.0)
        s2 = min(max(-d / a if a > 1e-24 else 0.0, 0.0), 1.0)
        if np.linalg.norm(a1 - (a2 + t * v)) > np.linalg.norm((a1 + s2 * u) - a2):
            s, t = s2, 0.0
        else:
            s = 0.0
    else:
        s = min(max((b * e - c * d) / denom, 0.0), 1.0)
        t = (b * s + e) / c
        if t < 0.0:
            t = 0.0
            s = min(max(-d / a if a > 1e-24 else 0.0, 0.0), 1.0)
        elif t > 1.0:
            t = 1.0
            s = min(max((b - d) / a if a > 1e-24 else 0.0, 0.0), 1.0)
    p1 = a1 + s * u
    p2 = a2 + t * v
    return np.linalg.norm(p1 - p2)

def capsule_lower_bound(s, r=R):
    return max(s - 2 * r, 0.0)

def cylinder_aabb(c, u, h, r=R):
    e = h[:, None] * np.abs(u) + r * np.sqrt(np.maximum(1.0 - u * u, 0.0))
    return c - e, c + e

def pbc_candidate_k(dc, side_len=L):
    k0 = np.clip(np.round(dc / side_len), -1, 1).astype(int)
    cands = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = (k0[0] + dx, k0[1] + dy, k0[2] + dz)
                if all(-1 <= v <= 1 for v in k):
                    cands.add(k)
    return np.array(sorted(cands), dtype=int)

def aabb_pbc_overlap(lo1, hi1, lo2, hi2, k, side_len=L, delta=DELTA):
    k = np.asarray(k, dtype=float)
    shifted_lo2 = lo2 + side_len * k
    shifted_hi2 = hi2 + side_len * k
    return bool(np.all((lo1 - shifted_hi2 <= delta) & (shifted_lo2 - hi1 <= delta)))

def electrode_contact(c, u, h, r=R, delta=DELTA, side='L', wrap_crossings=False):
    e_x = h * abs(u[0]) + r * np.sqrt(max(1.0 - u[0] * u[0], 0.0))
    x_min = c[0] - e_x
    x_max = c[0] + e_x
    if side == 'L':
        if x_min <= -HALF_L + delta:
            return True
        if wrap_crossings:

            raise NotImplementedError("wrap_crossings=True 仅限后续随机仿真使用")
        return False
    else:
        if x_max >= HALF_L - delta:
            return True
        if wrap_crossings:
            raise NotImplementedError("wrap_crossings=True 仅限后续随机仿真使用")
        return False

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def component_sizes(self):
        sizes = {}
        for x in range(len(self.parent)):
            r = self.find(x)
            sizes[r] = sizes.get(r, 0) + 1
        return sizes

def bfs_witness_path(n_nodes, adj, left, right):
    s, t = n_nodes, n_nodes + 1
    graph = {i: set(adj.get(i, ())) for i in range(n_nodes)}
    graph[s] = set(left)
    graph[t] = set(right)
    for i in left:
        graph.setdefault(i, set()).add(s)
    for i in right:
        graph.setdefault(i, set()).add(t)

    prev = {s: None}
    queue = [s]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == t:
            break
        for nxt in graph.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return ['S' if v == s else ('T' if v == t else f'A{v + 1}') for v in path]

def analyze_group(c, u, h, r=R, delta=DELTA, side_len=L, pbc=False,
                  wrap_crossings=False):
    n = len(c)
    u = np.asarray(u, dtype=float)
    c = np.asarray(c, dtype=float)
    h = np.asarray(h, dtype=float)

    left, right = [], []
    for i in range(n):
        if electrode_contact(c[i], u[i], h[i], r, delta, 'L', wrap_crossings):
            left.append(i)
        if electrode_contact(c[i], u[i], h[i], r, delta, 'R', wrap_crossings):
            right.append(i)
    left_set = set(left)
    right_set = set(right)

    lo, hi = cylinder_aabb(c, u, h, r)

    dc = c[:, None, :] - c[None, :, :]
    k0 = np.clip(np.round(dc / side_len), -1, 1).astype(int)

    if pbc:
        offsets = np.array([(dx, dy, dz)
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)], dtype=int)
    else:
        offsets = np.zeros((1, 3), dtype=int)

    pair_mask = np.zeros((n, n), dtype=bool)
    for off in offsets:
        if pbc:
            k = np.clip(k0 + off[None, None, :], -1, 1)
        else:
            k = np.zeros((n, n, 3), dtype=int)
        ok = np.ones((n, n), dtype=bool)
        for a in range(3):
            ka = k[:, :, a].astype(float)
            shift = side_len * ka
            ok &= (lo[:, None, a] - (hi[None, :, a] + shift) <= delta)
            ok &= ((lo[None, :, a] + shift) - hi[:, None, a] <= delta)
        pair_mask |= ok
    pair_mask &= np.triu(np.ones((n, n), dtype=bool), 1)

    i_pairs, j_pairs = np.nonzero(pair_mask)
    aabb_candidate_pairs = len(i_pairs)

    triples = []
    for i, j in zip(i_pairs.tolist(), j_pairs.tolist()):
        kcands = []
        if pbc:
            kbase = k0[i, j]
            for off in offsets:
                k = np.clip(kbase + off, -1, 1)
                if aabb_pbc_overlap(lo[i], hi[i], lo[j], hi[j], k,
                                    side_len, delta):
                    kcands.append(tuple(k.tolist()))
        else:

            k = np.zeros(3, dtype=int)
            if aabb_pbc_overlap(lo[i], hi[i], lo[j], hi[j], k,
                                side_len, delta):
                kcands.append((0, 0, 0))
        for k in kcands:
            triples.append((int(i), int(j), k))

    if triples:
        triples_arr = np.array(triples, dtype=object)
        i_arr = np.array([t[0] for t in triples], dtype=int)
        j_arr = np.array([t[1] for t in triples], dtype=int)
        k_arr = np.array([t[2] for t in triples], dtype=float)

        a1 = c[i_arr] - h[i_arr][:, None] * u[i_arr]
        b1 = c[i_arr] + h[i_arr][:, None] * u[i_arr]
        a2 = c[j_arr] - h[j_arr][:, None] * u[j_arr]
        b2 = c[j_arr] + h[j_arr][:, None] * u[j_arr]
        a2k = a2 + side_len * k_arr

        uvec = b1 - a1
        vvec = b2 - a2
        w = a1 - a2k

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
        p2 = a2k + t_par[:, None] * vvec
        axis_dist = np.linalg.norm(p1 - p2, axis=1)

        keep = axis_dist <= 2.0 * r + delta
        gjk_triples = [(i_arr[m], j_arr[m], k_arr[m]) for m in np.nonzero(keep)[0]]
    else:
        gjk_triples = []

    uf = UnionFind(n + 2)
    s_node, t_node = n, n + 1
    for i in left:
        uf.union(s_node, i)
    for i in right:
        uf.union(t_node, i)

    contact_edges = []
    best = {}
    for i, j, k in gjk_triples:
        c2k = c[j] + side_len * k
        d = gjk_distance(c[i], u[i], h[i], c2k, u[j], h[j], r)
        key = (i, j)
        if key not in best or d < best[key][0]:
            best[key] = (d, tuple(int(v) for v in k))

    for (i, j), (d, k) in best.items():
        if d <= delta:
            contact_edges.append((i, j, k, d))
            uf.union(i, j)

    conductive = uf.connected(s_node, t_node)

    comp_sizes = uf.component_sizes()
    if comp_sizes:
        root = max(comp_sizes, key=comp_sizes.get)
        media_in_largest = sum(1 for x in range(n) if uf.find(x) == root)
        max_component_size = media_in_largest
    else:
        max_component_size = 0

    adj = {}
    for i, j, k, d in contact_edges:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    witness_path = bfs_witness_path(n, adj, left, right)

    return {
        'n_cylinders': n,
        'left_contacts': sorted(left_set),
        'right_contacts': sorted(right_set),
        'n_left_contact': len(left),
        'n_right_contact': len(right),
        'aabb_candidate_pairs': aabb_candidate_pairs,
        'contact_edges': [(i, j, k, float(d)) for i, j, k, d in contact_edges],
        'conductive': bool(conductive),
        'max_component_size': max_component_size,
        'witness_path': witness_path,
    }

def build_cylinders(ends, r=R):
    ends = np.asarray(ends, dtype=float)
    p1, p2 = ends[:, 0, :], ends[:, 1, :]
    axis = p2 - p1
    lengths = np.linalg.norm(axis, axis=1)
    if np.any(lengths < 1e-12):
        bad = int(np.nonzero(lengths < 1e-12)[0][0])
        raise ValueError(f"介质 {bad + 1} 的两个轴线端点重合")
    u = axis / lengths[:, None]
    h = lengths / 2.0
    c = (p1 + p2) / 2.0
    return c, u, h

def load_cylinders(xlsx_path, sheet_index, r=R):
    import pandas as pd

    raw = pd.read_excel(xlsx_path, sheet_name=sheet_index,
                        header=None, skiprows=2, dtype=float)
    if raw.shape[1] < 6:
        raise ValueError(f"工作表 {sheet_index} 列数不足 6")
    values = raw.iloc[:, :6].values
    n = values.shape[0]
    for i in range(n):
        row = values[i]
        if np.all(np.isnan(row)):
            continue
        if np.any(np.isnan(row)):
            bad = int(np.nonzero(np.isnan(row))[0][0]) + 1
            raise ValueError(f"工作表 {sheet_index} 第 {i + 3} 行只填写了部分坐标"
                             f"（第 {bad} 列为空）")
        if np.any(np.abs(row) > HALF_L + 1e-9):
            raise ValueError(f"工作表 {sheet_index} 第 {i + 3} 行坐标超出 [-5000,5000]")
    ends = values[~np.isnan(values).any(axis=1)].reshape(-1, 2, 3)
    c, u, h = build_cylinders(ends, r)
    node_names = [f'A{j + 1}' for j in range(len(c))]
    return c, u, h, node_names
