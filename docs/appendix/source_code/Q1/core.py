"""问题1：三组微构体的导电判定模型 —— 几何与图论内核。

规格：docs/问题1.md（唯一权威）。
模型：有限平端圆柱（r=30 nm）填充边长 10000 nm 立方体，周期边界 27 镜像，
      接触判据 d <= 1.8 nm，并查集判导电 + BFS 见证路径。

本模块自包含，不 import 其他算法模块。纯 numpy，无文件 IO。
"""

import numpy as np

# ---------------- 常量 ----------------
R = 30.0          # 介质 A 半径 (nm)
DELTA = 1.8       # 电接触临界距离 (nm)
L = 10000.0       # 立方体边长 (nm)
HALF_L = 5000.0   # 立方体半边长 (nm)

# ---------------- 圆柱支撑与 GJK 距离 ----------------

def support_cylinder(v, c, u, h, r=R):
    """有限平端圆柱 C = {c + t u + q : |t|<=h, q⊥u, ||q||<=r} 沿方向 v 的支撑点。

    v 与 u 平行时法向分量 w 为零，支撑集为整个端面圆盘，
    取轴端点即可（圆盘上 v·q = 0，函数值不变）。
    |v·u| 极小时取 t=0（轴中点），避免把 simplex 点推到远离原点的轴端，
    保证相交检测能捕获原点。
    """
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
    """凸体支撑点：h<=0 视为球（半径 r，球心 c），否则为有限平端圆柱。

    球不是圆柱的特例（h=0 的圆柱退化为圆盘），故需单独分支。
    测试中的"球-球/圆柱-球"解析对照（规格第 14 节第 1、3 条）走球分支；
    附件数据全部 h > 0，走圆柱分支。
    """
    nv = np.linalg.norm(v)
    if h <= 0.0:
        if nv < 1e-14:
            return c  # 退化：方向为零，支撑点任意（取球心）
        return c + r * v / nv
    return support_cylinder(v, c, u, h, r)


class GJKError(RuntimeError):
    """GJK 迭代不收敛。"""


def _closest_point_to_origin(pts):
    """Johnson 子算法：求凸包 conv(pts) 中离原点最近的点。

    pts 为 1..4 个 (3,) 点。枚举全部非空子集（1 点 / 2 点线段 /
    3 点三角形 / 4 点四面体），取各面最近点的最小值。
    退化（共线、共面、重复点）由子集枚举天然覆盖：
    高维子集的最近点若投影落在面外，其最近点由低维子集给出。

    返回 (p, keep_idx)：最近点 p 与构成它的点索引列表。
    """
    best_p = None
    best_idx = None
    best_norm2 = np.inf
    n = len(pts)
    tol2 = 1e-20  # 原点入单纯形的判定阈值（平方），对应 1e-10 绝对量级

    # 1 点子集
    for i in range(n):
        p = pts[i]
        n2 = p @ p
        if n2 < best_norm2:
            best_norm2, best_p, best_idx = n2, p, [i]

    # 2 点子集（线段，clamp 参数化）
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pts[i], pts[j]
            d = b - a
            denom = d @ d
            if denom < 1e-24:  # 重复点退化，单点已覆盖
                continue
            t = min(max(-(a @ d) / denom, 0.0), 1.0)
            p = a + t * d
            n2 = p @ p
            if n2 < best_norm2:
                best_norm2, best_p, best_idx = n2, p, [i, j]

    # 3 点子集（三角形：叉积法求原点投影，重心坐标判定是否在面内）
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                e1, e2 = b - a, c - a
                nrm = np.cross(e1, e2)
                denom = nrm @ nrm
                if denom < 1e-24:  # 共线退化，线段已覆盖
                    continue
                t = (nrm @ a) / denom
                p = t * nrm  # 原点到平面的投影点
                # 重心坐标 lambda0 + lambda1 e1 + lambda2 e2 = p - a
                # 用行列式（平面内 2D 求解）
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

    # 4 点子集（四面体：原点在内部则距离为 0）
    if n == 4 and best_norm2 > tol2:
        p0, p1, p2, p3 = pts
        # 原点重心坐标：M = [p1-p0, p2-p0, p3-p0]，b = M⁻¹(-p0)，
        # lambda = [1 - Σb, b]。全部非负（容差内）则原点在四面体内。
        M = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
        try:
            b = np.linalg.solve(M, -p0)
        except np.linalg.LinAlgError:
            b = None  # 奇异：四面体退化为低维，低维子集已覆盖
        if b is not None:
            lam = np.concatenate([[1.0 - b.sum()], b])
            if np.all(lam >= -1e-12):
                return np.zeros(3), [0, 1, 2, 3]

    return best_p, best_idx


def gjk_distance(c1, u1, h1, c2, u2, h2, r=R, tol=1e-10, max_iter=64):
    """两个凸体（有限平端圆柱或球）之间的最短表面距离（GJK）。

    相交或重叠时返回 0.0。不收敛抛 GJKError。

    算法：凸集 D = C1 ⊖ C2 上离原点最近的点 v，用"支撑点迭代"。
    投影定理：v 是投影 ⇔ 对一切 x∈D 有 x·v >= |v|²，
    等价于 min_{x∈D} x·v = v·S_D(-v) >= |v|²（min 方向用 S_D(-v) 求）。
    注意不能用 S_D(v) 判据：v∈D 时 S_D(v)·v >= |v|² 恒成立，
    会把"v 在 D 内"误判为"v 是投影"（相交时得到非零距离的错误结果）。

    迭代：判据不满足时，优先加入 +v 方向支撑点；若该点已是 simplex
    顶点（v 是 +v 方向支撑顶点，无新信息），改用 -v 方向支撑点，
    保证 simplex 能跨过原点（相交例：两端点线段过原点 → 距离 0）。
    """
    def support(v):
        return _support_any(v, c1, u1, h1, r) - _support_any(-v, c2, u2, h2, r)

    v = c2 - c1
    if np.linalg.norm(v) < 1e-12:
        v = np.array([1.0, 0.0, 0.0])  # 同心退化
    W = []
    for _ in range(max_iter):
        wm = support(-v)  # min 方向支撑：判据用
        if v @ wm >= v @ v - tol * max(1.0, v @ v):
            return np.linalg.norm(v)  # 投影条件满足
        wp = support(v)
        if wp @ v < v @ v - tol * max(1.0, v @ v):
            w = wp  # +v 方向有更远点，能收紧 simplex
        else:
            w = wm  # v 已是 +v 方向支撑顶点 → 用 -v 方向点跨过原点
        W.append(w)
        v_new, keep = _closest_point_to_origin(W)
        W = [W[i] for i in keep]
        if np.linalg.norm(v_new) <= tol:
            return 0.0  # 原点进入单纯形：相交或接触
        if np.linalg.norm(v_new - v) <= tol * 1e-3:
            # 数值停滞：simplex 不再扩展。判据仍未满足说明 v∈D 且
            # 0 更接近原点（相交/重叠），保守返回 0。
            return 0.0
        v = v_new
    raise GJKError(f"GJK 未收敛: it={max_iter}, dist={np.linalg.norm(v)}")


# ---------------- 距离加速：线段、胶囊下界、AABB ----------------

def segment_distance(a1, b1, a2, b2):
    """两条线段 [a1,b1]、[a2,b2] 的最短距离（Ericson 5.1.9）。"""
    u = b1 - a1
    v = b2 - a2
    w = a1 - a2
    a = u @ u
    b = u @ v
    c = v @ v
    d = u @ w
    e = v @ w
    denom = a * c - b * b
    if denom < 1e-24:  # 平行退化：Ericson 两侧候选（s=0 与 t=0）取距离小者
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
    """胶囊距离下界：两条轴线距离 s 时，对应胶囊体距离 max(s - 2r, 0)。"""
    return max(s - 2 * r, 0.0)


def cylinder_aabb(c, u, h, r=R):
    """圆柱的轴对齐包围盒。各轴投影半宽 e_a = h|u_a| + r sqrt(1-u_a²)。"""
    e = h[:, None] * np.abs(u) + r * np.sqrt(np.maximum(1.0 - u * u, 0.0))
    return c - e, c + e


def pbc_candidate_k(dc, side_len=L):
    """根据中心差 dc 给出周期平移候选集（≤27 个）。

    对每轴取 round(dc_a/L)（clamp 到 [-1,1]）后，在其 {−1,0,1}³ 邻域内
    枚举并与 {-1,0,1}³ 求交。真接触对应的平移必落在该邻域中（不漏保证）。
    """
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
    """周期 AABB 重叠判定：三轴间隙均 <= delta 才返回 True。

    判据（逐轴）：lo1_a - (hi2_a + L k_a) <= delta 且
                  (lo2_a + L k_a) - hi1_a <= delta。
    AABB 是圆柱包络，盒距 <= 圆柱距，因此粗筛保守不漏真接触。
    """
    k = np.asarray(k, dtype=float)
    shifted_lo2 = lo2 + side_len * k
    shifted_hi2 = hi2 + side_len * k
    return bool(np.all((lo1 - shifted_hi2 <= delta) & (shifted_lo2 - hi1 <= delta)))


# ---------------- 电极接触 ----------------

def electrode_contact(c, u, h, r=R, delta=DELTA, side='L', wrap_crossings=False):
    """圆柱是否与带电面接触。

    圆柱 x 投影半宽 e_x = h|u_x| + r sqrt(1-u_x²)，
    x_min = c_x - e_x, x_max = c_x + e_x。
    side='L'：x_min <= -5000 + delta；side='R'：x_max >= 5000 - delta。
    电极接触不做周期处理（镜像只用于介质间距离）。

    wrap_crossings=True 仅供后续随机仿真（生成完整越界圆柱时，一条
    记录可能同时接到左右电极）使用；问题1 恒为 False，该分支不触发。
    """
    e_x = h * abs(u[0]) + r * np.sqrt(max(1.0 - u[0] * u[0], 0.0))
    x_min = c[0] - e_x
    x_max = c[0] + e_x
    if side == 'L':
        if x_min <= -HALF_L + delta:
            return True
        if wrap_crossings:
            # 仿真模式：标称长度指示该行是越界截断记录时，允许补接到左面
            raise NotImplementedError("wrap_crossings=True 仅限后续随机仿真使用")
        return False
    else:
        if x_max >= HALF_L - delta:
            return True
        if wrap_crossings:
            raise NotImplementedError("wrap_crossings=True 仅限后续随机仿真使用")
        return False


# ---------------- 图连通 ----------------

class UnionFind:
    """并查集（路径压缩 + 按秩合并）。"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 路径压缩
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
    """BFS 搜索从左侧电极（虚拟节点 S）到右侧电极（T）的一条路径。

    adj: dict[node -> set[node]]，介质节点 0..n_nodes-1。
    left/right: 与左右电极接触的介质编号列表。
    返回含 'S'/'T' 的节点名列表；无路径返回 None。
    """
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


# ---------------- 分组级主流程 ----------------

def analyze_group(c, u, h, r=R, delta=DELTA, side_len=L, pbc=True,
                  wrap_crossings=False):
    """计算一个分组（一组微构体）的完整导电判定。

    参数：
        c, u, h: (n,3), (n,3), (n,) 中心/单位方向/半长
        pbc: 是否启用周期边界（27 镜像）
        wrap_crossings: 问题1 恒 False（True 仅后续随机仿真）
    返回语义化 dict（键见 docstring 末尾示例）。

    管线（规格第 10 节）：
        1. 根据介质中心计算最小镜像位移；
        2. 周期 AABB 粗筛排除不可能接触的介质对；
        3. 对候选介质对枚举实际可能命中的周期平移；
        4. 轴线胶囊距离下界继续筛选；
        5. 对不能排除的情况调用 GJK；
        6. 距离 <= 1.8 时加入接触边并执行并查集合并；
        7. BFS 输出一条导电见证路径。
    """
    n = len(c)
    u = np.asarray(u, dtype=float)
    c = np.asarray(c, dtype=float)
    h = np.asarray(h, dtype=float)

    # ---- 电极接触（无周期）----
    left, right = [], []
    for i in range(n):
        if electrode_contact(c[i], u[i], h[i], r, delta, 'L', wrap_crossings):
            left.append(i)
        if electrode_contact(c[i], u[i], h[i], r, delta, 'R', wrap_crossings):
            right.append(i)
    left_set = set(left)
    right_set = set(right)

    # ---- AABB 预计算 ----
    lo, hi = cylinder_aabb(c, u, h, r)  # (n,3)

    # ---- 候选平移：中心最小镜像位移 + 27 偏移邻域 ----
    # k0: (n,n,3) 最小镜像位移（仅上三角有意义，全算简化索引）
    dc = c[:, None, :] - c[None, :, :]
    k0 = np.clip(np.round(dc / side_len), -1, 1).astype(int)

    if pbc:
        offsets = np.array([(dx, dy, dz)
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)], dtype=int)  # 27
    else:
        offsets = np.zeros((1, 3), dtype=int)

    # ---- 周期 AABB 粗筛（27 偏移逐掩码）----
    # 对每对 (i,j) 与候选 k：三轴间隙均 <= delta 才保留。
    # 逐轴比较广播：i 侧 (n,1,3)，j 侧 (1,n,3)，k (n,n,3)。
    pair_mask = np.zeros((n, n), dtype=bool)
    for off in offsets:
        if pbc:
            k = np.clip(k0 + off[None, None, :], -1, 1)  # (n,n,3)
        else:
            k = np.zeros((n, n, 3), dtype=int)  # 关闭 PBC：仅主晶胞
        ok = np.ones((n, n), dtype=bool)
        for a in range(3):
            ka = k[:, :, a].astype(float)
            shift = side_len * ka
            ok &= (lo[:, None, a] - (hi[None, :, a] + shift) <= delta)
            ok &= ((lo[None, :, a] + shift) - hi[:, None, a] <= delta)
        pair_mask |= ok
    pair_mask &= np.triu(np.ones((n, n), dtype=bool), 1)  # i < j

    i_pairs, j_pairs = np.nonzero(pair_mask)
    aabb_candidate_pairs = len(i_pairs)

    # ---- 候选平移枚举 + 胶囊下界批量筛选 ----
    # 收集 (i, j, k) 三元组；对每对保留通过 AABB 的全部 k
    triples = []  # (i, j, k) 用 int
    for i, j in zip(i_pairs.tolist(), j_pairs.tolist()):
        kbase = k0[i, j]
        kcands = []
        for off in offsets:
            k = np.clip(kbase + off, -1, 1)
            if aabb_pbc_overlap(lo[i], hi[i], lo[j], hi[j], k, side_len, delta):
                kcands.append(tuple(k.tolist()))
        for k in kcands:
            triples.append((int(i), int(j), k))

    if triples:
        triples_arr = np.array(triples, dtype=object)
        i_arr = np.array([t[0] for t in triples], dtype=int)
        j_arr = np.array([t[1] for t in triples], dtype=int)
        k_arr = np.array([t[2] for t in triples], dtype=float)

        # 胶囊下界：轴线距离 s，下界 = max(s - 2r, 0)。
        # 线段方向与平移无关：u_vec = b1-a1 固定，v_vec = b2-a2 固定，
        # 仅起点平移 a2_k = a2 + L k。
        a1 = c[i_arr] - h[i_arr][:, None] * u[i_arr]      # (m,3)
        b1 = c[i_arr] + h[i_arr][:, None] * u[i_arr]
        a2 = c[j_arr] - h[j_arr][:, None] * u[j_arr]
        b2 = c[j_arr] + h[j_arr][:, None] * u[j_arr]
        a2k = a2 + side_len * k_arr

        uvec = b1 - a1                                    # (m,3)
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
        # t < 0：clamp t=0, s = clamp(-uw/uu)
        m_lo = t_par < 0.0
        s_par = np.where(m_lo, np.clip(-uw / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
        t_par = np.where(m_lo, 0.0, t_par)
        # t > 1：clamp t=1, s = clamp((uv - uw)/uu)
        m_hi = t_par > 1.0
        s_par = np.where(m_hi, np.clip((uv - uw) / np.maximum(uu, 1e-24), 0.0, 1.0), s_par)
        t_par = np.where(m_hi, 1.0, t_par)

        p1 = a1 + s_par[:, None] * uvec
        p2 = a2k + t_par[:, None] * vvec
        axis_dist = np.linalg.norm(p1 - p2, axis=1)

        # 下界过滤：轴线距离 s <= 2r + delta 才可能接触（等价 d_cap <= delta）
        keep = axis_dist <= 2.0 * r + delta
        gjk_triples = [(i_arr[m], j_arr[m], k_arr[m]) for m in np.nonzero(keep)[0]]
    else:
        gjk_triples = []

    # ---- GJK 精确距离 ----
    uf = UnionFind(n + 2)  # 虚拟节点：S = n, T = n + 1
    s_node, t_node = n, n + 1
    for i in left:
        uf.union(s_node, i)
    for i in right:
        uf.union(t_node, i)

    contact_edges = []  # (i, j, k, d)
    best = {}           # (i,j) -> (d, k)
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

    # 最大连通分量 = 全图（含电极边）最大分量中的介质节点数，S/T 不计入大小。
    # 与文档口径一致：组1=8、组2=34、组3=308。
    comp_sizes = uf.component_sizes()
    if comp_sizes:
        root = max(comp_sizes, key=comp_sizes.get)
        media_in_largest = sum(1 for x in range(n) if uf.find(x) == root)
        max_component_size = media_in_largest
    else:
        max_component_size = 0

    # 邻接表 + BFS 见证路径
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
        'max_component_size': max_component_size,      # 全图最大分量中的介质数（不含 S/T）
        'witness_path': witness_path,
    }


# ---------------- 数据读取与检查 ----------------

def build_cylinders(ends, r=R):
    """由端点 (n,2,3) 计算中心/单位方向/半长。端点重合抛 ValueError。"""
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
    """读取附件一个工作表并做规格第 9 节全部数据检查。

    表结构：第 1 行表头（端点1/端点2）、第 2 行 X/Y/Z 标签、
    第 3 行起为数据，每行 6 列（前 3 列端点1，后 3 列端点2）。
    按工作表索引读取（表名为中文，GBK 环境下乱码，按索引避开）。

    检查项：6 坐标齐全、跳过空行、拒部分坐标、坐标范围 [-5000,5000]、
    端点不重合、编号按有效行从 A1 起、表间不混（调用方逐表调用）。

    返回 (c, u, h, node_names)。
    """
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
            continue  # 完全为空的数据行：忽略
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
