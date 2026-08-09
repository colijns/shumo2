# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题4 几何内核：介质A（圆柱）+ 介质B（球）混合接触网络与导通判定。

建模口径（docs/问题4.md 同步，主口径：片段独立参与接触判定）：
- A 片段：与 Q2/Q3 一致——clip_batch 截断平移，AABB+胶囊初筛+GJK 精算；
- B 片段：完整球模型（口径确认）。球心在盒内保留本体片段；越界球沿
  相应方向平移 ±L 生成完整球片段（球冠∩盒的超集近似，与 Q2 圆柱半径
  伸出盒外仍按完整圆柱处理的口径一致）。同源编号仅追踪来源，不自动
  连边，只有实际表面距离 ≤ δ 才建电连接；
- 三类接触（均 ≤ δ = 1.8 nm 建边）：
  A-A: 轴距只作安全拒绝；侧面最近点可解析接受，其余候选走 GJK 精算；
  A-B: 球心到有限平端实心圆柱的精确距离减 r_B；
  B-B: max(‖c_i - c_j‖ - 2r_B, 0)；
- 电极距离：A 用 Q2 electrode_dist；B 用 max(c_x ∓ r_B ± HALF_L, 0)；
- 成本：C = 1.05·N_A·V_A + 0.05·N_B·V_B（元，V 以 μm³ 计）。单件成本
  c_A ≈ 1.4844e-2、c_B ≈ 1.6755e-3 由题目尺寸与单位体积价格换算，
  属派生值，不是额外给定参数。

复用（只读）：Q2/geometry.py（clip_batch、electrode_dist、GJK 等）、
Q3/first_passage.py（crossing_count 圆柱跨壁自检）、Q1/core.py UnionFind。
"""

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
import geometry as geo  # noqa: E402
from core import UnionFind  # noqa: E402
from first_passage import crossing_count as crossing_count_cylinders  # noqa: E402

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001  （scipy 不可用时走 numpy 矩阵兜底）
    cKDTree = None
    _HAS_SCIPY = False

R_A = geo.R                    # 介质A 半径 (nm) = 30
R_B = 200.0                    # 介质B 半径 (nm) = 0.2 μm
DELTA = geo.DELTA              # 电接触临界距离 (nm) = 1.8
L = geo.L                      # 微构体边长 (nm) = 10000
HALF_L = geo.HALF_L            # 半边长 = 5000

CYL_LEN = 5000.0               # 介质A 圆柱长度 (nm) = 5 μm
V_A_NM3 = np.pi * R_A ** 2 * CYL_LEN          # ≈ 1.4137e7 nm³
V_B_NM3 = 4.0 / 3.0 * np.pi * R_B ** 3        # ≈ 3.3510e7 nm³
V_A_UM3 = V_A_NM3 / 1e9        # ≈ 1.4137e-2 μm³
V_B_UM3 = V_B_NM3 / 1e9        # ≈ 3.3510e-2 μm³
C_A_PER_UM3 = 1.05             # 介质A 单位体积价格 (元/μm³)
C_B_PER_UM3 = 0.05             # 介质B 单位体积价格 (元/μm³)
c_A = C_A_PER_UM3 * V_A_UM3    # 单件A成本 ≈ 1.4844e-2 元
c_B = C_B_PER_UM3 * V_B_UM3    # 单件B成本 ≈ 1.6755e-3 元

THRESH_AB = R_A + R_B + DELTA  # 231.8：球心到轴线段的安全候选阈值
THRESH_BB = 2.0 * R_B + DELTA  # 401.8：球心距初筛阈值（B-B 表面距 ≤ δ 等价）
TOL_CONTACT = 1e-6             # 建边浮点容差

def bb_pairs(coords, thresh=THRESH_BB):
    """球心候选对（距离 ≤ thresh，i<j）。scipy cKDTree 优先，numpy 矩阵兜底。"""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    if _HAS_SCIPY:
        pairs = cKDTree(coords).query_pairs(thresh, output_type='ndarray')
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    lo = coords - thresh
    hi = coords + thresh
    ok = np.ones((n, n), dtype=bool)
    for a in range(3):
        ok &= (lo[:, None, a] - hi[None, :, a] <= 0.0)
        ok &= (lo[None, :, a] - hi[:, None, a] <= 0.0)
    ok &= np.triu(np.ones((n, n), dtype=bool), 1)
    ii, jj = np.nonzero(ok)
    if len(ii) == 0:
        return np.empty((0, 2), dtype=np.int64)
    d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    m = d <= thresh + TOL_CONTACT
    return np.stack([ii[m], jj[m]], axis=1)


def generate_balls(n, rng):
    """随机生成 n 个介质B。返回 (n,3) 球心，三坐标独立均匀 U(-5000, 5000)。"""
    return rng.uniform(-HALF_L, HALF_L, size=(n, 3))


def clip_balls(cs):
    """球越界片段：完整球模型。

    球心在盒内保留本体片段；某轴 c_k + r_B > HALF_L 越 + 界时生成沿 -L
    平移的完整球片段，c_k - r_B < -HALF_L 时生成沿 +L 平移的完整球片段。
    多轴同时越界时必须枚举组合平移，例如同时跨 x、y 时还需生成
    (-L,-L,0) 镜像。因此每球最多产生 2^3=8 个周期片段。
    返回 {'c_img': (m,3) 片段球心, 'ball_idx': (m,) int 来源编号}。
    """
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
    """球跨壁比例（自检对照理论 1-(1-2r_B/L)³ ≈ 11.53%）。返回 float。"""
    cs = np.asarray(cs, dtype=float)
    if len(cs) == 0:
        return 0.0
    cross = np.any(np.abs(cs) + R_B > HALF_L, axis=1)
    return float(cross.mean())


def sphere_electrode_dist(cs):
    """球片段到左右带电面 x=±5000 的最短表面距离（向量化）。

    返回 (dL, dR)，clamp ≥ 0（球穿过电极面即接触，距离 0）。
    """
    cs = np.asarray(cs, dtype=float)
    # 电极是有限正方形而不是无限平面。周期镜像球心可能在 y/z 方向位于
    # 基本盒外，必须计算球心到电极正方形的距离，不能只看 x 坐标。
    dy = np.maximum(np.abs(cs[:, 1]) - HALF_L, 0.0)
    dz = np.maximum(np.abs(cs[:, 2]) - HALF_L, 0.0)
    center_left = np.sqrt((cs[:, 0] + HALF_L) ** 2 + dy ** 2 + dz ** 2)
    center_right = np.sqrt((cs[:, 0] - HALF_L) ** 2 + dy ** 2 + dz ** 2)
    dL = np.maximum(center_left - R_B, 0.0)
    dR = np.maximum(center_right - R_B, 0.0)
    return dL, dR


def cost(na, nb):
    """总成本（元）：C = 1.05·N_A·V_A + 0.05·N_B·V_B（V 以 μm³ 计）。"""
    return c_A * na + c_B * nb


def _segment_distance_params_batch(a1, b1, a2, b2):
    """线段距离及最近点参数 s,t（与 Q2 Ericson 批量实现同分支）。"""
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
    """A-A 全量接触边（片段全局 id 对）：AABB 预筛 → 轴距解析 → 临界带 GJK 复核。

    轴线段距离减 2R_A 是胶囊体距离，不是有限平端圆柱的精确距离。它只能
    安全拒绝 d_axis > 2R_A+δ 的候选；所有剩余候选均须调用 GJK 精算。
    """
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
        # 两最近轴点均严格位于线段内部时，公共法向同时垂直于两轴，
        # 此时平端圆柱的侧面距离精确等于 max(d_axis-2R,0)。
        # 涉及任一端点时必须走 GJK，防止把胶囊半球误当成圆柱端面。
        interior = (1e-10 < s[m] < 1.0 - 1e-10
                    and 1e-10 < t[m] < 1.0 - 1e-10)
        if interior or geo._fragment_gjk(
                p1s[fi], p2s[fi], p1s[fj], p2s[fj]) \
                <= DELTA + TOL_CONTACT:
            out.append((fi, fj))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def point_finite_cylinder_distance_batch(p1s, p2s, points, radius=R_A):
    """点到有限平端实心圆柱的精确距离（向量化）。

    设点相对圆柱中心的轴向超出量为 a，径向超出量为 b，则到实体的距离为
    hypot(a,b)。该公式正确处理圆柱侧面、端面及端面圆周外侧区域。
    """
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
    """A-B 全量接触边（A 段全局 id, B 片段全局 id）。

    球与圆柱接触当且仅当球心到有限平端圆柱实体的距离 ≤ r_B+δ。
    轴线段距离 ≤ r_A+r_B+δ 只作为安全候选筛选，不能直接判连。
    """
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
    """预计算一次试验的全部片段与全量接触边（跨候选复用，避免重复计算）。

    c, u, h: 完整 A 序列（na_max 根）；ball_cs: 完整 B 序列（nb_max 个）。
    共同随机数：trial_seed = BASE_SEED + trial_index，每个试验只 prepare 一次，
    全部候选 (N_A, N_B) 前缀共享同一份边列表（sample_prefix 内按前缀过滤）。

    接触边均带来源编号（cyl_idx / ball_idx），前缀过滤 = 两边来源都 < 候选数量。
    返回 dict：
        A 片段：p1s/p2s/cyl_idx/dL_A/dR_A/a_lo/a_hi；
        B 片段：c_img/ball_idx/dL_B/dR_B/b_lo/b_hi；
        全量接触边：aa_edges (A,A)、ab_edges (A段,B片段)、bb_edges (B片段,B片段)。
    """
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
    """前缀 (n_a, n_b) 导通判定：并查集建图，早停。

    节点布局：A 片段本地 0..nA-1（按全局顺序），B 片段 nA..nA+nB-1。
    返回 bool：S/T 是否同一连通分量。
    """
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

    # 接触边全部在 prepare_trial 全量预计算（本 trial 的 na_max/nb_max 全图）。
    # 前缀过滤：边的两端来源编号都 < 候选数量 → union（共同随机数下 Y_s 单调
    # 不减的来源：前缀子图是全集子图，边集合是子集）。
    # A-A：边 (A 段全局 id, A 段全局 id)
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

    # A-B：边 (A 段全局 id, B 片段全局 id)
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

    # B-B：边 (B 片段全局 id, B 片段全局 id)
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
    """完整图导通判定（测试/demo/绘图用）。返回 dict{conductive, ...}。"""
    pr = prepare_trial(c, u, h, ball_cs, len(ball_cs))
    return {'conductive': bool(sample_prefix(pr, len(c), len(ball_cs),
                                             delta=delta)),
            'n_cylinders': int(len(c)), 'n_balls': int(len(ball_cs))}


def first_contact_nb(ball_cs, delta=DELTA):
    """纯B 单次试验：增量逐球加入，返回首次导通球数 N_c（None = 未导通）。

    与 Q3 first_contact_n 同构：全部片段一次配对（bb_pairs），按
    max(ball_idx) 分桶，逐球加入时只处理该球引出的接触对；早停。
    """
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
    # 冒烟：混合小样本 + 纯B 首次导通
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
