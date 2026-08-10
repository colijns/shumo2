# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

from dataclasses import dataclass
import importlib.util
from itertools import product
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q1_DIR = os.path.join(ROOT, 'Q1')
Q2_DIR = os.path.join(ROOT, 'Q2')

_CORE_SPEC = importlib.util.spec_from_file_location(
    'q1_core_for_q4', os.path.join(Q1_DIR, 'core.py'))
_q1_core = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(_q1_core)
DELTA = _q1_core.DELTA
HALF_L = _q1_core.HALF_L
L = _q1_core.L
R = _q1_core.R
GJKError = _q1_core.GJKError
UnionFind = _q1_core.UnionFind
_closest_point_to_origin = _q1_core._closest_point_to_origin
gjk_distance = _q1_core.gjk_distance
segment_distance = _q1_core.segment_distance
support_cylinder = _q1_core.support_cylinder

_Q2_SPEC = importlib.util.spec_from_file_location(
    'q2_geometry_for_q4', os.path.join(Q2_DIR, 'geometry.py'))
q2geo = importlib.util.module_from_spec(_Q2_SPEC)
_Q2_SPEC.loader.exec_module(q2geo)


R_A = float(R)
R_B = 200.0
V_A = np.pi * R_A ** 2 * q2geo.CYL_LEN
V_B = 4.0 * np.pi * R_B ** 3 / 3.0
V_BOX = L ** 3
COST_DENSITY_A = 1.05 / 1e9
COST_DENSITY_B = 0.05 / 1e9
COST_PER_A = COST_DENSITY_A * V_A
COST_PER_B = COST_DENSITY_B * V_B
TOL = 1e-9


@dataclass
class Shape:

    kind: str
    source: int
    center: np.ndarray
    radius: float
    lo: np.ndarray
    hi: np.ndarray
    axis: np.ndarray | None = None
    half: float = 0.0
    clipped: bool = False

    def support(self, direction):
        direction = np.asarray(direction, dtype=float)
        if self.kind == 'A':
            return support_cylinder(
                direction, self.center, self.axis, self.half, self.radius)
        if self.clipped:
            return support_clipped_ball(
                direction, self.center, self.radius,
                -HALF_L * np.ones(3), HALF_L * np.ones(3))
        norm = np.linalg.norm(direction)
        if norm < 1e-14:
            return self.center.copy()
        return self.center + self.radius * direction / norm


def generate_spheres(n, rng):
    return rng.uniform(-HALF_L, HALF_L, size=(int(n), 3))


def _box_distance_sq(point, lower, upper):
    nearest = np.minimum(np.maximum(point, lower), upper)
    delta = point - nearest
    return float(delta @ delta)


def support_clipped_ball(direction, center, radius, lower, upper):
    v = np.asarray(direction, dtype=float)
    c = np.asarray(center, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if _box_distance_sq(c, lower, upper) > radius ** 2 + 1e-7:
        raise ValueError('球与盒子没有交集')

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-14:
        return np.minimum(np.maximum(c, lower), upper)

    rel_lo = lower - c
    rel_hi = upper - c
    y_ball = radius * v / norm_v
    if np.all(y_ball >= rel_lo - TOL) and np.all(y_ball <= rel_hi + TOL):
        return c + y_ball


    y_box = np.where(v > 0.0, rel_hi,
                     np.where(v < 0.0, rel_lo,
                              np.minimum(np.maximum(0.0, rel_lo), rel_hi)))
    if np.linalg.norm(y_box) <= radius + 1e-10:
        return c + y_box


    low = 0.0
    high = max(norm_v / max(radius, 1e-30), 1e-12)
    for _ in range(100):
        y = np.minimum(np.maximum(v / high, rel_lo), rel_hi)
        if np.linalg.norm(y) <= radius:
            break
        high *= 2.0
    else:
        raise RuntimeError('截断球支撑函数未找到可行拉格朗日乘子')

    for _ in range(90):
        mid = 0.5 * (low + high)
        y = np.minimum(np.maximum(v / max(mid, 1e-300), rel_lo), rel_hi)
        if np.linalg.norm(y) > radius:
            low = mid
        else:
            high = mid
    y = np.minimum(np.maximum(v / high, rel_lo), rel_hi)
    return c + y


def wrap_spheres(centers, radius=R_B):
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    box_lo = -HALF_L * np.ones(3)
    box_hi = HALF_L * np.ones(3)
    shapes = []
    for source, original in enumerate(centers):
        for shift_index in product((-1, 0, 1), repeat=3):
            shifted = original + L * np.asarray(shift_index, dtype=float)
            if _box_distance_sq(shifted, box_lo, box_hi) > radius ** 2 + TOL:
                continue
            raw_lo = shifted - radius
            raw_hi = shifted + radius
            lo = np.maximum(raw_lo, box_lo)
            hi = np.minimum(raw_hi, box_hi)
            clipped = bool(
                np.any(lo > raw_lo + TOL) or np.any(hi < raw_hi - TOL))
            shapes.append(Shape(
                kind='B', source=source, center=shifted.copy(), radius=radius,
                lo=lo, hi=hi, clipped=clipped))
    return shapes


def cylinder_shapes(c, u, h):
    frag = q2geo.clip_batch(c, u, h)
    out = []
    for p1, p2, source in zip(frag['p1s'], frag['p2s'], frag['cyl_idx']):
        axis_vec = p2 - p1
        length = np.linalg.norm(axis_vec)
        if length <= 1e-12:
            continue
        center = 0.5 * (p1 + p2)
        axis = axis_vec / length

        lo = np.minimum(p1, p2) - R_A
        hi = np.maximum(p1, p2) + R_A
        out.append(Shape(
            kind='A', source=int(source), center=center, radius=R_A,
            lo=lo, hi=hi, axis=axis, half=0.5 * length, clipped=False))
    return out


def gjk_support_distance(shape1, shape2, tol=1e-10, max_iter=80):
    def support(v):
        return shape1.support(v) - shape2.support(-v)

    v = shape2.center - shape1.center
    if np.linalg.norm(v) < 1e-12:
        v = np.array([1.0, 0.0, 0.0])
    simplex = []
    for _ in range(max_iter):
        wm = support(-v)
        scale = tol * max(1.0, float(v @ v))
        if v @ wm >= v @ v - scale:
            return float(np.linalg.norm(v))
        wp = support(v)
        w = wp if wp @ v < v @ v - scale else wm
        simplex.append(w)
        v_new, keep = _closest_point_to_origin(simplex)
        simplex = [simplex[i] for i in keep]
        if np.linalg.norm(v_new) <= tol:
            return 0.0
        if np.linalg.norm(v_new - v) <= tol * 1e-3:
            return 0.0
        v = v_new
    raise GJKError(f'Q4通用GJK未收敛: dist={np.linalg.norm(v)}')


def point_to_cylinder_distance(point, cylinder):
    q = np.asarray(point, dtype=float) - cylinder.center
    axial = abs(float(q @ cylinder.axis))
    radial_vec = q - (q @ cylinder.axis) * cylinder.axis
    radial = np.linalg.norm(radial_vec)
    da = max(axial - cylinder.half, 0.0)
    dr = max(radial - cylinder.radius, 0.0)
    return float(np.hypot(da, dr))


def shape_distance(shape1, shape2):
    if shape1.kind == 'B' and shape2.kind == 'B':
        if not shape1.clipped and not shape2.clipped:
            return max(float(np.linalg.norm(shape1.center - shape2.center))
                       - shape1.radius - shape2.radius, 0.0)
        return gjk_support_distance(shape1, shape2)

    if shape1.kind == 'A' and shape2.kind == 'B':
        cyl, ball = shape1, shape2
    elif shape1.kind == 'B' and shape2.kind == 'A':
        cyl, ball = shape2, shape1
    else:
        return gjk_distance(
            shape1.center, shape1.axis, shape1.half,
            shape2.center, shape2.axis, shape2.half, r=R_A)

    if not ball.clipped:
        return max(point_to_cylinder_distance(ball.center, cyl)
                   - ball.radius, 0.0)
    return gjk_support_distance(cyl, ball)


def contact_within(shape1, shape2, delta=DELTA):
    if shape1.kind == 'A' and shape2.kind == 'A':
        p1 = shape1.center - shape1.half * shape1.axis
        p2 = shape1.center + shape1.half * shape1.axis
        q1 = shape2.center - shape2.half * shape2.axis
        q2 = shape2.center + shape2.half * shape2.axis
        axis_gap = segment_distance(p1, p2, q1, q2)
        if axis_gap > shape1.radius + shape2.radius + delta:
            return False, False
        distance = gjk_distance(
            shape1.center, shape1.axis, shape1.half,
            shape2.center, shape2.axis, shape2.half, r=R_A)
        return distance <= delta + 1e-6, True

    if shape1.kind == 'B' and shape2.kind == 'B':
        full_ball_gap = max(
            float(np.linalg.norm(shape1.center - shape2.center))
            - shape1.radius - shape2.radius, 0.0)
        if full_ball_gap > delta:
            return False, False
        if not shape1.clipped and not shape2.clipped:
            return True, False
        return gjk_support_distance(shape1, shape2) <= delta + 1e-6, True

    if shape1.kind == 'A':
        cylinder, ball = shape1, shape2
    else:
        cylinder, ball = shape2, shape1
    full_ball_gap = max(
        point_to_cylinder_distance(ball.center, cylinder) - ball.radius, 0.0)
    if full_ball_gap > delta:
        return False, False
    if not ball.clipped:
        return True, False
    return gjk_support_distance(cylinder, ball) <= delta + 1e-6, True


def aabb_candidate_pairs(shapes, delta=DELTA):
    n = len(shapes)
    if n < 2:
        return []
    lo = np.asarray([s.lo for s in shapes])
    hi = np.asarray([s.hi for s in shapes])
    order = np.argsort(lo[:, 0], kind='stable')
    pairs = []
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            if lo[j, 0] - hi[i, 0] > delta:
                break
            gap_y = max(lo[i, 1] - hi[j, 1], lo[j, 1] - hi[i, 1], 0.0)
            if gap_y > delta:
                continue
            gap_z = max(lo[i, 2] - hi[j, 2], lo[j, 2] - hi[i, 2], 0.0)
            if gap_z <= delta:
                pairs.append((int(i), int(j)))
    return pairs


def electrode_flags(shape, delta=DELTA):
    if shape.kind == 'B':
        xmin, xmax = shape.lo[0], shape.hi[0]
    else:
        ux = float(shape.axis[0])
        ex = shape.half * abs(ux) + shape.radius * np.sqrt(max(1.0 - ux * ux, 0.0))
        xmin = shape.center[0] - ex
        xmax = shape.center[0] + ex
    return (xmin + HALF_L <= delta, HALF_L - xmax <= delta)


def sample_conductive(c_a, u_a, h_a, c_b, delta=DELTA):
    a_shapes = cylinder_shapes(c_a, u_a, h_a) if len(c_a) else []
    b_shapes = wrap_spheres(c_b) if len(c_b) else []
    shapes = a_shapes + b_shapes
    n = len(shapes)
    if n == 0:
        return {'conductive': False, 'n_fragments': 0, 'n_A_fragments': 0,
                'n_B_fragments': 0, 'n_A_crossing': 0,
                'n_B_crossing': 0, 'n_edges': 0, 'n_candidates': 0,
                'n_exact': 0, 'n_left': 0, 'n_right': 0}

    a_counts = np.bincount(
        [shape.source for shape in a_shapes], minlength=len(c_a))
    b_counts = np.bincount(
        [shape.source for shape in b_shapes], minlength=len(c_b))
    n_a_crossing = int(np.count_nonzero(a_counts > 1))
    n_b_crossing = int(np.count_nonzero(b_counts > 1))

    flags = [electrode_flags(shape, delta) for shape in shapes]
    left = [i for i, (is_left, _) in enumerate(flags) if is_left]
    right = [i for i, (_, is_right) in enumerate(flags) if is_right]
    if not left or not right:
        return {'conductive': False, 'n_fragments': n,
                'n_A_fragments': len(a_shapes), 'n_B_fragments': len(b_shapes),
                'n_A_crossing': n_a_crossing,
                'n_B_crossing': n_b_crossing,
                'n_edges': 0, 'n_candidates': 0, 'n_exact': 0,
                'n_left': len(left), 'n_right': len(right)}

    uf = UnionFind(n + 2)
    source_node, target_node = n, n + 1
    for i in left:
        uf.union(source_node, i)
    for i in right:
        uf.union(target_node, i)

    candidates = aabb_candidate_pairs(shapes, delta)
    n_edges = 0
    n_exact = 0
    for i, j in candidates:
        is_contact, used_iterative = contact_within(
            shapes[i], shapes[j], delta)
        n_exact += int(used_iterative)
        if is_contact:
            uf.union(i, j)
            n_edges += 1

    return {
        'conductive': bool(uf.connected(source_node, target_node)),
        'n_fragments': n,
        'n_A_fragments': len(a_shapes),
        'n_B_fragments': len(b_shapes),
        'n_A_crossing': n_a_crossing,
        'n_B_crossing': n_b_crossing,
        'n_edges': n_edges,
        'n_candidates': len(candidates),
        'n_exact': n_exact,
        'n_left': len(left),
        'n_right': len(right),
    }


def generate_sample(n_a, n_b, rng):
    c_a, u_a, h_a = q2geo.generate_cylinders(int(n_a), rng)
    c_b = generate_spheres(int(n_b), rng)
    return sample_conductive(c_a, u_a, h_a, c_b)
