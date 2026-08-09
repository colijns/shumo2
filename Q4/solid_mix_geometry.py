"""Q4严格混合实体几何：A圆柱与B球体的内接/外切多面体夹逼。

本模块不替换原 ``geometry_mix`` 的快速搜索核，而是提供正式复核核：

* A：复用Q3的内接/外切正多棱柱及周期实体裁剪；
* B：用内接/外切二十面体细分多面体逼近球，并严格裁剪到基本盒；
* A-A、A-B、B-B：统一用AABB预筛和凸体GJK距离建边；
* 同源周期片段只记录来源，不自动电连接；
* 候选使用完整介质序列前缀，保持样本级单调性。

因此同一随机样本上应满足 ``Y_inner <= Y_outer``。当 ``N_B=0`` 时，
本模块与Q3的实体首次导通核使用完全相同的A片段生成和接触判据。
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Q2_DIR = os.path.join(ROOT, 'Q2')
Q1_DIR = os.path.join(ROOT, 'Q1')
for path in (Q2_DIR, Q1_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry  # noqa: E402
import solid_geometry  # noqa: E402
from core import UnionFind  # noqa: E402


R_B = 200.0
DELTA = axis_geometry.DELTA
L = axis_geometry.L
HALF_L = axis_geometry.HALF_L
TOL = 1e-8


@dataclass
class MixedFragment:
    """统一凸片段；kind=0为A，kind=1为B。"""

    kind: int
    source: int
    vertices: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    axis_p1: np.ndarray
    axis_p2: np.ndarray
    enclosing_radius: float
    clipped: bool = False

    def support(self, direction):
        values = self.vertices @ np.asarray(direction, dtype=float)
        return self.vertices[int(np.argmax(values))]


def _base_icosahedron():
    """单位外接球上的正二十面体顶点和三角面。"""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int64)
    return vertices, faces


@lru_cache(maxsize=None)
def unit_icosphere(subdivisions=2):
    """返回单位球上的细分二十面体网格（顶点、三角面、最小面内半径）。"""
    subdivisions = int(subdivisions)
    if subdivisions < 0 or subdivisions > 4:
        raise ValueError('subdivisions必须位于0到4之间')
    vertices, faces = _base_icosahedron()
    vertices = vertices.tolist()
    faces = faces.tolist()
    for _ in range(subdivisions):
        midpoint_cache = {}

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoint_cache:
                point = np.asarray(vertices[i]) + np.asarray(vertices[j])
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(vertices)
                vertices.append(point.tolist())
            return midpoint_cache[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab),
                            (c, ca, bc), (ab, bc, ca)))
        faces = refined
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    distances = []
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        distances.append(abs(float(normal @ a)))
    min_face_radius = float(min(distances))
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces, min_face_radius


def ball_polyhedron_error(radius=R_B, subdivisions=2):
    """球与内/外接多面体的最大单侧径向误差上界。"""
    _, _, inradius = unit_icosphere(subdivisions)
    inner_error = float(radius) * (1.0 - inradius)
    outer_error = float(radius) * (1.0 / inradius - 1.0)
    return max(inner_error, outer_error)


def ball_polyhedron_faces(center, radius=R_B, subdivisions=2,
                          mode='inscribed'):
    """生成球的内接或外切凸多面体三角面。"""
    unit_vertices, face_indices, min_face_radius = unit_icosphere(subdivisions)
    if mode == 'inscribed':
        scale = float(radius)
    elif mode == 'circumscribed':
        # 将所有顶点统一放大，使每个支撑面到球心的距离都至少为radius。
        scale = float(radius) / min_face_radius
    else:
        raise ValueError("mode必须为'inscribed'或'circumscribed'")
    vertices = np.asarray(center, dtype=float) + scale * unit_vertices
    faces = [vertices[face].copy() for face in face_indices]
    return faces, vertices, scale


def wrap_ball_fragments(centers, radius=R_B, subdivisions=2,
                        mode='inscribed'):
    """球体多面体的周期镜像与基本盒实体裁剪。"""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    unit_vertices, face_indices, min_face_radius = unit_icosphere(subdivisions)
    if mode == 'inscribed':
        scale = float(radius)
    elif mode == 'circumscribed':
        scale = float(radius) / min_face_radius
    else:
        raise ValueError("mode必须为'inscribed'或'circumscribed'")
    relative_vertices = scale * unit_vertices
    fragments = []
    for source, center in enumerate(centers):
        # 绝大多数球完全位于盒内，只生成顶点；仅跨壁时构造320个三角面，
        # 避免为每个内部球重复分配大量小数组。
        base_vertices = center + relative_vertices
        base_lo = base_vertices.min(axis=0)
        base_hi = base_vertices.max(axis=0)
        choices = []
        for coordinate in range(3):
            axis_choices = [0]
            if base_lo[coordinate] < -HALF_L - TOL:
                axis_choices.append(1)
            if base_hi[coordinate] > HALF_L + TOL:
                axis_choices.append(-1)
            choices.append(axis_choices)
        for shift_index in product(*choices):
            shift = L * np.asarray(shift_index, dtype=float)
            shifted_lo = base_lo + shift
            shifted_hi = base_hi + shift
            if (np.any(shifted_hi < -HALF_L - TOL)
                    or np.any(shifted_lo > HALF_L + TOL)):
                continue
            if (np.all(shifted_lo >= -HALF_L - TOL)
                    and np.all(shifted_hi <= HALF_L + TOL)):
                vertices = base_vertices + shift
                was_clipped = False
            else:
                base_faces = [base_vertices[face] for face in face_indices]
                shifted_faces = [face + shift for face in base_faces]
                _, vertices = solid_geometry.clip_faces_to_box(shifted_faces)
                was_clipped = True
            if len(vertices) == 0:
                continue
            shifted_center = center + shift
            fragments.append(MixedFragment(
                kind=1, source=source, vertices=vertices,
                lo=vertices.min(axis=0), hi=vertices.max(axis=0),
                axis_p1=shifted_center, axis_p2=shifted_center,
                enclosing_radius=scale, clipped=was_clipped))
    return fragments


def wrap_cylinder_fragments(c, u, h, n_sides=32, mode='inscribed'):
    """复用Q3实体圆柱片段，并添加混合介质类型标签。"""
    raw = solid_geometry.wrap_prism_fragments(
        c, u, h, n_sides=n_sides, mode=mode)
    return [MixedFragment(
        kind=0, source=fragment.source, vertices=fragment.vertices,
        lo=fragment.lo, hi=fragment.hi,
        axis_p1=fragment.axis_p1, axis_p2=fragment.axis_p2,
        enclosing_radius=fragment.enclosing_radius,
        clipped=bool(np.any(np.isclose(np.abs(fragment.vertices), HALF_L,
                                       atol=1e-7))))
        for fragment in raw]


def _candidate_pairs(fragments, delta=DELTA):
    """安全AABB扫线候选；返回全局片段编号对。"""
    return solid_geometry.aabb_candidate_pairs(fragments, delta)


def _point_finite_cylinder_distance(p1, p2, point, radius):
    """点到有限平端实心圆柱的精确距离。"""
    axis = np.asarray(p2) - np.asarray(p1)
    length = float(np.linalg.norm(axis))
    unit = axis / max(length, 1e-30)
    center = 0.5 * (np.asarray(p1) + np.asarray(p2))
    relative = np.asarray(point) - center
    axial = float(relative @ unit)
    radial = float(np.linalg.norm(relative - axial * unit))
    axial_excess = max(abs(axial) - 0.5 * length, 0.0)
    radial_excess = max(radial - float(radius), 0.0)
    return float(np.hypot(axial_excess, radial_excess))


def _segment_distance_params(p1, p2, q1, q2):
    """两线段最短距离及最近点参数，参数位于[0,1]。"""
    a1 = np.asarray(p1, dtype=float)[None, :]
    b1 = np.asarray(p2, dtype=float)[None, :]
    a2 = np.asarray(q1, dtype=float)[None, :]
    b2 = np.asarray(q2, dtype=float)[None, :]
    uvec = b1 - a1
    vvec = b2 - a2
    w = a1 - a2
    uu = np.einsum('ij,ij->i', uvec, uvec)
    uv = np.einsum('ij,ij->i', uvec, vvec)
    vv = np.einsum('ij,ij->i', vvec, vvec)
    uw = np.einsum('ij,ij->i', uvec, w)
    vw = np.einsum('ij,ij->i', vvec, w)
    denominator = uu * vv - uv * uv
    s = np.where(
        denominator < 1e-24, 0.0,
        np.clip((uv * vw - vv * uw) / np.maximum(denominator, 1e-24),
                0.0, 1.0))
    t = (uv * s + vw) / np.maximum(vv, 1e-24)
    low = t < 0.0
    s = np.where(low, np.clip(-uw / np.maximum(uu, 1e-24), 0.0, 1.0), s)
    t = np.where(low, 0.0, t)
    high = t > 1.0
    s = np.where(high,
                 np.clip((uv - uw) / np.maximum(uu, 1e-24), 0.0, 1.0), s)
    t = np.where(high, 1.0, t)
    first_point = a1 + s[:, None] * uvec
    second_point = a2 + t[:, None] * vvec
    distance = np.linalg.norm(first_point - second_point, axis=1)
    return float(distance[0]), float(s[0]), float(t[0])


def _contained_radius(fragment, cyl_sides, ball_subdivisions, mode):
    if fragment.kind == 0:
        return (axis_geometry.R * np.cos(np.pi / int(cyl_sides))
                if mode == 'inscribed' else axis_geometry.R)
    _, _, inradius = unit_icosphere(ball_subdivisions)
    return R_B * inradius if mode == 'inscribed' else R_B


def _contact_edges(fragments, cyl_sides=32, ball_subdivisions=2,
                   mode='inscribed', delta=DELTA,
                   use_analytic_bounds=True):
    """AABB预筛后用安全解析界缩小临界带，其余统一调用凸体GJK。"""
    edges = []
    candidates = _candidate_pairs(fragments, delta)
    n_gjk = 0
    for i, j in candidates:
        first, second = fragments[i], fragments[j]
        accepted = False
        rejected = False
        # A-A轴线距离减外接半径是安全下界，可排除长AABB造成的大量伪候选。
        if first.kind == second.kind == 0:
            axis_distance, first_t, second_t = _segment_distance_params(
                first.axis_p1, first.axis_p2,
                second.axis_p1, second.axis_p2)
            rejected = axis_distance > (
                first.enclosing_radius + second.enclosing_radius
                + delta + TOL)
            if (use_analytic_bounds and not rejected
                    and not first.clipped and not second.clipped
                    and 1e-10 < first_t < 1.0 - 1e-10
                    and 1e-10 < second_t < 1.0 - 1e-10):
                contained = _contained_radius(
                    first, cyl_sides, ball_subdivisions, mode)
                accepted = axis_distance <= 2.0 * contained + delta + TOL
        # 未裁剪B-B：外接半径可安全拒绝，内含球可安全接受；仅窄带走GJK。
        elif (use_analytic_bounds and first.kind == second.kind == 1
              and not first.clipped and not second.clipped):
            center_distance = float(np.linalg.norm(first.axis_p1 - second.axis_p1))
            if center_distance > (first.enclosing_radius
                                  + second.enclosing_radius + delta + TOL):
                rejected = True
            else:
                contained = _contained_radius(
                    first, cyl_sides, ball_subdivisions, mode)
                accepted = center_distance <= 2.0 * contained + delta + TOL
        # 未裁剪A-B：以同轴内含/外包圆柱和B半径给安全接受/拒绝界。
        elif (use_analytic_bounds and first.kind != second.kind
              and not first.clipped and not second.clipped):
            a = first if first.kind == 0 else second
            b = second if first.kind == 0 else first
            a_contained = _contained_radius(
                a, cyl_sides, ball_subdivisions, mode)
            b_contained = _contained_radius(
                b, cyl_sides, ball_subdivisions, mode)
            lower_distance = _point_finite_cylinder_distance(
                a.axis_p1, a.axis_p2, b.axis_p1, a.enclosing_radius)
            if lower_distance > b.enclosing_radius + delta + TOL:
                rejected = True
            else:
                upper_distance = _point_finite_cylinder_distance(
                    a.axis_p1, a.axis_p2, b.axis_p1, a_contained)
                accepted = upper_distance <= b_contained + delta + TOL
        if rejected:
            continue
        if accepted:
            edges.append((i, j))
            continue
        distance = solid_geometry.gjk_polytope_distance(first, second)
        n_gjk += 1
        if distance <= delta + 1e-6:
            edges.append((i, j))
    return (np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            len(candidates), n_gjk)


def prepare_trial(c, u, h, ball_centers, n_a_max=None, n_b_max=None,
                  cyl_sides=32, ball_subdivisions=2, mode='inscribed'):
    """准备一次严格实体试验的完整前缀图。"""
    n_a_max = len(c) if n_a_max is None else int(n_a_max)
    n_b_max = len(ball_centers) if n_b_max is None else int(n_b_max)
    a_fragments = wrap_cylinder_fragments(
        np.asarray(c)[:n_a_max], np.asarray(u)[:n_a_max],
        np.asarray(h)[:n_a_max], n_sides=cyl_sides, mode=mode)
    b_fragments = wrap_ball_fragments(
        np.asarray(ball_centers)[:n_b_max], subdivisions=ball_subdivisions,
        mode=mode)
    fragments = a_fragments + b_fragments
    kind = np.asarray([fragment.kind for fragment in fragments], dtype=np.int8)
    source = np.asarray([fragment.source for fragment in fragments], dtype=np.int64)
    left = np.asarray([
        fragment.lo[0] + HALF_L <= DELTA + TOL for fragment in fragments],
        dtype=bool)
    right = np.asarray([
        HALF_L - fragment.hi[0] <= DELTA + TOL for fragment in fragments],
        dtype=bool)
    edges, n_candidates, n_gjk = _contact_edges(
        fragments, cyl_sides, ball_subdivisions, mode)
    return {
        'fragments': fragments,
        'kind': kind,
        'source': source,
        'left': left,
        'right': right,
        'edges': edges,
        'n_candidates': n_candidates,
        'n_gjk': n_gjk,
        'mode': mode,
        'cyl_sides': int(cyl_sides),
        'ball_subdivisions': int(ball_subdivisions),
    }


def sample_prefix(prepared, n_a, n_b):
    """判断完整介质前缀 ``(n_a,n_b)`` 是否左右导通。"""
    kind = prepared['kind']
    source = prepared['source']
    active = ((kind == 0) & (source < int(n_a))) \
        | ((kind == 1) & (source < int(n_b)))
    active_indices = np.nonzero(active)[0]
    n = len(active_indices)
    if n == 0:
        return False
    left_global = np.nonzero(active & prepared['left'])[0]
    right_global = np.nonzero(active & prepared['right'])[0]
    if len(left_global) == 0 or len(right_global) == 0:
        return False

    mapping = np.full(len(kind), -1, dtype=np.int64)
    mapping[active_indices] = np.arange(n, dtype=np.int64)
    uf = UnionFind(n + 2)
    s_node, t_node = n, n + 1
    for index in left_global:
        uf.union(s_node, int(mapping[index]))
    for index in right_global:
        uf.union(t_node, int(mapping[index]))
    if uf.connected(s_node, t_node):
        return True

    edges = prepared['edges']
    if len(edges):
        keep = active[edges[:, 0]] & active[edges[:, 1]]
        for i, j in edges[keep]:
            uf.union(int(mapping[i]), int(mapping[j]))
    return bool(uf.connected(s_node, t_node))


def prepare_paired(c, u, h, ball_centers, n_a_max=None, n_b_max=None,
                   cyl_sides=32, ball_subdivisions=2):
    """同一随机微构体的内接/外切成对准备。"""
    inner = prepare_trial(
        c, u, h, ball_centers, n_a_max, n_b_max,
        cyl_sides, ball_subdivisions, mode='inscribed')
    outer = prepare_trial(
        c, u, h, ball_centers, n_a_max, n_b_max,
        cyl_sides, ball_subdivisions, mode='circumscribed')
    return inner, outer


def paired_prefix(inner, outer, n_a, n_b):
    """返回(inner, outer)，并检查几何夹逼不被违反。"""
    y_inner = sample_prefix(inner, n_a, n_b)
    y_outer = sample_prefix(outer, n_a, n_b)
    if y_inner and not y_outer:
        raise AssertionError('实体几何夹逼违例：inner导通但outer不导通')
    return y_inner, y_outer
