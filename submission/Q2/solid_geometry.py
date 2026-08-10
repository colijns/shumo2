# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

from dataclasses import dataclass
from itertools import product
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
Q1_DIR = os.path.join(os.path.dirname(HERE), 'Q1')
for path in (HERE, Q1_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geo
from core import GJKError, UnionFind, _closest_point_to_origin


TOL = 1e-9


@dataclass
class PolyFragment:
    source: int
    vertices: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    axis_p1: np.ndarray
    axis_p2: np.ndarray
    enclosing_radius: float

    def support(self, direction):
        values = self.vertices @ np.asarray(direction, dtype=float)
        return self.vertices[int(np.argmax(values))]


def polygon_radial_radius(radius, n_sides, mode):
    n_sides = int(n_sides)
    if n_sides < 8:
        raise ValueError('横截面边数至少为8')
    if mode == 'inscribed':
        return float(radius)
    if mode == 'circumscribed':
        return float(radius) / np.cos(np.pi / n_sides)
    raise ValueError("mode 必须为 'inscribed' 或 'circumscribed'")


def radial_error_bound(radius, n_sides):
    angle = np.pi / int(n_sides)
    inner = float(radius) * (1.0 - np.cos(angle))
    outer = float(radius) * (1.0 / np.cos(angle) - 1.0)
    return max(inner, outer)


def _radial_basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = (np.array([1.0, 0.0, 0.0])
              if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def prism_faces(center, axis, half, radius=axis_geo.R, n_sides=64,
                mode='inscribed'):
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    e1, e2 = _radial_basis(axis)
    vertex_radius = polygon_radial_radius(radius, n_sides, mode)
    angles = 2.0 * np.pi * np.arange(int(n_sides)) / int(n_sides)
    radial = vertex_radius * (
        np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
    bottom = center - float(half) * axis + radial
    top = center + float(half) * axis + radial
    faces = [bottom[::-1].copy(), top.copy()]
    for j in range(int(n_sides)):
        k = (j + 1) % int(n_sides)
        faces.append(np.asarray(
            [bottom[j], bottom[k], top[k], top[j]], dtype=float))
    return faces


def _clip_polygon(poly, axis, bound, keep_leq, tol=TOL):
    if len(poly) < 3:
        return None, []

    def signed(point):
        value = point[axis] - bound
        return value if keep_leq else -value

    output = []
    intersections = []
    previous = poly[-1]
    d_previous = signed(previous)
    previous_inside = d_previous <= tol
    for current in poly:
        d_current = signed(current)
        current_inside = d_current <= tol
        if current_inside != previous_inside:
            denom = d_previous - d_current
            t = d_previous / denom if abs(denom) > 1e-30 else 0.5
            point = previous + t * (current - previous)
            point[axis] = bound
            output.append(point)
            intersections.append(point)
        if current_inside:
            output.append(current.copy())
        previous = current
        d_previous = d_current
        previous_inside = current_inside
    if len(output) < 3:
        return None, intersections
    return np.asarray(output), intersections


def _unique_points(points, tol=1e-7):
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    array = np.asarray(points, dtype=float).reshape(-1, 3)
    quantized = np.rint(array / tol).astype(np.int64)
    _, first = np.unique(quantized, axis=0, return_index=True)
    return array[np.sort(first)]


def _sort_cap(points, axis):
    points = _unique_points(points)
    if len(points) < 3:
        return None
    free = [value for value in range(3) if value != axis]
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, free[1]] - center[free[1]],
                        points[:, free[0]] - center[free[0]])
    return points[np.argsort(angles)]


def clip_faces_to_box(faces, half_l=axis_geo.HALF_L):
    clipped = [np.asarray(face, dtype=float) for face in faces]
    for axis in range(3):
        for bound, keep_leq in ((-half_l, False), (half_l, True)):
            coordinates = np.concatenate([face[:, axis] for face in clipped])
            if keep_leq:
                if coordinates.max() <= bound + TOL:
                    continue
                if coordinates.min() > bound + TOL:
                    return [], np.empty((0, 3), dtype=float)
            else:
                if coordinates.min() >= bound - TOL:
                    continue
                if coordinates.max() < bound - TOL:
                    return [], np.empty((0, 3), dtype=float)
            new_faces = []
            cut_points = []
            for face in clipped:
                new_face, intersections = _clip_polygon(
                    face, axis, bound, keep_leq)
                if new_face is not None:
                    new_faces.append(new_face)
                cut_points.extend(intersections)
            if not new_faces:
                return [], np.empty((0, 3), dtype=float)
            cap = _sort_cap(cut_points, axis)
            if cap is not None:
                new_faces.append(cap)
            clipped = new_faces

    vertices = _unique_points([point for face in clipped for point in face])
    if len(vertices) < 4:
        return [], np.empty((0, 3), dtype=float)
    if np.linalg.matrix_rank(vertices[1:] - vertices[0], tol=1e-8) < 3:
        return [], np.empty((0, 3), dtype=float)
    return clipped, vertices


def wrap_prism_fragments(c, u, h, radius=axis_geo.R, n_sides=64,
                         mode='inscribed'):
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    fragments = []
    for source in range(len(c)):
        base_faces = prism_faces(
            c[source], u[source], h[source], radius, n_sides, mode)

        base_vertices = np.vstack([base_faces[0], base_faces[1]])
        base_lo = base_vertices.min(axis=0)
        base_hi = base_vertices.max(axis=0)
        shift_choices = []
        for coordinate in range(3):
            choices = [0]
            if base_lo[coordinate] < -axis_geo.HALF_L - TOL:
                choices.append(1)
            if base_hi[coordinate] > axis_geo.HALF_L + TOL:
                choices.append(-1)
            shift_choices.append(choices)
        p1 = c[source] - h[source] * u[source]
        p2 = c[source] + h[source] * u[source]
        enclosing_radius = polygon_radial_radius(radius, n_sides, mode)
        for shift_index in product(*shift_choices):
            shift = axis_geo.L * np.asarray(shift_index, dtype=float)
            shifted_vertices = base_vertices + shift
            shifted_lo = base_lo + shift
            shifted_hi = base_hi + shift
            if (np.any(shifted_hi < -axis_geo.HALF_L - TOL)
                    or np.any(shifted_lo > axis_geo.HALF_L + TOL)):
                continue
            if (np.all(shifted_lo >= -axis_geo.HALF_L - TOL)
                    and np.all(shifted_hi <= axis_geo.HALF_L + TOL)):
                vertices = shifted_vertices
            else:
                shifted_faces = [face + shift for face in base_faces]
                _, vertices = clip_faces_to_box(shifted_faces)
            if len(vertices) == 0:
                continue
            fragments.append(PolyFragment(
                source=source, vertices=vertices,
                lo=vertices.min(axis=0), hi=vertices.max(axis=0),
                axis_p1=p1 + shift, axis_p2=p2 + shift,
                enclosing_radius=enclosing_radius))
    return fragments


def gjk_polytope_distance(shape1, shape2, tol=1e-10, max_iter=80):
    def support(direction):
        return shape1.support(direction) - shape2.support(-direction)

    v = shape2.vertices.mean(axis=0) - shape1.vertices.mean(axis=0)
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
        simplex = [simplex[index] for index in keep]
        if np.linalg.norm(v_new) <= tol:
            return 0.0
        if np.linalg.norm(v_new - v) <= tol * 1e-3:
            return 0.0
        v = v_new
    raise GJKError(f'多面体GJK未收敛: dist={np.linalg.norm(v)}')


def aabb_candidate_pairs(fragments, delta=axis_geo.DELTA):
    if len(fragments) < 2:
        return []
    lo = np.asarray([fragment.lo for fragment in fragments])
    hi = np.asarray([fragment.hi for fragment in fragments])
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


def sample_conductive_solid(c, u, h, delta=axis_geo.DELTA, n_sides=64,
                            mode='inscribed'):
    fragments = wrap_prism_fragments(c, u, h, n_sides=n_sides, mode=mode)
    n = len(fragments)
    if n == 0:
        return {'conductive': False, 'n_fragments': 0, 'n_crossing': 0,
                'n_edges': 0, 'n_candidates': 0, 'n_gjk': 0,
                'n_left': 0, 'n_right': 0}
    source_counts = np.bincount(
        [fragment.source for fragment in fragments], minlength=len(c))
    n_crossing = int(np.count_nonzero(source_counts > 1))
    left = [index for index, fragment in enumerate(fragments)
            if fragment.lo[0] + axis_geo.HALF_L <= delta]
    right = [index for index, fragment in enumerate(fragments)
             if axis_geo.HALF_L - fragment.hi[0] <= delta]
    if not left or not right:
        return {'conductive': False, 'n_fragments': n,
                'n_crossing': n_crossing, 'n_edges': 0,
                'n_candidates': 0, 'n_gjk': 0,
                'n_left': len(left), 'n_right': len(right)}

    uf = UnionFind(n + 2)
    source_node, target_node = n, n + 1
    for index in left:
        uf.union(source_node, index)
    for index in right:
        uf.union(target_node, index)

    pairs = aabb_candidate_pairs(fragments, delta)
    n_aabb_candidates = len(pairs)
    if pairs:
        i_pairs = np.asarray([pair[0] for pair in pairs], dtype=int)
        j_pairs = np.asarray([pair[1] for pair in pairs], dtype=int)
        p1 = np.asarray([fragment.axis_p1 for fragment in fragments])
        p2 = np.asarray([fragment.axis_p2 for fragment in fragments])
        axis_distance = axis_geo.segment_distance_batch(
            p1[i_pairs], p2[i_pairs], p1[j_pairs], p2[j_pairs])
        radii = np.asarray(
            [fragment.enclosing_radius for fragment in fragments])
        keep = axis_distance <= radii[i_pairs] + radii[j_pairs] + delta + TOL
        pairs = [(int(i), int(j))
                 for i, j in zip(i_pairs[keep], j_pairs[keep])]
    n_edges = 0
    for i, j in pairs:
        if gjk_polytope_distance(fragments[i], fragments[j]) <= delta + 1e-6:
            uf.union(i, j)
            n_edges += 1
    return {
        'conductive': bool(uf.connected(source_node, target_node)),
        'n_fragments': n,
        'n_crossing': n_crossing,
        'n_edges': n_edges,
        'n_candidates': n_aabb_candidates,
        'n_gjk': len(pairs),
        'n_left': len(left),
        'n_right': len(right),
    }
