# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q2_DIR = os.path.join(ROOT, 'Q2')
Q1_DIR = os.path.join(ROOT, 'Q1')
for path in (Q2_DIR, Q1_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import geometry as axis_geometry
import solid_geometry as solid_geometry
from core import UnionFind


def _group_fragment_indices(fragments, n_sources):
    groups = [[] for _ in range(n_sources)]
    for index, fragment in enumerate(fragments):
        groups[fragment.source].append(index)
    return [np.asarray(group, dtype=int) for group in groups]


def _candidate_pairs(new_indices, old_count, fragments, delta):
    pairs_i = []
    pairs_j = []
    if old_count:
        old_indices = np.arange(old_count, dtype=int)
        new_lo = np.asarray([fragments[i].lo for i in new_indices])
        new_hi = np.asarray([fragments[i].hi for i in new_indices])
        old_lo = np.asarray([fragments[i].lo for i in old_indices])
        old_hi = np.asarray([fragments[i].hi for i in old_indices])
        ok = np.ones((len(new_indices), old_count), dtype=bool)
        for coordinate in range(3):
            ok &= (new_lo[:, None, coordinate]
                   - old_hi[None, :, coordinate] <= delta)
            ok &= (old_lo[None, :, coordinate]
                   - new_hi[:, None, coordinate] <= delta)
        ii, jj = np.nonzero(ok)
        pairs_i.extend(new_indices[ii].tolist())
        pairs_j.extend(old_indices[jj].tolist())

    for left in range(len(new_indices)):
        for right in range(left + 1, len(new_indices)):
            i = int(new_indices[left])
            j = int(new_indices[right])
            gap = np.maximum(
                np.maximum(fragments[i].lo - fragments[j].hi,
                           fragments[j].lo - fragments[i].hi),
                0.0,
            )
            if np.all(gap <= delta):
                pairs_i.append(i)
                pairs_j.append(j)

    if not pairs_i:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0

    i_array = np.asarray(pairs_i, dtype=int)
    j_array = np.asarray(pairs_j, dtype=int)
    p1 = np.asarray([fragment.axis_p1 for fragment in fragments])
    p2 = np.asarray([fragment.axis_p2 for fragment in fragments])
    axis_distance = axis_geometry.segment_distance_batch(
        p1[i_array], p2[i_array], p1[j_array], p2[j_array])
    radii = np.asarray(
        [fragment.enclosing_radius for fragment in fragments])
    keep = axis_distance <= radii[i_array] + radii[j_array] + delta + 1e-9
    return i_array[keep], j_array[keep], len(i_array)


def first_contact_n(c, u, h, n_sides=64, mode='inscribed',
                    delta=axis_geometry.DELTA):
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    h = np.asarray(h, dtype=float)
    n_sources = len(c)
    if n_sources == 0:
        return {
            'n_contact': None, 'conductive': False, 'n_fragments': 0,
            'n_crossing': 0, 'n_edges': 0, 'n_aabb_candidates': 0,
            'n_gjk': 0,
        }

    fragments = solid_geometry.wrap_prism_fragments(
        c, u, h, n_sides=n_sides, mode=mode)
    groups = _group_fragment_indices(fragments, n_sources)
    source_crossing = np.asarray([len(group) > 1 for group in groups])
    n_fragments = len(fragments)
    uf = UnionFind(n_fragments + 2)
    source_node, target_node = n_fragments, n_fragments + 1
    n_edges = 0
    n_aabb_candidates = 0
    n_gjk = 0
    old_count = 0

    for source, indices in enumerate(groups):
        if len(indices) == 0:
            continue
        for index in indices:
            fragment = fragments[int(index)]
            if fragment.lo[0] + axis_geometry.HALF_L <= delta:
                uf.union(source_node, int(index))
            if axis_geometry.HALF_L - fragment.hi[0] <= delta:
                uf.union(target_node, int(index))

        i_pairs, j_pairs, aabb_count = _candidate_pairs(
            indices, old_count, fragments, delta)
        n_aabb_candidates += aabb_count
        for i, j in zip(i_pairs, j_pairs):
            distance = solid_geometry.gjk_polytope_distance(
                fragments[int(i)], fragments[int(j)])
            n_gjk += 1
            if distance <= delta + 1e-6:
                uf.union(int(i), int(j))
                n_edges += 1

        old_count += len(indices)
        if uf.connected(source_node, target_node):
            return {
                'n_contact': source + 1,
                'conductive': True,
                'n_fragments': n_fragments,
                'n_crossing': int(source_crossing[:source + 1].sum()),
                'n_edges': n_edges,
                'n_aabb_candidates': n_aabb_candidates,
                'n_gjk': n_gjk,
            }

    return {
        'n_contact': None,
        'conductive': False,
        'n_fragments': n_fragments,
        'n_crossing': int(source_crossing.sum()),
        'n_edges': n_edges,
        'n_aabb_candidates': n_aabb_candidates,
        'n_gjk': n_gjk,
    }


def paired_first_contacts(c, u, h, n_sides=64):
    outer = first_contact_n(
        c, u, h, n_sides=n_sides, mode='circumscribed')
    inner = first_contact_n(
        c, u, h, n_sides=n_sides, mode='inscribed')
    n_max = len(c)
    outer_value = outer['n_contact'] if outer['n_contact'] is not None else n_max + 1
    inner_value = inner['n_contact'] if inner['n_contact'] is not None else n_max + 1
    return {
        'outer': outer,
        'inner': inner,
        'bracket_valid': bool(outer_value <= inner_value),
        'uncertain_width_n': int(inner_value - outer_value),
    }


def simulate_paired_trials(n_max, m=20, seed=42, n_sides=64):
    rng = np.random.default_rng(seed)
    outer_contacts = []
    inner_contacts = []
    violations = 0
    uncertain_samples = 0
    stats = {
        'outer_gjk': 0,
        'inner_gjk': 0,
        'outer_crossing': 0,
        'inner_crossing': 0,
    }
    for _ in range(m):
        c, u, h = axis_geometry.generate_cylinders(n_max, rng)
        result = paired_first_contacts(c, u, h, n_sides=n_sides)
        outer_contacts.append(result['outer']['n_contact'])
        inner_contacts.append(result['inner']['n_contact'])
        violations += int(not result['bracket_valid'])
        uncertain_samples += int(result['uncertain_width_n'] > 0)
        for mode in ('outer', 'inner'):
            stats[mode + '_gjk'] += result[mode]['n_gjk']
            stats[mode + '_crossing'] += result[mode]['n_crossing']
    stats.update({
        'm': m,
        'n_max': n_max,
        'bracket_violations': violations,
        'uncertain_samples': uncertain_samples,
    })
    return outer_contacts, inner_contacts, stats
