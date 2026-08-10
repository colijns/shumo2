# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import analyze_group, load_cylinders

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(os.path.dirname(HERE), 'attachment', '附件.xlsx')
OUT_JSON = os.path.join(HERE, 'results', 'question1_result.json')

GROUP_NAMES = ['组1', '组2', '组3']


def _edge_dict(i, j, k, d):
    return {'i': f'A{i + 1}', 'j': f'A{j + 1}', 'k': [int(v) for v in k],
            'distance': round(float(d), 6)}


def serialize_group(name, res, res_periodic):
    edges = sorted(res['contact_edges'], key=lambda e: (e[0], e[1]))
    periodic_edges = sorted(
        res_periodic['contact_edges'], key=lambda e: (e[0], e[1]))
    return {
        'name': name,
        'n_cylinders': res['n_cylinders'],
        'left_contacts': [f'A{i + 1}' for i in res['left_contacts']],
        'right_contacts': [f'A{i + 1}' for i in res['right_contacts']],
        'n_left_contact': res['n_left_contact'],
        'n_right_contact': res['n_right_contact'],
        'aabb_candidate_pairs': res['aabb_candidate_pairs'],
        'contact_edges': [_edge_dict(*e) for e in edges],
        'conductive': res['conductive'],
        'max_component_size': res['max_component_size'],
        'witness_path': res['witness_path'],
        'periodic_contact_sensitivity': {
            'contact_edges': [_edge_dict(*e) for e in periodic_edges],
            'conductive': res_periodic['conductive'],
            'max_component_size': res_periodic['max_component_size'],
        },
    }


def main():
    if not os.path.exists(XLSX):
        print(f'ERROR: 附件不存在: {XLSX}')
        sys.exit(1)

    groups = []
    for idx, name in enumerate(GROUP_NAMES):
        c, u, h, _ = load_cylinders(XLSX, idx)
        t0 = time.time()


        res = analyze_group(c, u, h, pbc=False)
        res_periodic = analyze_group(c, u, h, pbc=True)
        elapsed = time.time() - t0

        groups.append(serialize_group(name, res, res_periodic))

        state = 'CONDUCTIVE' if res['conductive'] else 'NOT CONDUCTIVE'
        periodic_state = ('CONDUCTIVE' if res_periodic['conductive']
                          else 'NOT CONDUCTIVE')
        path = ' -> '.join(res['witness_path']) if res['witness_path'] else '(none)'
        print(f'{name} [sheet {idx}]: n={res["n_cylinders"]:4d} '
              f'left/right={res["n_left_contact"]}/{res["n_right_contact"]} '
              f'AABB={res["aabb_candidate_pairs"]:5d} edges={len(res["contact_edges"]):4d} '
              f'maxcomp={res["max_component_size"]:3d}')
        print(f'    unified fragments : {state:<15s} path: {path}')
        print(f'    periodic sensitivity: {periodic_state:<15s} '
              f'edges={len(res_periodic["contact_edges"]):4d}  [{elapsed:.3f}s]')

    payload = {
        'main_model': 'fragment_actual_position_without_periodic_auto_contact',
        'groups': groups,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nJSON written: {OUT_JSON}')

    states = ', '.join(
        f'{g["name"]}={"导通" if g["conductive"] else "不导通"}'
        for g in groups)
    print(f'RESULT: {states}')


if __name__ == '__main__':
    main()
