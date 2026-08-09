"""问题1：按片段独立统一口径复算附件三组介质 A。

规格：docs/问题1.md（第 15 节）。运行方式：
    python run_question1.py
输出：控制台三组结论 + results/question1_result.json（确定性序列化）。

JSON 确定性保证：
    - 接触边按 (i, j) 字典序排序；
    - 距离 round 6 位小数；
    - 主结果只使用零平移 k=(0,0,0)；
    - 周期镜像接触结果仅作为敏感性对照；
    - 节点名 A{idx+1}（附件行序从 1 起）。
"""

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

GROUP_NAMES = ['组1', '组2', '组3']  # 附件工作表按索引 0/1/2（表名 GBK 乱码，按索引读）


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
        c, u, h, _ = load_cylinders(XLSX, idx)  # 含规格第 9 节全部数据检查
        t0 = time.time()
        # 与问题2、3统一：附件每行按当前盒内片段实际位置判定，不通过
        # 相对边界周期镜像自动连边。周期镜像只保留为敏感性对照。
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
