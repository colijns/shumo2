"""问题1：读取附件三组介质 A，计算导电判定并输出 JSON 结果。

规格：docs/问题1.md（第 15 节）。运行方式：
    python run_question1.py
输出：控制台三组结论 + results/question1_result.json（确定性序列化）。

JSON 确定性保证：
    - 接触边按 (i, j) 字典序排序；
    - 距离 round 6 位小数；
    - 周期平移 k 为 int 列表；
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


def serialize_group(name, res, res_off):
    edges = sorted(res['contact_edges'], key=lambda e: (e[0], e[1]))
    off_edges = sorted(res_off['contact_edges'], key=lambda e: (e[0], e[1]))
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
        'pbc_off': {
            'contact_edges': [_edge_dict(*e) for e in off_edges],
            'conductive': res_off['conductive'],
            'max_component_size': res_off['max_component_size'],
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
        res = analyze_group(c, u, h)                     # 周期边界主计算
        res_off = analyze_group(c, u, h, pbc=False)      # 敏感性复算（关 PBC）
        elapsed = time.time() - t0

        groups.append(serialize_group(name, res, res_off))

        state = 'CONDUCTIVE' if res['conductive'] else 'NOT CONDUCTIVE'
        off_state = 'CONDUCTIVE' if res_off['conductive'] else 'NOT CONDUCTIVE'
        path = ' -> '.join(res['witness_path']) if res['witness_path'] else '(none)'
        print(f'{name} [sheet {idx}]: n={res["n_cylinders"]:4d} '
              f'left/right={res["n_left_contact"]}/{res["n_right_contact"]} '
              f'AABB={res["aabb_candidate_pairs"]:5d} edges={len(res["contact_edges"]):4d} '
              f'maxcomp={res["max_component_size"]:3d}')
        print(f'    PBC on : {state:<15s} path: {path}')
        print(f'    PBC off: {off_state:<15s} edges={len(res_off["contact_edges"]):4d} '
              f'(sensitivity)  [{elapsed:.3f}s]')

    payload = {'groups': groups}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nJSON written: {OUT_JSON}')

    all_ok = all(g['conductive'] for g in groups)
    print(f'RESULT: all three groups conductive = {all_ok}')


if __name__ == '__main__':
    main()
