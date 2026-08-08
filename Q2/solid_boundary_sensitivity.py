# 本程序及代码是在AI工具辅助下完成的
"""问题2完整圆柱实体跨壁敏感性诊断。

正式内核按中心轴线与盒面交点切段。本程序不改变导通算法，只比较：
1. 中心轴线发生跨壁；
2. 完整平端圆柱实体发生跨壁。

二者之差是“轴线仍在盒内，但30 nm半径已越界”的边界薄层情形，可用于判断
是否值得进一步实现圆柱实体与基本盒的精确凸交片段。
"""

import json
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as geo  # noqa: E402


N_SAMPLES = int(os.environ.get('SHUMO_Q2_SOLID_SAMPLES', '1000000'))
SEED = int(os.environ.get('SHUMO_Q2_SOLID_SEED', '20260808'))
RESULT_PATH = os.path.join(
    HERE, 'results', 'solid_boundary_sensitivity.json')


def run(n_samples=N_SAMPLES, seed=SEED):
    if n_samples <= 0:
        raise ValueError('样本数必须为正整数')
    rng = np.random.default_rng(seed)
    c, u, h = geo.generate_cylinders(n_samples, rng)
    axis = geo.axis_crossing_flags(c, u, h)
    solid = geo.solid_crossing_flags(c, u, h)
    radial_only = solid & ~axis
    if np.any(axis & ~solid):
        raise AssertionError('轴线跨壁必须蕴含完整圆柱实体跨壁')

    q2_counts = [354, 424, 495, 707]
    radial_rate = float(radial_only.mean())
    return {
        'n_samples': int(n_samples),
        'seed': int(seed),
        'radius_nm': float(geo.R),
        'length_nm': float(geo.CYL_LEN),
        'axis_crossing_rate': float(axis.mean()),
        'solid_crossing_rate': float(solid.mean()),
        'radial_only_crossing_rate': radial_rate,
        'relative_increase_vs_axis': float(
            radial_rate / max(float(axis.mean()), 1e-30)),
        'expected_radial_only_count_at_q2_sizes': {
            str(n): n * radial_rate for n in q2_counts
        },
        'interpretation': (
            'radial_only表示现有轴线切段不会生成相对侧薄片、但完整圆柱实体已越界的情形'
        ),
    }


def main():
    result = run()
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f'结果已写入: {RESULT_PATH}', flush=True)


if __name__ == '__main__':
    main()
