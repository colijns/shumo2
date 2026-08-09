# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash，版本 / 型号：DeepSeek-V4-Flash-0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026-07-31

"""问题3配图：早停检查根数诊断（图5）。

嵌套共同样本下，每次试验在首次导通（N_c 根）后早停，实际检查根数
C(N) = sum_{k<=N} (M - x(k-1))，x(k) 为 k 根处已导通试验数。左轴画
C(N)（外切/内接），灰虚线为名义值 N*M（不早停时的 7500000 根）；
右轴画已导通试验数 x(N)。跨壁比例的分母即 C(750)：
3011987 / 4950009 = 0.6085（外切）。
数据来源：Q3/results/question3_solid_curve.csv（M=10000，K=32，
seed=20260808+i）+ question3_solid_summary.json。
输出：Q3/figures/问题3_早停检查根数_诊断.png + .pdf（300dpi，双格式），
并复制入 docs/appendix/figures/ 图池（competition-record 步骤 2+4）。
"""

import csv
import json
import os
import shutil
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_Q3 = os.path.dirname(os.path.abspath(__file__))
# 与模板 plot_style 同款中文样式（内联，避免依赖 templates 路径）
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

CURVE_CSV = os.path.join(_Q3, 'results', 'question3_solid_curve.csv')
SUMMARY_JSON = os.path.join(_Q3, 'results', 'question3_solid_summary.json')
FIG_DIR = os.path.join(_Q3, 'figures')
APPENDIX_FIG_DIR = os.path.join(os.path.dirname(_Q3), 'docs', 'appendix', 'figures')

N_MAX = 750
N_POINT_OUTER = 613
N_POINT_INNER = 616
N_SAFE = 619


def save_pair(fig, stem):
    """competition-record 步骤 2+4：png/pdf 双格式保存 + 复制入图池。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, stem)
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
        shutil.copy2(base + '.' + ext,
                     os.path.join(APPENDIX_FIG_DIR, stem + '.' + ext))
    plt.close(fig)
    print(f'已保存并入池：{stem}.png / .pdf（dpi=300）')


def load_curve():
    rows = {}
    with open(CURVE_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows[int(r['n'])] = {k: float(v) for k, v in r.items()}
    return rows


def load_summary():
    with open(SUMMARY_JSON, encoding='utf-8') as f:
        return json.load(f)


def checked_roots(rows, mode, m, n_max):
    """C(N) = sum_{k<=N} (M - x(k-1))，x(0)=0，x(k)=p_hat(k)*M。"""
    c = np.zeros(n_max + 1)
    prev = 0.0
    for k in range(1, n_max + 1):
        c[k] = c[k - 1] + (m - prev)
        prev = rows[k][mode + '_p_hat'] * m
    return c


def main():
    curve = load_curve()
    summary = load_summary()
    m = int(summary['config']['trials'])
    n_max = int(summary['config']['n_max'])
    assert m == 10000 and n_max == 750, f'M={m}, N_max={n_max}'

    c_outer = checked_roots(curve, 'outer', m, n_max)
    c_inner = checked_roots(curve, 'inner', m, n_max)
    x_outer = np.array([curve[k]['outer_p_hat'] * m
                        for k in range(1, n_max + 1)])
    nominal = m * np.arange(0, n_max + 1)

    # 数值断言（与报告 §4.1 修正一致）
    assert abs(c_outer[N_MAX] - 4950009) < 1e-6, \
        f'C(750) outer={c_outer[N_MAX]}'
    assert abs(c_inner[N_MAX] - 4970014) < 1e-6, \
        f'C(750) inner={c_inner[N_MAX]}'
    assert abs(x_outer[N_MAX - 1] - 9989) < 1.0, \
        f'x(750)={x_outer[N_MAX-1]}'
    print(f'断言通过：C(750) 外切={c_outer[N_MAX]:.0f} / '
          f'内接={c_inner[N_MAX]:.0f}，x(750)={x_outer[N_MAX-1]:.0f}')

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    n_axis = np.arange(0, n_max + 1)
    ax.plot(n_axis, nominal, color='#7f7f7f', linewidth=1.4,
            linestyle=':', label='名义检查根数 $N\\times M$（不早停）')
    ax.plot(n_axis, c_outer, color='#1f77b4', linewidth=2,
            label='外切模型实际检查 $C(N)$')
    ax.plot(n_axis, c_inner, color='#2ca02c', linewidth=1.8,
            linestyle='--', label='内接模型实际检查 $C(N)$')

    for n_mark, color, ls in ((N_POINT_OUTER, '#1f77b4', '-'),
                              (N_POINT_INNER, '#2ca02c', '-'),
                              (N_SAFE, '#2ca02c', '--')):
        ax.axvline(n_mark, color=color, linewidth=1.2, linestyle=ls)

    ax.set_xlabel('完整介质数量 N（根）')
    ax.set_ylabel('累计检查根数 C(N)（根）')
    ax.set_xlim(0, N_MAX)
    ax.set_ylim(0, 7.6e6)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(n_axis[1:], x_outer, color='#ff7f0e', linewidth=1.6,
             label='已导通试验数 $x(N)=M\\cdot\\widehat P(N)$')
    ax2.set_ylabel('已导通试验数 x(N)（次）')
    ax2.set_ylim(0, m * 1.02)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=9)

    save_pair(fig, '问题3_早停检查根数_诊断')


if __name__ == '__main__':
    main()
