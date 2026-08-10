# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

import csv
import json
import os
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_Q3 = os.path.dirname(os.path.abspath(__file__))
sys_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = sys_fonts
plt.rcParams['axes.unicode_minus'] = False

RESULT_CSV = os.path.join(_Q3, 'results', 'question3_solid_curve.csv')
SUMMARY_JSON = os.path.join(_Q3, 'results', 'question3_solid_summary.json')
FIG_DIR = os.path.join(_Q3, 'figures')


def load_curve():
    rows = []
    with open(RESULT_CSV, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r['n'])
    return rows


def load_summary():
    with open(SUMMARY_JSON, encoding='utf-8') as f:
        return json.load(f)


def main():
    rows = load_curve()
    summary = load_summary()
    trials = summary['config']['trials']
    n_sides = summary['config']['n_sides']
    n = np.array([r['n'] for r in rows])
    phi = np.array([r['phi_percent'] for r in rows])
    outer_p = np.array([r['outer_p_hat'] for r in rows])
    outer_lo = np.array([r['outer_ci_lower'] for r in rows])
    outer_hi = np.array([r['outer_ci_upper'] for r in rows])
    inner_p = np.array([r['inner_p_hat'] for r in rows])
    inner_lo = np.array([r['inner_ci_lower'] for r in rows])
    inner_hi = np.array([r['inner_ci_upper'] for r in rows])

    outer_safe = summary['outer_lower_threshold']['safe']['n']
    inner_safe = summary['inner_upper_threshold']['safe']['n']
    outer_point = summary['outer_lower_threshold']['point']['n']
    inner_point = summary['inner_upper_threshold']['point']['n']



    fig = plt.figure(figsize=(8, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.14,
                          left=0.105, right=0.97, top=0.97, bottom=0.11)
    ax = fig.add_subplot(gs[0])
    ax.plot(n, outer_p, '-', color='#1f77b4', linewidth=2.4,
            label='外切多棱柱（几何下界）')
    ax.fill_between(n, outer_lo, outer_hi, color='#1f77b4', alpha=0.35,
                    edgecolor='#0f5290', linewidth=0.6,
                    label='外切 95% Wilson 区间')
    ax.plot(n, inner_p, '--', color='#2ca02c', linewidth=1.6,
            label='内接多棱柱（几何上界）')
    ax.fill_between(n, inner_lo, inner_hi, color='#2ca02c', alpha=0.35,
                    edgecolor='#1c701f', linewidth=0.6,
                    label='内接 95% Wilson 区间')
    ax.axhline(0.90, color='red', linestyle=':', linewidth=1.5,
               label='90% 要求线')

    for value, color, label in (
            (outer_safe, '#1f77b4', '外切保守临界'),
            (inner_safe, '#2ca02c', '内接保守临界'),
            (outer_point, '#1f77b4', '外切点估计'),
            (inner_point, '#2ca02c', '内接点估计')):
        if value is None:
            continue
        ax.axvline(value, color=color, alpha=0.5, linewidth=1.0,
                   linestyle='-.')

    ax.set_ylabel('导通概率')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.tick_params(labelbottom=False)


    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ticks = np.linspace(n.min(), n.max(), 7)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([f'{value:.3f}' for value in np.interp(ticks, n, phi)])
    ax_top.set_xlabel('体积分数 φ（%，精确到百分号下两位以内）')


    axd = fig.add_subplot(gs[1], sharex=ax)
    delta = inner_p - outer_p
    axd.plot(n, delta, color='#333333', linewidth=1.4)
    axd.fill_between(n, delta, 0, color='#333333', alpha=0.15)
    axd.axhline(0, color='#7f7f7f', linewidth=0.8, linestyle=':')
    axd.set_ylabel('内接 $-$ 外切\n$\\Delta\\widehat P$')
    axd.set_xlabel('完整介质数量 N（根）')
    axd.set_xlim(n.min(), n.max())
    axd.set_ylim(-0.012, 0.012)
    axd.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, '问题3_实体夹逼_导通概率_vs_介质数量')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved:', base + '.png / .pdf')


if __name__ == '__main__':
    main()
