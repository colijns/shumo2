# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


import csv
import os
import shutil
import tempfile

os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'shumo2-matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

_BASE = os.path.dirname(os.path.abspath(__file__))
CSV_AUDIT = os.path.join(_BASE, 'results', 'question4_strict_mixed_candidates.csv')
FIG_DIR = os.path.join(_BASE, 'figures')
APPENDIX_FIG_DIR = os.path.join(
    os.path.dirname(_BASE), 'docs', 'appendix', 'figures')

P_TARGET = 0.90
C_INNER = '#4C72B0'
C_OUTER = '#C44E52'
C_INSUF = '#C44E52'
C_UNDET = '#E69F00'
C_FEAS = '#55A868'
VCOLOR = {'reliably_insufficient': C_INSUF,
          'undetermined': C_UNDET,
          'reliably_feasible': C_FEAS}


def load_audit(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append({
                'na': int(r['N_A']), 'nb': int(r['N_B']),
                'cost': float(r['cost_yuan']),
                'inner_p': float(r['inner_p_hat']),
                'inner_lo': float(r['inner_ci_lower']),
                'inner_hi': float(r['inner_ci_upper']),
                'outer_p': float(r['outer_p_hat']),
                'outer_lo': float(r['outer_ci_lower']),
                'outer_hi': float(r['outer_ci_upper']),
                'joint_inner_lo': float(r['joint_inner_ci_lower']),
                'joint_inner_hi': float(r['joint_inner_ci_upper']),
                'joint_outer_lo': float(r['joint_outer_ci_lower']),
                'joint_outer_hi': float(r['joint_outer_ci_upper']),
                'v': r['strict_verdict']})
    rows.sort(key=lambda r: r['cost'])
    return rows


def main():
    rows = load_audit(CSV_AUDIT)
    if not rows:
        raise SystemExit(f'缺 {CSV_AUDIT}，先跑严格混合实体复核生成结果')

    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    y = list(range(len(rows) - 1, -1, -1))

    ax.plot([], [], color=C_INNER, lw=3.0, label='内界模型（导通概率下界）')
    ax.plot([], [], color=C_OUTER, lw=3.0, label='外界模型（导通概率上界）')
    for yi, r in zip(y, rows):

        ax.plot([r['inner_lo'], r['inner_hi']], [yi, yi], color=C_INNER,
                lw=3.2, alpha=0.9, solid_capstyle='round')
        ax.plot([r['outer_lo'], r['outer_hi']], [yi, yi], color=C_OUTER,
                lw=3.2, alpha=0.9, solid_capstyle='round')

        ax.plot([r['inner_p']], [yi], marker='D', ms=6, mfc=C_INNER,
                mec='#111111', mew=0.6, ls='none')
        ax.plot([r['outer_p']], [yi], marker='o', ms=6, mfc=C_OUTER,
                mec='#111111', mew=0.6, ls='none')

        ax.text(r['outer_hi'] + 0.0015, yi,
                f"({r['na']},{r['nb']})  {r['cost']:.3f} 元", va='center',
                fontsize=8.5, color=VCOLOR[r['v']])

    for r in rows:
        if r['na'] == 598 and r['nb'] == 62:
            yi = y[rows.index(r)]
            ax.annotate(f'联合下界 {r["joint_inner_lo"]:.4f} < 0.90，可靠不足',
                        xy=(r['outer_hi'], yi), xytext=(r['outer_hi'] + 0.004,
                        yi + 1.15), fontsize=8.5, color=C_INSUF,
                        arrowprops=dict(arrowstyle='->', color=C_INSUF, lw=0.9))
    ax.axvline(P_TARGET, color='#111111', ls='--', lw=1.5,
               label='目标概率 $P=0.90$')
    ax.set_xlabel('导通概率（共同随机数 $M=4000$ 严格复核）')
    ax.set_yticks([])
    ax.set_ylim(-1.4, len(rows) + 0.4)
    ax.legend(loc='lower right', fontsize=8.5, ncol=2)
    ax.grid(axis='x', alpha=0.25)
    fig.text(0.01, 0.01,
             'A 为 32 边内接/外切多棱柱，B 为三级细分二十面体内接/外切多面体；'
             'B 单侧径向误差上界约 0.9098 nm；4000 次试验上下界违例为 0。'
             '纯 A 保守推荐 (619,0)（引用 Q3 M=10000 严格实体复核）。',
             fontsize=8, color='#444444')
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)
    stem = '问题4_严格实体复核夹逼'
    base = os.path.join(FIG_DIR, stem)
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
        shutil.copy2(base + '.' + ext,
                     os.path.join(APPENDIX_FIG_DIR, stem + '.' + ext))
    plt.close(fig)
    print(f'图4 严格实体复核：{len(rows)} 候选已保存并入池（dpi=300）')


if __name__ == '__main__':
    main()
