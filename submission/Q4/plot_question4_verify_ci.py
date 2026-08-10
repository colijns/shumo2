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
CSV_VERIFY = os.path.join(_BASE, 'results', 'question4_verify.csv')
FIG_DIR = os.path.join(_BASE, 'figures')
APPENDIX_FIG_DIR = os.path.join(
    os.path.dirname(_BASE), 'docs', 'appendix', 'figures')

P_TARGET = 0.90
COLOR = {'reliable': '#55A868', 'crossing': '#4C72B0'}
LABEL = {'reliable': '可靠可行（CI 下界 ≥ 0.90）',
         'crossing': '跨线（CI 包含 0.90）'}


def load_verify(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'na': int(r['N_A']), 'nb': int(r['N_B']),
                'cost': float(r['cost_元']),
                'm': int(r['m']), 'p': float(r['p_hat']),
                'lo': float(r['ci_lower']), 'hi': float(r['ci_upper']),
                'v': r['verdict']})
    rows.sort(key=lambda r: r['cost'])
    return rows


def main():
    rows = load_verify(CSV_VERIFY)
    if not rows:
        raise SystemExit(f'缺 {CSV_VERIFY}，先跑独立种子复算生成结果')

    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    y = list(range(len(rows) - 1, -1, -1))
    drawn = set()
    for yi, r in zip(y, rows):
        c = COLOR.get(r['v'], '#888888')
        lab = LABEL.get(r['v'])
        if lab in drawn:
            lab = None
        elif lab:
            drawn.add(lab)
        ax.plot([r['lo'], r['hi']], [yi, yi], color=c, lw=3.0, alpha=0.9,
                label=lab, solid_capstyle='round')
        ax.plot([r['p']], [yi], marker='o', ms=6.5, mfc=c, mec='#111111',
                mew=0.8, ls='none')
        ax.text(r['hi'] + 0.0015, yi,
                f"({r['na']},{r['nb']})  {r['cost']:.3f} 元",
                va='center', fontsize=8.5)
    ax.axvline(P_TARGET, color='#C44E52', ls='--', lw=1.5,
               label='目标概率 $P=0.90$')
    ax.set_xlabel('导通概率（独立种子 $M=4000$ 复算）')
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(APPENDIX_FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, '问题4_独立复算置信区间')
    for ext in ('png', 'pdf'):
        fig.savefig(base + '.' + ext, dpi=300, bbox_inches='tight')
        shutil.copy2(base + '.' + ext,
                     os.path.join(APPENDIX_FIG_DIR, '问题4_独立复算置信区间.' + ext))
    plt.close(fig)
    print(f'图3 独立复算 CI：{len(rows)} 候选已保存并入池（dpi=300）')


if __name__ == '__main__':
    main()
