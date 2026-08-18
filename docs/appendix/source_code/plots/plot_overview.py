# -*- coding: utf-8 -*-
# ============================================================
# 本绘图程序在AI工具辅助下完成
# AI工具名称：DeepSeek-V4-Flash（Claude Code）
# 版本：v4-flash（2026-08 使用）
# 开发机构：杭州深度求索人工智能基础技术研究有限公司
# 使用日期：2026-08-18
# 人工修改记录：见 docs/appendix/interaction_logs/edit_trace.md
# ============================================================
"""
通用数据概况图（图1、图2）：巡检点空间分布图 + 巡检等级构成图。

图1：四个算例（Case1~4）的巡检点空间分布，散点按巡检等级着色
     （I级=3次巡检 / II级=2次 / III级=1次），飞行基地 (0,0) 以五角星标注。
图2：四个算例的巡检等级构成堆叠柱状图，标注各等级点数与展开任务数。

数据来源：attachment/附件1.xlsx（坐标单位：1 单位 = 100 m）。
输出：plots/figs/fig1_巡检点分布.png/.pdf、fig2_巡检等级构成.png/.pdf

用法（仓库根目录，math 环境）：
    python plots/plot_overview.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式（与 templates/common/plot_style.py 同名接口）。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")

ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
FIG_DIR = REPO_ROOT / "plots" / "figs"
CASES = ["Case1", "Case2", "Case3", "Case4"]

# 巡检等级 → (颜色, 所需巡检次数, 中文名)
LEVEL_STYLE = {
    "I":   ("#C0392B", 3, "I级（3次）"),
    "II":  ("#F39C12", 2, "II级（2次）"),
    "III": ("#2980B9", 1, "III级（1次）"),
}


def load_all_cases() -> dict[str, pd.DataFrame]:
    """读取附件1 四个算例，返回 {case: DataFrame}。"""
    return {c: pd.read_excel(ATTACH_XLSX, sheet_name=c) for c in CASES}


def fig1_distribution(cases: dict[str, pd.DataFrame]) -> plt.Figure:
    """图1：巡检点空间分布图（2×2 子图）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, case in zip(axes.flat, CASES):
        df = cases[case]
        for lvl, (color, _, label) in LEVEL_STYLE.items():
            sub = df[df["Inspection_Level"] == lvl]
            ax.scatter(sub["X_Coordinate"], sub["Y_Coordinate"],
                       s=28, c=color, alpha=0.85, edgecolors="white",
                       linewidths=0.4, label=label, zorder=3)
        # 飞行基地 (0,0)
        ax.plot(0, 0, marker="*", markersize=15, color="gold",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5,
                label="飞行基地 (0,0)")
        n = len(df)
        ax.set_title(f"{case}（{n} 个巡检点）", fontsize=13)
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=11)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    # 共享图例（取最后一个子图的）
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("各算例巡检点空间分布（I/II/III 级所需巡检次数为 3/2/1 次）",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    return fig


def fig2_level_composition(cases: dict[str, pd.DataFrame]) -> plt.Figure:
    """图2：巡检等级构成堆叠柱状图。"""
    levels = list(LEVEL_STYLE.keys())
    counts = np.zeros((len(CASES), len(levels)), dtype=int)
    n_tasks = np.zeros(len(CASES), dtype=int)
    for i, case in enumerate(CASES):
        vc = cases[case]["Inspection_Level"].value_counts()
        for j, lvl in enumerate(levels):
            counts[i, j] = int(vc.get(lvl, 0))
        n_tasks[i] = int((counts[i] * [3, 2, 1]).sum())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CASES))
    bottom = np.zeros(len(CASES))
    colors = [LEVEL_STYLE[l][0] for l in levels]
    for j, lvl in enumerate(levels):
        ax.bar(x, counts[:, j], bottom=bottom, color=colors[j], width=0.55,
               edgecolor="white", label=LEVEL_STYLE[lvl][2])
        # 柱内数值标注
        for i in range(len(CASES)):
            if counts[i, j] > 0:
                ax.text(x[i], bottom[i] + counts[i, j] / 2, str(counts[i, j]),
                        ha="center", va="center", color="white", fontsize=10,
                        fontweight="bold")
        bottom += counts[:, j]
    # 展开任务数标注（柱顶）
    for i in range(len(CASES)):
        ax.text(x[i], bottom[i] + 2, f"任务数 {n_tasks[i]}", ha="center",
                va="bottom", fontsize=10, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    ax.set_ylabel("巡检点数量", fontsize=12)
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.set_title("各算例巡检等级构成与展开任务数（I/II/III 级对应 3/2/1 次巡检）",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_all_cases()

    for fn, name in [(fig1_distribution, "fig1_巡检点空间分布"),
                     (fig2_level_composition, "fig2_巡检等级构成")]:
        fig = fn(cases)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
