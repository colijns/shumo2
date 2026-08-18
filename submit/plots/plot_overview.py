import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
def save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
FIG_DIR = REPO_ROOT / "plots" / "figs"
CASES = ["Case1", "Case2", "Case3", "Case4"]
LEVEL_STYLE = {
    "I":   ("#C0392B", 3, "I级（3次）"),
    "II":  ("#F39C12", 2, "II级（2次）"),
    "III": ("#2980B9", 1, "III级（1次）"),
}
def load_all_cases() -> dict[str, pd.DataFrame]:
    return {c: pd.read_excel(ATTACH_XLSX, sheet_name=c) for c in CASES}
def fig1_distribution(cases: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, case in zip(axes.flat, CASES):
        df = cases[case]
        for lvl, (color, _, label) in LEVEL_STYLE.items():
            sub = df[df["Inspection_Level"] == lvl]
            ax.scatter(sub["X_Coordinate"], sub["Y_Coordinate"],
                       s=28, c=color, alpha=0.85, edgecolors="white",
                       linewidths=0.4, label=label, zorder=3)
        ax.plot(0, 0, marker="*", markersize=15, color="gold",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5,
                label="飞行基地 (0,0)")
        n = len(df)
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=11)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    return fig
def fig2_level_composition(cases: dict[str, pd.DataFrame]) -> plt.Figure:
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
        for i in range(len(CASES)):
            if counts[i, j] > 0:
                ax.text(x[i], bottom[i] + counts[i, j] / 2, str(counts[i, j]),
                        ha="center", va="center", color="white", fontsize=10,
                        fontweight="bold")
        bottom += counts[:, j]
    for i in range(len(CASES)):
        ax.text(x[i], bottom[i] + 2, f"任务数 {n_tasks[i]}", ha="center",
                va="bottom", fontsize=10, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    ax.set_ylabel("巡检点数量", fontsize=12)
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.legend(loc="upper left", fontsize=11)
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
