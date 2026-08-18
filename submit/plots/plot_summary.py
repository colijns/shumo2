
































import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
FIG_DIR = REPO_ROOT / "plots" / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"

Q2_ARCHIVE = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"

Q_COLORS = {"问题1": "#4C72B0", "问题2": "#DD8452", "问题3": "#55A868"}


def save_fig(fig, path, dpi=300):

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def collect() -> dict[str, dict[str, float]]:

    out = {}
    for c in CASES:
        q1 = json.load(open(ARCHIVE / f"q1_solution_{c}.json", encoding="utf-8"))
        q2 = json.load(open(Q2_ARCHIVE / f"{c}.json", encoding="utf-8"))
        q3 = json.load(open(ARCHIVE / "q3" / "strict" / f"{c}.json", encoding="utf-8"))
        out[c] = {
            "问题1": {"N": q1["N"], "Tmax": q1["Tmax_s"] / 3600.0,
                      "delta": (q1["Tmax_s"] - q1["Tmin_s"]) / 3600.0},
            "问题2": {"N": q2["fleet_size"], "Tmax": q2["metrics"]["Tmax_s"] / 3600.0,
                      "delta": q2["metrics"]["delta_s"] / 3600.0},
            "问题3": {"N": q3["fleet_size"], "Tmax": q3["metrics"]["S_max_s"] / 3600.0,
                      "delta": q3["metrics"]["delta_s"] / 3600.0},
        }
    return out


def fig12_summary(data: dict[str, dict[str, dict[str, float]]]) -> plt.Figure:










    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    x = np.arange(len(CASES)) * 1.25
    width = 0.24
    x_lo, x_hi = -0.9, 5.7
    handles = []


    ax = axes[0]
    ax.text(0.008, 0.965, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=7)
    for j, (qname, color) in enumerate(Q_COLORS.items()):
        vals = [data[c][qname]["Tmax"] for c in CASES]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color,
                      edgecolor="white", label=qname, zorder=3)
        handles.append(bars)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, zorder=6)
    q3_t = [data[c]["问题3"]["Tmax"] for c in CASES]
    q2_t = [data[c]["问题2"]["Tmax"] for c in CASES]
    ymax_t = max(q3_t) * 1.18
    ax.set_ylim(0, ymax_t)

    ax.axhline(9, color="black", ls="--", lw=1.2, zorder=1)
    ax.text(x[-1] + width + 0.02, 9, "原 9 小时时限（问题1 约束）", ha="left",
            va="center", fontsize=9, color="black", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))

    for i in range(len(CASES)):
        pct = (q3_t[i] - q2_t[i]) / q2_t[i] * 100
        xc = x[i] + width
        ax.text(xc, q3_t[i] + 0.075 * ymax_t, f"{pct:+.1f}%$^{1}$",
                ha="center", va="bottom", fontsize=9, color="#C0392B", zorder=6)
    ax.set_ylabel("最长完成时间 (h)", fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)


    ax = axes[1]
    ax.text(0.008, 0.965, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=7)
    for j, (qname, color) in enumerate(Q_COLORS.items()):
        vals = [data[c][qname]["delta"] * 60 for c in CASES]
        bars = ax.bar(x + (j - 1) * width, vals, width, color=color,
                      edgecolor="white", zorder=3)
        for b, v in zip(bars, vals):
            if v < 0.05:
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, zorder=6)
    d1 = [data[c]["问题1"]["delta"] * 60 for c in CASES]
    d2 = [data[c]["问题2"]["delta"] * 60 for c in CASES]
    d3 = [data[c]["问题3"]["delta"] * 60 for c in CASES]
    ymax_d = max(d1) * 1.18
    ax.set_ylim(0, ymax_d)

    for i in range(len(CASES)):
        for j, dv in ((0, d2[i]), (1, d3[i])):
            xpos = x[i] + j * width
            if dv < 0.05:
                ax.text(xpos, 0.09 * ymax_d, "★ 完全均衡（δ=0）$^{2}$",
                        ha="center", va="bottom", fontsize=9.5,
                        color="#D4A017", fontweight="bold", zorder=6)
            else:
                pct = (d1[i] - dv) / d1[i] * 100
                xr = xpos + width / 2 + 0.01
                ax.text(xr, dv + 0.01 * ymax_d, f"-{pct:.0f}%$^{2}$", ha="left",
                        va="bottom", fontsize=9, color="#2E7D32", zorder=6)
    ax.set_ylabel("工作时长极差 δ (min)", fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)

    fig.text(0.5, 0.008,
             "$^{1}$ 增幅以问题 2 方案为基准；$^{2}$ 降幅以问题 1 方案为基准"
             "（Case2 问题3 完全均衡，降幅为 -100%）",
             ha="center", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.legend(handles, [h.get_label() for h in handles], loc="upper right",
               fontsize=9, frameon=True)
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig = fig12_summary(data)
    for ext in ("png", "pdf"):
        save_fig(fig, FIG_DIR / f"fig12_三问题结果汇总.{ext}", dpi=300)
    plt.close(fig)
    print("完成：fig12_三问题结果汇总")


if __name__ == "__main__":
    main()
