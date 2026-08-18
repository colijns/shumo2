








































import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q2_DIR = REPO_ROOT / "Q2"
FIG_DIR = Q2_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"



ARCHIVE_Q2 = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"

C_Q1 = "#4C72B0"
C_Q2 = "#DD8452"
C_GRAY = "#AAAAAA"
C_TXT = "#333333"


def save_fig(fig, path, dpi=300):

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_q1(case: str) -> dict:
    return json.load(open(ARCHIVE / f"q1_solution_{case}.json", encoding="utf-8"))


def load_q2(case: str) -> dict:
    return json.load(open(ARCHIVE_Q2 / f"{case}.json", encoding="utf-8"))


def fig6_workload_compare(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:









    case = "Case1"
    a1, a2 = q1[case], q2[case]
    w1 = [u["work_s"] / 3600.0 for u in a1["uavs"]]
    w2 = [s / 3600.0 for s in a2["metrics"]["route_work_s"]]
    ids = np.arange(1, len(w1) + 1)

    order = np.argsort(w1)
    ids, w1, w2 = ids[order], [w1[i] for i in order], [w2[i] for i in order]
    n = len(w1)
    y = np.arange(n)[::-1]
    delta_min = [(b - a) * 60.0 for a, b in zip(w1, w2)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) > 1e-9:
            ax.plot([v1, v2], [yi, yi], color="#999999", linewidth=1.4,
                    zorder=1, alpha=0.9)

    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) < 1e-9:
            ax.scatter([v1], [yi], s=80, marker="o", color=C_GRAY, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.text(v1 + 0.012, yi, f"{v1:.4f}", fontsize=8, color="#999999",
                    va="center", ha="left")

    for yi, v1, v2, d in zip(y, w1, w2, delta_min):
        if abs(d) < 1e-9:
            continue
        ax.scatter([v1], [yi], s=90, marker="o", color=C_Q1, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([v2], [yi], s=90, marker="D", color=C_Q2, zorder=3,
                   edgecolors="white", linewidths=0.8)

        ax.text(v1 - 0.012, yi + 0.16, f"{v1:.4f}", fontsize=8.5, color=C_Q1,
                ha="right", va="center")
        ax.text(v2 + 0.012, yi - 0.16, f"{v2:.4f}", fontsize=8.5, color=C_Q2,
                ha="left", va="center")

        ax.text((v1 + v2) / 2.0, yi + 0.38, f"Δ {d:+.1f} min", ha="center",
                fontsize=8.5, color=C_TXT, fontweight="bold")

    tmax1 = a1["Tmax_h"]
    tmax2 = a2["metrics"]["Tmax_s"] / 3600.0
    ax.axvline(tmax1, color=C_Q1, linestyle="--", linewidth=1.2, alpha=0.85,
               zorder=2)
    ax.axvline(tmax2, color=C_Q2, linestyle="--", linewidth=1.2, alpha=0.85,
               zorder=2)

    ax.set_xlim(8.0, 8.9)
    ax.set_xticks(np.arange(8.0, 8.91, 0.1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("工作时长（h）", fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"UAV{i}" for i in ids])
    ax.set_ylabel("无人机（按问题1 工时升序）", fontsize=10.5)
    ax.set_ylim(-0.6, 3.75)
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.012, 0.97, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    q1_h = plt.Line2D([], [], marker="o", linestyle="none", color=C_Q1,
                      markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    q2_h = plt.Line2D([], [], marker="D", linestyle="none", color=C_Q2,
                      markersize=6.5, markeredgecolor="white", markeredgewidth=0.8)
    gray_h = plt.Line2D([], [], marker="o", linestyle="none", color=C_GRAY,
                        markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    t1_h = plt.Line2D([], [], color=C_Q1, linestyle="--", linewidth=1.2)
    t2_h = plt.Line2D([], [], color=C_Q2, linestyle="--", linewidth=1.2)
    fig.legend([q1_h, q2_h, gray_h, t1_h, t2_h],
               ["问题1", "问题2", "未调整（Δ=0）",
                f"T_max(问题1)={tmax1:.2f} h", f"T_max(问题2)={tmax2:.2f} h"],
               loc="upper center", ncol=5, fontsize=8.5, frameon=True,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig7_tradeoff(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:









    case = "Case1"
    a1, a2 = q1[case], q2[case]
    t1, d1 = a1["Tmax_h"], (a1["Tmax_s"] - a1["Tmin_s"]) / 60.0
    t2, d2 = a2["metrics"]["Tmax_s"] / 3600.0, a2["metrics"]["delta_s"] / 60.0
    pct_t = (t2 - t1) / t1 * 100.0
    pct_d = (d2 - d1) / d1 * 100.0

    fig = plt.figure(figsize=(8.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])


    ax1.scatter([d1], [t1], s=220, marker="*", color=C_Q1, zorder=5,
                label=f"问题1（T_max={t1:.2f} h，δ={d1:.1f} min）")
    ax1.scatter([d2], [t2], s=180, marker="D", color=C_Q2, zorder=5,
                label=f"问题2（T_max={t2:.2f} h，δ={d2:.1f} min）")

    ax1.annotate("", xy=(d2, t2), xytext=(d1, t1),
                 arrowprops=dict(arrowstyle="->,head_length=0.7,head_width=0.5",
                                 color="#555555", lw=3.0,
                                 shrinkA=16, shrinkB=16))
    ax1.text((d1 + d2) / 2.0, t1 + 0.055, "帕累托改进", ha="center",
             va="bottom", fontsize=11, fontweight="bold", color="#C0392B",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax1.set_xlabel("负载差 δ（min）", fontsize=11)
    ax1.set_ylabel("总体完成时间 T_max（h）", fontsize=11)
    ax1.set_xlim(0, 30)
    ax1.set_xticks(range(0, 31, 5))
    ax1.set_ylim(t1 - 0.1, t1 + 0.1)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left", labelspacing=1.5)
    ax1.text(0.015, 0.98, case, transform=ax1.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="left", zorder=6,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))


    axb = fig.add_subplot(gs[1])
    yidx = np.arange(2)[::-1]
    vals = [pct_t, pct_d]
    names = ["T_max（h）", "δ（min）"]
    colors = [C_Q1, C_Q2]
    axb.barh(yidx, vals, height=0.5, color=colors, edgecolor="white", zorder=3)
    axb.axvline(0, color="#333333", linewidth=1.5, zorder=4)
    for yi, v, nm in zip(yidx, vals, names):
        if abs(v) < 1e-9:

            axb.text(1.5, yi, f"{nm}: +0.0%（零代价）", va="center",
                     fontsize=10, color=C_TXT)
        else:
            axb.text(v - 2.0, yi, f"{v:+.1f}%", ha="right", va="center",
                     fontsize=11, fontweight="bold", color=colors[1])
    axb.set_yticks(yidx)
    axb.set_yticklabels(names)
    axb.set_xlim(-75, 20)
    axb.set_xticks([-75, -50, -25, 0, 25])
    axb.set_xlabel("相对变化率（%）（负值=下降，以问题1 为基准）", fontsize=10)
    axb.set_ylim(-0.6, 1.6)
    axb.grid(axis="x", alpha=0.3)

    fig.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.05, hspace=0.5)
    return fig


def fig8_cross_case_delta(q1: dict[str, dict], q2: dict[str, dict]) -> plt.Figure:







    d1 = [(q1[c]["Tmax_s"] - q1[c]["Tmin_s"]) / 60.0 for c in CASES]
    d2 = [q2[c]["metrics"]["delta_s"] / 60.0 for c in CASES]
    pct = [(b - a) / a * 100.0 for a, b in zip(d1, d2)]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(CASES))
    width = 0.35
    ax.bar(x - width / 2, d1, width, color=C_Q1, edgecolor="white",
           label="问题1", zorder=3)
    ax.bar(x + width / 2, d2, width, color=C_Q2, edgecolor="white",
           label="问题2", zorder=3)

    for xi, a, b, p in zip(x, d1, d2, pct):
        top = max(a, b)
        ax.text(xi, top + 0.6, f"{p:+.1f}%", ha="center", fontsize=11,
                fontweight="bold", color="#C0392B", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(CASES)
    ax.set_xlabel("测试算例", fontsize=11)
    ax.set_ylabel("负载差 δ（min）", fontsize=11)
    ax.set_ylim(0, max(d1) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    q1 = {c: load_q1(c) for c in CASES}
    q2 = {c: load_q2(c) for c in CASES}

    for fn, name in [(fig6_workload_compare, "fig6_问题1vs2工作时长对比"),
                     (fig7_tradeoff, "fig7_问题2双目标权衡"),
                     (fig8_cross_case_delta, "fig8_跨算例δ均衡效果对比")]:
        fig = fn(q1, q2)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
