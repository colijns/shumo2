



















































import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


plt.rcParams["font.family"] = ["Arial", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "axes.labelsize": 9.5, "axes.titlesize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 100,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q1_DIR = REPO_ROOT / "Q1"
FIG_DIR = Q1_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
BASE = (0.0, 0.0)
KM_UNIT = 0.1



LEVEL_STYLE = {
    "I":   ("#E69F00", 30, True, "I级（实心圆）"),
    "II":  ("#0072B2", 30, False, "II级（空心圆）"),
    "III": ("#CC79A7", 14, True, "III级（小圆点）"),
}

UAV_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GRAY = "#999999"
C_RED = "#C0392B"


def save_fig(fig, path, dpi=300):

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_points(case: str) -> dict[int, dict]:

    df = pd.read_excel(ATTACH_XLSX, sheet_name=case)
    pts = {}
    for row in df.itertuples(index=False):
        pts[int(row.Point_ID)] = {
            "x": float(row.X_Coordinate),
            "y": float(row.Y_Coordinate),
            "level": str(row.Inspection_Level).strip().upper(),
        }
    return pts


def load_archives() -> dict[str, dict]:
    return {c: json.load(open(ARCHIVE / f"q1_solution_{c}.json", encoding="utf-8"))
            for c in CASES}


def plot_route_points(ax, pts: dict[int, dict]):





    for lvl, (color, s, filled, _) in LEVEL_STYLE.items():
        xs = [p["x"] for p in pts.values() if p["level"] == lvl]
        ys = [p["y"] for p in pts.values() if p["level"] == lvl]
        if filled:
            ax.scatter(xs, ys, s=s, marker="o", c=color, alpha=0.85,
                       edgecolors="none", zorder=3)
        else:
            ax.scatter(xs, ys, s=s, marker="o", facecolors="none",
                       edgecolors=color, linewidths=0.9, zorder=3)
    ax.plot(*BASE, marker="*", markersize=16, color="black",
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)


def _scale_bar(ax, fontsize=8):





    x0, x1 = ax.get_xlim()
    target_km = (x1 - x0) * KM_UNIT / 4.0
    mag = 10 ** math.floor(math.log10(target_km))
    km = 0.0
    for m in (1, 2, 5, 10):
        if m * mag >= target_km:
            km = m * mag
            break
    L = km / KM_UNIT
    y0, y1 = ax.get_ylim()
    bx = x1 - L * 0.70
    by = y0 + (y1 - y0) * 0.055
    ax.plot([bx, bx + L], [by, by], color="black", linewidth=2.0,
            solid_capstyle="butt", zorder=6)
    tick = L * 0.05
    ax.plot([bx, bx], [by - tick, by + tick], color="black", linewidth=1.2, zorder=6)
    ax.plot([bx + L, bx + L], [by - tick, by + tick], color="black",
            linewidth=1.2, zorder=6)
    ax.text(bx + L / 2.0, by + tick * 2.2, f"{km:.0f} km",
            ha="center", va="bottom", fontsize=fontsize, zorder=6)


def fig3_routes(archives: dict[str, dict], pts_all: dict[str, dict]) -> plt.Figure:






    fig, axes = plt.subplots(2, 2, figsize=(10, 8.8))
    labels = "abcd"

    uav_handles: dict[int, object] = {}
    for ax, case, lab in zip(axes.flat, CASES, labels):
        arc, pts = archives[case], pts_all[case]
        plot_route_points(ax, pts)
        n = len(arc["uavs"])
        for i, u in enumerate(arc["uavs"]):
            seq = [BASE] + [(pts[pid]["x"], pts[pid]["y"]) for pid in u["point_seq"]] + [BASE]
            xs, ys = zip(*seq)
            (ln,) = ax.plot(xs, ys, "-", color=UAV_COLORS[i % len(UAV_COLORS)],
                            linewidth=0.9, alpha=0.7, zorder=2)
            uav_handles.setdefault(i, ln)

        ax.text(0.015, 0.97, f"({lab}) {case}  N={n}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=8.5)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=8.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        _scale_bar(ax)

    lvl_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                   markerfacecolor=c if filled else "none",
                   markeredgecolor=c, markeredgewidth=0.9,
                   alpha=0.85 if filled else 1.0)
        for c, _, filled, _ in LEVEL_STYLE.values()]
    base_handle = plt.Line2D([], [], marker="*", linestyle="none",
                             markersize=11, color="black",
                             markeredgecolor="white", markeredgewidth=0.6)
    handles = lvl_handles + [base_handle]
    handles += [uav_handles[k] for k in sorted(uav_handles)]
    labels_leg = ([f"{lvl}级（{desc}）" for lvl, desc in
                   zip(LEVEL_STYLE.keys(), [s[3] for s in LEVEL_STYLE.values()])]
                  + ["基地（0,0）"]
                  + [f"无人机{k + 1}" for k in sorted(uav_handles)])
    fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    return fig


def fig4_workload(archives: dict[str, dict]) -> plt.Figure:







    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        uavs = arc["uavs"]
        n = len(uavs)
        ids = [u["uav_id"] for u in uavs]
        work_h = [u["work_h"] for u in uavs]
        mean = sum(work_h) / n
        dev = [(w - mean) * 60.0 for w in work_h]
        y = np.arange(n)[::-1]

        ax.axvline(0, color="black", linewidth=1.6, zorder=2)
        ax.barh(y, dev, height=0.55,
                color=[C_ORANGE if d >= 0 else C_BLUE for d in dev],
                edgecolor="white", zorder=3)
        for yi, d in zip(y, dev):
            off = 1.2 if d >= 0 else -1.2
            ax.text(d + off, yi, f"{d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8.5,
                    color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([f"无人机{i}" for i in ids])
        ax.set_xlabel("偏离均值（min，负=低于均值）", fontsize=8.5)

        lim = (max(abs(d) for d in dev) + 1.6) * 1.2
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", alpha=0.3)

        delta_min = (arc["Tmax_s"] - arc["Tmin_s"]) / 60.0
        ax.text(0.97, 0.97,
                f"δ = T_max − T_min = {delta_min:.1f} min\n均值 = {mean:.2f} h",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                zorder=6, linespacing=1.5,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        ax.text(0.015, 0.97, case, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    fig.tight_layout()
    return fig


def fig5_convergence() -> plt.Figure:















    NBIN = 60
    YL_MARGIN = 0.03
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.0))
    for ax, case in zip(axes.flat, CASES):
        csv_path = Q1_DIR / "logs" / f"convergence_{case}.csv"
        df = pd.read_csv(csv_path)
        t = df["elapsed_s"].values
        obj = df["objective"].values


        best = np.minimum.accumulate(obj)
        edges = np.linspace(t[0], t[-1] + 1e-9, NBIN + 1)
        bin_idx = np.clip(np.searchsorted(edges, t) - 1, 0, NBIN - 1)
        bin_best = np.full(NBIN, np.inf)
        np.minimum.at(bin_best, bin_idx, best)
        bin_max = np.full(NBIN, -np.inf)
        np.maximum.at(bin_max, bin_idx, obj)

        bin_best = pd.Series(np.where(np.isfinite(bin_best), bin_best, np.nan)).ffill().to_numpy()
        bin_max = pd.Series(np.where(np.isfinite(bin_max), bin_max, np.nan)).ffill().to_numpy()
        bin_t = (edges[:-1] + edges[1:]) / 2.0



        ax.fill_between(bin_t, bin_best / 3600.0, bin_max / 3600.0,
                        step="post", color="#BBBBBB", alpha=0.15,
                        linewidth=0, zorder=1)


        ax.plot(bin_t, bin_best / 3600.0, drawstyle="steps-post",
                color=C_BLUE, linewidth=1.8, zorder=3)



        chg_idx = np.flatnonzero(np.diff(bin_best) < 0)
        chg_t = bin_t[chg_idx]
        chg_v = bin_best[chg_idx + 1]
        n_imp = len(chg_idx)
        if n_imp > 0:
            ax.scatter(chg_t, chg_v / 3600.0, s=32, marker="o",
                       color=C_RED, zorder=4, edgecolors="white",
                       linewidths=0.5)
            for i0, (ct, cv) in enumerate(zip(chg_t, chg_v)):
                imp = bin_best[chg_idx[i0]] - cv
                ax.annotate(f"−{imp / 3600.0:.2f} h",
                            xy=(ct, cv / 3600.0),
                            xytext=(ct + 8, cv / 3600.0 + 0.10),
                            fontsize=8, color=C_RED, fontweight="bold",
                            zorder=5,
                            arrowprops=dict(arrowstyle="->", lw=0.7,
                                            color=C_RED, shrinkA=2, shrinkB=2))



        final = bin_best[-1] / 3600.0
        ax.axhline(final, color=C_RED, linestyle="--", linewidth=1.2,
                   alpha=0.9, zorder=4)
        ax.axvline(300, color=C_GRAY, linestyle="--", linewidth=1.0,
                   alpha=0.8, zorder=2)

        ax.text(0.985, 0.97, f"改进: {n_imp}次 | 最优值: {final:.4f} h",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color="#333333", zorder=6, linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.text(0.015, 0.97, case, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

        ax.set_ylim(final * (1 - YL_MARGIN), final * (1 + YL_MARGIN))
        ax.set_xlim(0, 305)
        ax.set_xticks(np.arange(0, 301, 60))
        ax.set_xlabel("搜索时间（s）", fontsize=8.5)
        ax.set_ylabel("目标值 f（h）", fontsize=8.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    best_h = plt.Line2D([], [], color=C_BLUE, linewidth=1.8)
    band_h = plt.Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", alpha=0.4)
    imp_h = plt.Line2D([], [], marker="o", linestyle="none", color=C_RED,
                       markersize=6, markeredgecolor="white", markeredgewidth=0.5)
    line_h = plt.Line2D([], [], color=C_GRAY, linestyle="--", linewidth=1.0)
    fig.legend([best_h, band_h, imp_h, line_h],
               ["best-so-far 最优值包络", "搜索尝试区间", "改进点",
                "阶段二搜索时长 300 s"],
               loc="upper center", ncol=4, fontsize=8.5, frameon=True,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    archives = load_archives()
    pts_all = {c: load_points(c) for c in CASES}

    for fn, name in [
        (fig3_routes, "fig3_问题1路径规划方案"),
        (fig4_workload, "fig4_问题1各机工作时长"),
        (fig5_convergence, "fig5_问题1求解收敛曲线"),
    ]:
        args = (archives, pts_all) if name.startswith("fig3") else (archives,) if name.startswith("fig4") else ()
        fig = fn(*args)
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
