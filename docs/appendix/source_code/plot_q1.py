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
问题1 成果图（图3、图4、图5）。

图3：任务分配与路径规划图（2×2，每算例一子图）——各无人机自基地 (0,0)
     出发、巡检、返回的完整航迹，按无人机编号分色，巡检点按等级着色。
图4：各无人机工作时长偏离均值条形图（2×2）——各机时长在绝对轴上差异
     不可见（全算例极差 ≤ 27.8 min），改为分钟尺度偏离均值条形，
     0 线=均值、蓝条=低于均值、橙条=高于均值，直观呈现 min-max 目标
     下的负载均衡性。
图5：求解收敛曲线（2×2）——best-so-far 最优值随搜索时间下降的单调包络
     （原始日志含被拒绝移动的高频抖动，按时间分桶取桶内最优值平滑）。

数据来源：
- 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].point_seq / work_s / work_h）
- 附件1 附件1.xlsx（坐标与巡检等级，1 单位 = 100 m）
- 收敛日志 Q1/logs/convergence_Case*.csv（elapsed_s, objective）
输出：Q1/figs/fig3_*.png/.pdf、fig4_*.png/.pdf、fig5_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q1/plot_q1.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
Q1_DIR = REPO_ROOT / "Q1"
FIG_DIR = Q1_DIR / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
BASE = (0.0, 0.0)  # 飞行基地坐标（单位：100 m）

LEVEL_STYLE = {
    "I":   ("#C0392B", 3, "I级（3次）"),
    "II":  ("#F39C12", 2, "II级（2次）"),
    "III": ("#2980B9", 1, "III级（1次）"),
}
# 无人机航迹配色（tab20 循环，N 最大 5 架）
UAV_CMAP = plt.get_cmap("tab20")


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


def load_points(case: str) -> dict[int, dict]:
    """附件1 某算例 → {pid: {x, y, level}}。"""
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
    """在子图上绘制巡检点（按等级着色）。"""
    for lvl, (color, _, _) in LEVEL_STYLE.items():
        xs = [p["x"] for p in pts.values() if p["level"] == lvl]
        ys = [p["y"] for p in pts.values() if p["level"] == lvl]
        ax.scatter(xs, ys, s=22, c=color, alpha=0.8, edgecolors="white",
                   linewidths=0.3, zorder=3)
    ax.plot(*BASE, marker="*", markersize=13, color="gold",
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)


def fig3_routes(archives: dict[str, dict], pts_all: dict[str, dict]) -> plt.Figure:
    """图3：任务分配与路径规划图（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        pts = pts_all[case]
        plot_route_points(ax, pts)
        for i, u in enumerate(arc["uavs"]):
            seq = [BASE] + [(pts[pid]["x"], pts[pid]["y"]) for pid in u["point_seq"]] + [BASE]
            xs, ys = zip(*seq)
            ax.plot(xs, ys, "-", color=UAV_CMAP(2 * i % 20), linewidth=1.4,
                    alpha=0.85, zorder=2, label=f"无人机{i + 1}")
        ax.text(0.985, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="right", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=10)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


def fig4_workload(archives: dict[str, dict]) -> plt.Figure:
    """图4：各无人机工作时长偏离均值条形图（2×2，分钟尺度）。

    问题1 各机工作时长高度接近（全算例极差 ≤ 27.8 min），在绝对时长轴
    （8~9 h）上条形差异几乎不可见。改为画"偏离均值"：0 线=该算例各机
    平均工作时长，蓝条=低于均值、橙条=高于均值，x 轴单位切换为 min，
    差异放大到直接可读——直观呈现 min-max 目标下的负载均衡性。
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        uavs = arc["uavs"]
        n = len(uavs)
        ids = [u["uav_id"] for u in uavs]
        work_h = [u["work_h"] for u in uavs]
        mean = sum(work_h) / n
        dev = [(w - mean) * 60.0 for w in work_h]  # 偏离均值（min）
        y = np.arange(n)[::-1]  # 从上到下：无人机1 → 无人机n

        ax.axvline(0, color="#333333", linewidth=1.0, zorder=2)
        ax.barh(y, dev, height=0.55,
                color=["#DD8452" if d >= 0 else "#4C72B0" for d in dev],
                edgecolor="white", zorder=3)
        for yi, d in zip(y, dev):
            off = 1.2 if d >= 0 else -1.2
            ax.text(d + off, yi, f"{d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9,
                    color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([f"无人机{i}" for i in ids])
        ax.set_xlabel("偏离均值（min，负=低于均值）", fontsize=10)
        # 轴范围给条端标注留出空间（Case2 条长仅 ±0.7 min，固定偏移 1.2 须在轴内）
        lim = (max(abs(d) for d in dev) + 1.6) * 1.2
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    fig.tight_layout()
    return fig


def fig5_convergence() -> plt.Figure:
    """图5：求解收敛曲线（2×2，best-so-far 包络 + 尝试区间带 + 改进点）。

    原始收敛日志记录每次搜索尝试的目标值（含被拒绝的移动），直接绘制会
    呈现高频抖动噪声带。此处：
    - 灰带：按时间分桶的"尝试区间"（桶内 min~max），展示搜索尝试范围；
    - 蓝线：running-minimum（best-so-far）阶梯线，每级台阶即一次真实改进；
    - 红点：改进发生的位置与幅度。
    图内文字注明改进次数——Case1/3/4 热启动初始解即最优（0 次改进），
    Case2 在搜索早期完成 4 次改进，体现初始解质量与求解器验证作用。
    """
    NBIN = 60  # 时间分桶数（300 s 搜索 → 每桶 5 s）
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, case in zip(axes.flat, CASES):
        csv_path = Q1_DIR / "logs" / f"convergence_{case}.csv"
        df = pd.read_csv(csv_path)
        t = df["elapsed_s"].values
        obj = df["objective"].values

        # best-so-far（单调不增）与时间分桶
        best = np.minimum.accumulate(obj)
        edges = np.linspace(t[0], t[-1] + 1e-9, NBIN + 1)
        bin_idx = np.clip(np.searchsorted(edges, t) - 1, 0, NBIN - 1)
        bin_min = np.full(NBIN, np.inf)
        np.minimum.at(bin_min, bin_idx, obj)
        bin_max = np.full(NBIN, -np.inf)
        np.maximum.at(bin_max, bin_idx, obj)
        bin_best = np.full(NBIN, np.inf)
        np.minimum.at(bin_best, bin_idx, best)
        # 空桶前向填充
        bin_min = pd.Series(np.where(np.isfinite(bin_min), bin_min, np.nan)).ffill().to_numpy()
        bin_max = pd.Series(np.where(np.isfinite(bin_max), bin_max, np.nan)).ffill().to_numpy()
        bin_best = pd.Series(np.where(np.isfinite(bin_best), bin_best, np.nan)).ffill().to_numpy()
        bin_t = (edges[:-1] + edges[1:]) / 2.0

        # 1) 尝试区间带（浅灰，透明）
        ax.fill_between(bin_t / 60.0, bin_min / 3600.0, bin_max / 3600.0,
                        step="post", color="#999999", alpha=0.22,
                        linewidth=0, zorder=1)
        # 2) best-so-far 阶梯线
        ax.plot(bin_t / 60.0, bin_best / 3600.0, drawstyle="steps-post",
                color="#4C72B0", linewidth=1.8, zorder=3)
        # 3) 改进点（桶级 best 下降处）
        chg = np.diff(bin_best) < 0
        chg_t, chg_v = bin_t[:-1][chg], bin_best[:-1][chg]
        n_imp = int(chg.sum())
        if n_imp > 0:
            ax.scatter(chg_t / 60.0, chg_v / 3600.0, s=34, marker="o",
                       color="#C0392B", zorder=4, edgecolors="white",
                       linewidths=0.5)
        final = bin_best[-1] / 3600.0
        ax.axhline(final, color="#C0392B", linestyle="--", linewidth=1.0,
                   label=f"最终目标 {final:.2f} h")
        # 改进次数说明（左下角）
        note = ("搜索期间无改进，初始解即最优" if n_imp == 0
                else f"搜索期间 {n_imp} 次改进（末次 @ {chg_t[-1]:.0f} s）")
        ax.text(0.015, 0.90, note, transform=ax.transAxes, fontsize=9,
                color="#555555", va="top", ha="left", zorder=6)
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_xlabel("求解时间（min）", fontsize=10)
        ax.set_ylabel("目标值（h）", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
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
