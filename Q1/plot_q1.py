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
问题1 成果图（图3、图4、图5）——2026-08-18 重构版。

图3：任务分配与路径规划图（2×2，每算例一子图）——各无人机自基地 (0,0)
     出发、巡检、返回的完整闭合航迹（不省略首尾段），按无人机编号分色
     （seaborn muted 低饱和 5 色，色盲友好），线宽 0.9、透明度 0.7；
     巡检点用"颜色 + 形状"双维度区分等级（I 级橙实心圆 / II 级蓝空心圆 /
     III 级粉紫小圆点），色盲与打印场景均可辨识；基地为黑色实心大五角星；
     子图左上角标注 (a)~(d) 与 N=X（无人机数量，与表 2 呼应）；
     每子图右下角附线段比例尺（km，按子图尺度取 1/2/5×10^k 整值）；
     图例整图底部两组（巡检点等级 / 无人机航迹 + 基地），图内无标题。
     四个算例空间尺度差异极大（Case2 仅十几公里，Case1 达上百公里），
     各子图自适应坐标、不强制统一比例。

图4：各无人机工作时长偏离均值条形图（2×2）——各机时长在绝对轴上差异
     不可见（全算例极差 ≤ 27.8 min），改为分钟尺度偏离均值条形：
     0 线=该算例平均时长（黑色粗实线）、蓝条=低于均值、橙条=高于均值，
     条端标注偏差（min，1 位小数）；子图右上角补充量化值 δ = T_max − T_min
     （min）与均值（h），与论文表 2、表 3 指标直接对应；各子图横轴尺度
     独立（Case2 极差仅 1.4 min，统一尺度会失去"放大可读"的意义）。

图5：求解收敛曲线（2×2，best-so-far 包络）——固定 N 下第二阶段优化的
     单调不增阶梯线（禁止画成平滑/波动曲线）：灰色带 = 每轮搜索候选解
     的取值区间（best ~ 桶内最差尝试，始终位于 best-so-far 上方）；
     改进点实心圆 + 幅度标注（如 −0.02 h）；0 改进算例标注
     "改进次数：0 | 初始解即最优"；横轴搜索时间（s），300 s 处虚线标注
     阶段二搜索时长终止线。
     【语义诚实性】日志记录的 objective = routing.CostVar().Max()，为
     求解器目标 f = Σt_k + T_max（总工时 + 最大单机工时；已验证与档案
     sum(work_s)+Tmax_s 精确一致），并非 T_max 本身——纵轴标注"目标值
     f（h）"并保留 4 位小数，论文图注须写明 f 定义，不可伪标为 T_max。
     另：收敛日志仅覆盖 0~300 s（无 1800 s 长时搜索日志），故不延伸曲线。

数据来源：
- 解档案 outputs/workbooks/q1_solution_Case*.json（uavs[].point_seq / work_s / work_h）
- 附件1 附件1.xlsx（坐标与巡检等级，1 单位 = 100 m = 0.1 km）
- 收敛日志 Q1/logs/convergence_Case*.csv（elapsed_s, objective）
输出：Q1/figs/fig3_*.png/.pdf、fig4_*.png/.pdf、fig5_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q1/plot_q1.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
# 顶层 family 列表实现逐字形回退（sans-serif 子列表不触发回退）：
# 数字/英文用 Arial，中文字形逐字形回退 SimHei → Microsoft YaHei
plt.rcParams["font.family"] = ["Arial", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# 统一字号规范（8~10 号）
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
BASE = (0.0, 0.0)  # 飞行基地坐标（单位：100 m）
KM_UNIT = 0.1  # 附件坐标 1 单位 = 100 m = 0.1 km

# 巡检点等级：颜色 + 形状双维度（色盲友好：橙/蓝/粉紫三色互不混淆，
# 形状/尺寸区分叠加，打印灰度下仍可辨识）
LEVEL_STYLE = {
    "I":   ("#E69F00", 30, True, "I级（实心圆）"),     # 橙实心圆（大）
    "II":  ("#0072B2", 30, False, "II级（空心圆）"),   # 蓝空心圆
    "III": ("#CC79A7", 14, True, "III级（小圆点）"),   # 粉紫小实心点
}
# 无人机航迹配色：seaborn muted 低饱和色板（蓝/橙/绿/紫/棕），色盲友好
UAV_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]
C_BLUE = "#4C72B0"   # 低于均值 / best-so-far 曲线
C_ORANGE = "#DD8452"  # 高于均值
C_GRAY = "#999999"   # 尝试区间带
C_RED = "#C0392B"    # 改进点 / 终止线


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
    """绘制巡检点（颜色 + 形状双维度区分等级）与基地黑色实心五角星。

    等级维度：I 级橙实心圆（大）、II 级蓝空心圆、III 级粉紫小实心点；
    基地用黑色实心五角星，尺寸显著大于巡检点，避免与普通点混淆。
    """
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
    """子图右下角线段比例尺（工程图习惯，替代纯坐标刻度）。

    取 1/2/5×10^k 中首个不小于 x 跨度 1/4 的整公里值，画黑色粗线段 +
    两端竖端线 + 上方公里文字；坐标单位换算：1 附件单位 = 0.1 km。
    """
    x0, x1 = ax.get_xlim()
    target_km = (x1 - x0) * KM_UNIT / 4.0
    mag = 10 ** math.floor(math.log10(target_km))
    km = 0.0
    for m in (1, 2, 5, 10):
        if m * mag >= target_km:
            km = m * mag
            break
    L = km / KM_UNIT  # 比例尺长度（附件单位）
    y0, y1 = ax.get_ylim()
    bx = x1 - L * 0.70  # 右下角留白
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
    """图3：任务分配与路径规划图（2×2，单图大图级重构）。

    子图左上角 (a)~(d) + N=X（无人机数量）；闭合航迹（基地出发→返回基地）；
    航迹低饱和 5 色、线宽 0.9、透明度 0.7（Case3 密集点群不糊成团）；
    每子图右下角线段比例尺；图例整图底部两组（巡检点等级 / 无人机航迹+基地）。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.8))
    labels = "abcd"
    # 航迹图例句柄按无人机编号去重收集（跨子图同编号同色）
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
        # 子图标签：左上角 (a)~(d) + N=X（与论文表 2 呼应）
        ax.text(0.015, 0.97, f"({lab}) {case}  N={n}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        ax.set_xlabel("X 坐标（单位：100 m）", fontsize=8.5)
        ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=8.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        _scale_bar(ax)
    # 整图图例（底部两组，避免每子图重复放置）：等级 3 + 基地 1 + 航迹 5
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
    """图4：各无人机工作时长偏离均值条形图（2×2，分钟尺度）。

    0 线=该算例平均时长（黑色粗实线）、蓝条=低于均值、橙条=高于均值，
    条端标注偏差（min，1 位小数）；子图右上角补充 δ = T_max − T_min（min）
    与均值（h），与论文表 2/表 3 指标直接对应；各子图横轴尺度独立。
    偏差口径：单机时长 − 平均时长，正值右偏、负值左偏（图注须写明）。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        uavs = arc["uavs"]
        n = len(uavs)
        ids = [u["uav_id"] for u in uavs]
        work_h = [u["work_h"] for u in uavs]
        mean = sum(work_h) / n
        dev = [(w - mean) * 60.0 for w in work_h]  # 偏离均值（min）
        y = np.arange(n)[::-1]  # 从上到下：无人机1 → 无人机n

        ax.axvline(0, color="black", linewidth=1.6, zorder=2)  # 0 线黑色粗实线
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
        # 轴范围给条端标注留出空间（Case2 条长仅 ±0.7 min，固定偏移 1.2 须在轴内）
        lim = (max(abs(d) for d in dev) + 1.6) * 1.2
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", alpha=0.3)
        # 核心量化值：δ = T_max − T_min 与均值（与论文表 2、表 3 对齐）
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
    """图5：固定机群下局部搜索验证曲线（2×2，目标值 best-so-far 包络）。

    语义（图注须写明）：本图展示固定最小机群数（N=N_min）下，局部搜索
    对构造初始解的验证过程——best-so-far 单调不增包络 + 搜索尝试区间。
    视觉设计：
    - 纵轴以最终最优值 f* 为基准收紧 ±3% 余量，最优值线（红虚线）处于
      画面视觉中心，灰带只保留紧贴曲线上方的"更差解背景"（淡灰 α=0.15，
      不再抢视线）；
    - 4 子图共用顶部一个图例（不重复放置）；
    - 子图右上角整合标注 `改进: N次 | 最优值: x.xxxx h`（4 位小数与
      求解日志一致）；
    - Case2 改进点用红色实心圆 + 箭头 + 幅度数值明确标注（四算例中唯一
      有优化的算例，与其余三个形成对比）。
    纵轴语义诚实性：f = Σt_k + T_max（见模块 docstring），非 T_max 本身。
    """
    NBIN = 60  # 时间分桶数（300 s 搜索 → 每桶 5 s）
    YL_MARGIN = 0.03  # 纵轴余量（以最终最优值为基准的 ±3%）
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.0))
    for ax, case in zip(axes.flat, CASES):
        csv_path = Q1_DIR / "logs" / f"convergence_{case}.csv"
        df = pd.read_csv(csv_path)
        t = df["elapsed_s"].values
        obj = df["objective"].values

        # best-so-far（单调不增）与时间分桶
        best = np.minimum.accumulate(obj)
        edges = np.linspace(t[0], t[-1] + 1e-9, NBIN + 1)
        bin_idx = np.clip(np.searchsorted(edges, t) - 1, 0, NBIN - 1)
        bin_best = np.full(NBIN, np.inf)
        np.minimum.at(bin_best, bin_idx, best)
        bin_max = np.full(NBIN, -np.inf)
        np.maximum.at(bin_max, bin_idx, obj)
        # 空桶前向填充
        bin_best = pd.Series(np.where(np.isfinite(bin_best), bin_best, np.nan)).ffill().to_numpy()
        bin_max = pd.Series(np.where(np.isfinite(bin_max), bin_max, np.nan)).ffill().to_numpy()
        bin_t = (edges[:-1] + edges[1:]) / 2.0

        # 1) 尝试区间带（best ~ 桶内最差尝试，恒在 best-so-far 上方；
        #    淡灰 α=0.15 弱化为"探索过的更差解"背景）
        ax.fill_between(bin_t, bin_best / 3600.0, bin_max / 3600.0,
                        step="post", color="#BBBBBB", alpha=0.15,
                        linewidth=0, zorder=1)
        # 2) best-so-far 单调不增阶梯线（x 为秒，与 xlim 0~305 一致；
        #    注意：曾误写 bin_t/3600 导致曲线被压缩在 x≈0 处不可见）
        ax.plot(bin_t, bin_best / 3600.0, drawstyle="steps-post",
                color=C_BLUE, linewidth=1.8, zorder=3)
        # 3) 改进点（桶级 best 下降处）+ 幅度标注
        #    chg_idx[i] 表示 bin_best[i+1] < bin_best[i]：改进时刻取第 i 桶
        #    中心，画点高度取下降后的 bin_best[i+1]，幅度 = 前值 − 后值
        chg_idx = np.flatnonzero(np.diff(bin_best) < 0)
        chg_t = bin_t[chg_idx]
        chg_v = bin_best[chg_idx + 1]
        n_imp = len(chg_idx)
        if n_imp > 0:
            ax.scatter(chg_t, chg_v / 3600.0, s=32, marker="o",
                       color=C_RED, zorder=4, edgecolors="white",
                       linewidths=0.5)
            for i0, (ct, cv) in enumerate(zip(chg_t, chg_v)):
                imp = bin_best[chg_idx[i0]] - cv  # 该级台阶下降量（h 换算）
                ax.annotate(f"−{imp / 3600.0:.2f} h",
                            xy=(ct, cv / 3600.0),
                            xytext=(ct + 8, cv / 3600.0 + 0.10),
                            fontsize=8, color=C_RED, fontweight="bold",
                            zorder=5,
                            arrowprops=dict(arrowstyle="->", lw=0.7,
                                            color=C_RED, shrinkA=2, shrinkB=2))
        # 4) 最终最优值线（红虚线，视觉中心基准，zorder 最高：
        #    0 改进算例中与蓝线包络重合，红线在上、虚线缺口透出蓝线，
        #    两个元素同时可辨）+ 300 s 阶段终止线
        final = bin_best[-1] / 3600.0
        ax.axhline(final, color=C_RED, linestyle="--", linewidth=1.2,
                   alpha=0.9, zorder=4)
        ax.axvline(300, color=C_GRAY, linestyle="--", linewidth=1.0,
                   alpha=0.8, zorder=2)
        # 右上角整合标注：改进次数与最优值一行（格式统一）
        ax.text(0.985, 0.97, f"改进: {n_imp}次 | 最优值: {final:.4f} h",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color="#333333", zorder=6, linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.text(0.015, 0.97, case, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        # 视觉重心：纵轴以最终最优值为基准收紧 ±3%，最优值线居画面中央
        ax.set_ylim(final * (1 - YL_MARGIN), final * (1 + YL_MARGIN))
        ax.set_xlim(0, 305)
        ax.set_xticks(np.arange(0, 301, 60))
        ax.set_xlabel("搜索时间（s）", fontsize=8.5)
        ax.set_ylabel("目标值 f（h）", fontsize=8.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    # 4 子图共用顶部一个图例（不重复放置）
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
