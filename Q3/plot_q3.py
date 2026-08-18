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
问题3 成果图（图9、图10、图11）。

图9：禁飞区与绕行路径图（典型算例 Case2 单图大图）——四个算例中仅
     Case2 存在真实绕行（detour 5 段 + wait_direct 2 段 + visibility
     2 段共 9 个非直飞段；Case1 全程直飞，Case3/4 各仅 1~2 段），故
     图9 选取 Case2 放大绘制。视觉分层：直飞（含等待后直飞）退视觉
     底层（极浅灰细线、低透明度，仅示意路径存在——wait_direct 空间
     形态与直飞无差异，时间属性不在此区分）；绕行（单圆绕飞与多圆
     可见图路径）深红粗线+白色细描边强化主角；巡检点分级（普通点
     极浅灰小点，绕行段起点/终点/切点深灰实心圆点作路径锚点）；禁飞区
     深红虚线边框+10% 透明度极浅红填充（多圆重叠不加深）、Zone 编号
     标注于圆心旁（时间窗信息在图注）。
     关键：detour/visibility 段绘制真实绕行路径（切点直线 + 沿禁飞区
     圆边弧线，迭代绕行保证路径不穿过任何禁飞区），而非起终点直线——
     其直线弦恰好穿过禁飞区（几何验证 d=3.1~43.3 < r），直线画法会
     造成"红线穿禁飞区"的物理矛盾；direct/wait_direct 段保持直线
     （飞行时刻与禁飞窗不冲突，等待后直飞段已等禁区解除）。现存多圆
     绕行在 Z4/Z3（同窗 10:30-11:30 生效、按并集避让，134→66 可见图段
     为典型例）；Z1/Z2 为空间相交且同时生效的圆对，聚焦轮后其并集
     穿越段已被消除。
图10：无人机任务甘特图（2×2）——每算例一子图：独立横轴刻度（自 8:00
      起至各自 S_max，仅整数小时刻度），纵轴 UAV 1~N 自上而下；飞行段
      与巡检作业合并为单一浅蓝块「任务执行中」（小时级横轴下 300 s
      巡检只是细竖条噪音，仅以深蓝细竖线标记各巡检作业时刻，保留整块
      感同时体现多巡检点串联节奏）、深灰实心块=必要等待（条上方标注
      时长，红色虚线自等待起点连至禁飞带顶，直观呈现"边界整定等待=
      等禁区解除再出发"，与报告 4.3 节机理一一对应）、浅红背景带=禁飞
      时段（单区/重叠区深浅两档、顶部边缘 Zone 编号、时间窗入图注；
      零等待算例透明度再降并角注"无等待"结论）；行内非作业空白极浅灰
      底、每行右端标注单机完成时刻（呼应 δ 负载均衡）、横轴细虚线锚定
      禁飞起止时刻；右上角右对齐 S_max 与总等待；整图底部居中 3 项图例。
图11：绕行代价对比（单图三子图）——S_max（h）与总飞行距离（km）两组
      "问题2 无禁飞区基线 vs 问题3 方案"双柱对比（柱顶红色增幅，以
      问题2 为基准），下子图为四算例总必要等待时长（s）——"时间代价-
      距离代价-等待代价"三维成本闭环。问题2 距离由 task_routes 按
      欧氏直线逐段重建（Q2 无禁飞约束，直线飞行）。

数据来源：
- Q3 解档案 outputs/workbooks/q3/strict/Case*.json（schedules/segments/metrics）
- Q2 最终档案 enhanced_run_20260817_e2000/q2/strict/Case*.json
  （task_routes/fleet_size/metrics.Tmax_s；注意 outputs/workbooks/q2/strict/
  为过时档案）
- 附件1（坐标与巡检等级）、附件2（禁飞区圆心/半径/时间窗）
输出：Q3/figs/fig9_*.png/.pdf、fig10_*.png/.pdf、fig11_*.png/.pdf

用法（仓库根目录，math 环境）：
    python Q3/plot_q3.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke

# 中文字体：须在 style.use 之后再设置，否则被 seaborn 风格覆盖为 Arial
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ["Case1", "Case2", "Case3", "Case4"]
FIG_DIR = REPO_ROOT / "Q3" / "figs"
ARCHIVE = REPO_ROOT / "outputs" / "workbooks"
# Q2 最终档案（图11 基线；报告表3 依据）；outputs/workbooks/q2/strict/
# 为过时档案不可用
Q2_ARCHIVE = REPO_ROOT / "enhanced_run_20260817_e2000" / "q2" / "strict"
ATTACH_XLSX = REPO_ROOT / "attachment" / "附件1.xlsx"
ATTACH_ZONES = REPO_ROOT / "attachment" / "附件2.xlsx"
START_HOUR = 8.0  # 每天 8:00 开始执行任务
BASE = (0.0, 0.0)

LEVEL_VISITS = {"I": 3, "II": 2, "III": 1}  # 与 Q1 solve_q1.py 一致的展开规则
# 图9 视觉分层（空间图聚焦"禁飞避让"主体）：
#   直飞（含等待后直飞）退视觉底层（极浅灰细线示意存在）；绕行（单圆/
#   可见图）深红粗线+白描边强化主角；巡检点分级（普通点极浅灰小点、
#   绕行锚点深灰实心圆点）；禁飞区深红虚线边框+极浅红填充、编号入圆内
PT_COLOR = "#DDDDDD"          # 普通巡检点（极浅灰小点，示意点位存在）
ANCHOR_COLOR = "#333333"      # 绕行锚点（起终点/切点，深灰实心圆点）
DIRECT_COLOR = "#E0E0E0"      # 直飞 / 等待后直飞（空间形态无差异）
DETOUR_COLOR = "#C0392B"      # 绕行 / 可见图路径（深红，全图最粗）
ZONE_FACE = "#E74C3C"         # 禁飞区浅红填充
ZONE_EDGE = "#8B1A1A"         # 禁飞区深红实线边界
# 图10 甘特图（时间维度聚焦）：
FLIGHT_COLOR = "#9DC3E6"      # 任务执行时段浅蓝（飞行+巡检合并，单一色块）
SERVICE_TICK = "#2F5D8C"      # 巡检作业时刻竖线（深蓝细线，弱颗粒度节奏标记）
WAIT_COLOR = "#666666"        # 必要等待深灰实心块
ROW_BG = "#F5F5F5"            # 行内非作业空白极浅灰底（柔化空白感）
Q2_BASE_COLOR = "#DD8452"     # 图11 问题2 无禁飞区基线（橙，与 Q2 体系一致）
Q3_COLOR = "#55A868"          # 图11 问题3 方案（绿）


def save_fig(fig, path, dpi=300):
    """保存 png/pdf 双格式。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"图片已保存到：{path}（dpi={dpi}）")


# ---------------- 数据装载 ----------------
def load_case(case: str) -> dict:
    """附件1 → {pid: {x, y, level}}，并按 Q1 规则展开任务：{task_id: pid}。"""
    df = pd.read_excel(ATTACH_XLSX, sheet_name=case)
    points = {}
    for row in df.itertuples(index=False):
        pid = int(row.Point_ID)
        points[pid] = {"x": float(row.X_Coordinate), "y": float(row.Y_Coordinate),
                       "level": str(row.Inspection_Level).strip().upper()}
    task_of = {}
    tid = 1
    for pid in points:
        for _ in range(LEVEL_VISITS[points[pid]["level"]]):
            task_of[tid] = pid
            tid += 1
    return {"points": points, "task_of": task_of}


def load_zones(case: str) -> list[dict]:
    """附件2 → 禁飞区列表（时间窗转自 8:00 起的秒数）。"""
    df = pd.read_excel(ATTACH_ZONES, sheet_name=case)
    zones = []
    for row in df.itertuples(index=False):
        def to_s(t: str) -> int:
            h, m = str(t).strip().split(":")
            return (int(h) - START_HOUR) * 3600 + int(m) * 60
        zones.append({
            "id": str(row.Zone_ID),
            "x": float(row.Center_X), "y": float(row.Center_Y),
            "r": float(row.Radius),
            "t0": to_s(row.Start_Time), "t1": to_s(row.End_Time),
            "win": f"{row.Start_Time}–{row.End_Time}",
        })
    return zones


def load_q3(case: str) -> dict:
    return json.load(open(ARCHIVE / "q3" / "strict" / f"{case}.json", encoding="utf-8"))


def seg_point(case_data: dict, node_id: int):
    """节点号 → (x, y)：0 为基地，其余为任务号。"""
    if node_id == 0:
        return BASE
    pid = case_data["task_of"][node_id]
    p = case_data["points"][pid]
    return (p["x"], p["y"])


# ---------------- 禁飞区绕行路径重建（复用求解器同款几何） ----------------
sys.path.insert(0, str(REPO_ROOT))  # domain 依赖顶层包 Q2
sys.path.insert(0, str(REPO_ROOT / "Q3"))
from geometry import detour_path, visibility_path, TAU  # noqa: E402
from domain import EPS_ARC_KM, SAFETY_MARGIN_KM  # noqa: E402

KM_UNIT = 0.1  # 附件坐标 1 单位 = 100 m = 0.1 km（求解器几何在 km 空间）


def _to_km(p) -> tuple[float, float]:
    return (p[0] * KM_UNIT, p[1] * KM_UNIT)


def _to_att(p) -> tuple[float, float]:
    return (p[0] / KM_UNIT, p[1] / KM_UNIT)


def _arc_sample(center, r, a0, a1, n=36) -> list[tuple[float, float]]:
    """圆边 a0→a1 的短弧采样（不含端点）。

    方向与 safe_path._arc_bounds 同款重算：forward 取短弧（≤ π），
    不直接依赖 VisibilityPath.arc_legs 的 direction——该字段对反向边
    （b→a 的边）未按路径方向翻转，直接使用会走出长弧。
    """
    forward = (a1 - a0) % TAU
    sweep = forward if forward <= TAU - forward else forward - TAU
    return [_to_att((center[0] + r * math.cos(a0 + sweep * j / n),
                     center[1] + r * math.sin(a0 + sweep * j / n)))
            for j in range(1, n)]


def _detour_polyline(p0, p1, zone):
    """单圆绕行路径（与求解器 safe_path 的 detour 分支同构）：
    p0 → 切点 → 短弧 → 切点 → p1，弧构造半径 = safe_radius + EPS。
    返回 (polyline, 切点列表)。"""
    c = _to_km((zone["x"], zone["y"]))
    r = zone["r"] * KM_UNIT + SAFETY_MARGIN_KM + EPS_ARC_KM
    res = detour_path(_to_km(p0), _to_km(p1), c, r)
    if res is None:  # 端点无切点（求解器不会产生此情形），直线兜底
        return [p0, p1], []
    points, _, _ = res
    a0 = math.atan2(points[1][1] - c[1], points[1][0] - c[0])
    a1 = math.atan2(points[2][1] - c[1], points[2][0] - c[0])
    poly = [p0] + _arc_sample(c, r, a0, a1) + [p1]
    return poly, [_to_att(points[1]), _to_att(points[2])]


def _visibility_polyline(p0, p1, zones_active):
    """多圆可见性图绕行路径（与求解器 visibility 分支同构）。
    返回 (polyline, 中间骨架切点列表)。"""
    if not zones_active:
        return [p0, p1], []
    disks = [(_to_km((z["x"], z["y"])), z["r"] * KM_UNIT + SAFETY_MARGIN_KM)
             for z in zones_active]
    vp = visibility_path(_to_km(p0), _to_km(p1), disks, margin=EPS_ARC_KM)
    if vp is None:
        return [p0, p1], []
    pts_list = []
    for i in range(len(vp.points) - 1):
        pts_list.append(_to_att(vp.points[i]))
        leg = next((lg for lg in vp.arc_legs if lg.start == i), None)
        if leg is not None:
            a0 = math.atan2(vp.points[i][1] - leg.center[1],
                            vp.points[i][0] - leg.center[0])
            a1 = math.atan2(vp.points[i + 1][1] - leg.center[1],
                            vp.points[i + 1][0] - leg.center[0])
            pts_list += _arc_sample(leg.center, leg.radius_km, a0, a1)
    pts_list.append(_to_att(vp.points[-1]))
    tangs = [_to_att(vp.points[i]) for i in range(1, len(vp.points) - 1)]
    return pts_list, tangs


# ---------------- 图9：禁飞区与绕行路径图 ----------------
def _draw_case_routes(ax, case: str, case_data: dict, arc3: dict, zones: list[dict]):
    """在子图上绘制某算例的禁飞区与全部航迹（视觉分层版）。

    路径绘制规则（几何诚实性）：
    - direct / wait_direct：直线——飞行时刻与禁飞窗不冲突（wait_direct
      已等待禁区解除），真实路径即直线；两者空间形态无差异（分类属时间
      属性），统一极浅灰细线、低透明度，退到视觉底层仅示意路径存在；
    - detour / visibility：真实绕行路径（深红粗线 + 白色细描边）——
      解档案只存起终点（其直线弦恰好穿过禁飞区圆），绘图用求解器同款
      几何（geometry.detour_path / visibility_path）重建"切点直线 + 沿
      禁飞区圆边弧线"的路径。
    巡检点分级：普通点极浅灰小点；绕行段起点/终点/切点深灰实心圆点作
    锚点。基地黑色实心五角星（Case2 基地位于 Z3 圆内，星形醒目保留）。
    "绕行"文字标注全图仅保留 1 处（最长 detour 弧，箭头引出）。
    """
    zones_by_id = {z["id"]: z for z in zones}
    # 禁飞区：深红虚线边框 + 10% 透明度极浅红填充（多圆重叠不加深），
    # Zone 编号标注于圆心旁（时间窗移图注）
    for z in zones:
        ax.add_patch(Circle((z["x"], z["y"]), z["r"], facecolor=ZONE_FACE,
                            alpha=0.08, edgecolor=ZONE_EDGE, linewidth=1.0,
                            linestyle="--", zorder=1))
        ax.text(z["x"], z["y"], z["id"], ha="center", va="center",
                fontsize=7.5, color=ZONE_EDGE, fontweight="bold", zorder=4)
    # 普通巡检点（极浅灰小点，仅示意点位存在）
    for pid, p in case_data["points"].items():
        ax.scatter(p["x"], p["y"], s=8, c=PT_COLOR, alpha=0.9,
                   edgecolors="none", zorder=2)
    # 航迹：直飞极浅灰细线（alpha 0.25）；绕行深红粗线 + 白描边
    anchors = set()
    best_arc = None  # 最长 detour 弧（"绕行"标注锚点）
    for sched in arc3["schedules"]:
        for seg in sched["segments"]:
            p0 = seg_point(case_data, seg["from_id"])
            p1 = seg_point(case_data, seg["to_id"])
            if seg["path_type"] in ("detour", "visibility"):
                # 真实绕行路径：复用求解器同款几何重建（切点直线 + 禁飞区
                # 圆边弧线），detour 绕 affected_zones 声明的单圆、visibility
                # 绕受影响圆集——解档案只存起终点，直线弦恰好穿过禁飞区
                if seg["path_type"] == "detour" and seg.get("affected_zones"):
                    poly, tangs = _detour_polyline(
                        p0, p1, zones_by_id[seg["affected_zones"][0]])
                else:
                    active = [z for z in zones
                              if z["id"] in seg.get("affected_zones", [])]
                    poly, tangs = _visibility_polyline(p0, p1, active)
                xs, ys = zip(*poly)
                ax.plot(xs, ys, "-", color=DETOUR_COLOR, linewidth=2.2,
                        alpha=0.95, zorder=3, solid_capstyle="round",
                        path_effects=[withStroke(linewidth=2.9,
                                                 foreground="white")])
                if p0 != BASE:
                    anchors.add(p0)
                if p1 != BASE:
                    anchors.add(p1)
                for t in tangs:
                    anchors.add(t)
                if seg["path_type"] == "detour":
                    arc_len = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                                  for i in range(len(xs) - 1))
                    if best_arc is None or arc_len > best_arc[0]:
                        best_arc = (arc_len, poly[len(poly) // 2])
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=DIRECT_COLOR,
                        linewidth=0.5, alpha=0.25, zorder=2,
                        solid_capstyle="round")
    # 绕行锚点（起点/终点/切点，深灰实心圆点）
    for a in anchors:
        ax.plot(*a, marker="o", markersize=4.5, color=ANCHOR_COLOR,
                markeredgecolor="white", markeredgewidth=0.3, zorder=4)
    # 基地（黑色实心五角星，醒目保留——Case2 基地恰位于 Z3 圆内）
    ax.plot(*BASE, marker="*", markersize=15, color="black",
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    # "绕行"标注：仅 1 处（最长 detour 弧中点，箭头引出）
    if best_arc is not None:
        mid = best_arc[1]
        ax.annotate("绕行", xy=mid, xytext=(mid[0] + 22, mid[1] - 14),
                    fontsize=8, color=DETOUR_COLOR, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=DETOUR_COLOR))


def fig9_routes(archives: dict[str, dict], data: dict[str, dict],
                zones_all: dict[str, list]) -> plt.Figure:
    """图9：典型算例 Case2 禁飞区绕行路径大图（单图）。

    四算例中仅 Case2 存在真实绕行（detour 5 段 + wait_direct 2 段 +
    visibility 2 段共 9 个非直飞段；Case1 无任何非直飞段，Case3 仅 2 段
    可见性调整、Case4 仅 1 段等待后直飞，绕行细节无展示价值），故选取
    Case2 画单算例大图。图幅按数据纵横比动态调整（Case2 纵向跨度约为
    横向的 2.5 倍）。"""
    case = "Case2"
    arc3, case_data, zones = archives[case], data[case], zones_all[case]
    all_pts = list(case_data["points"].values())
    xs = [p["x"] for p in all_pts] + [z["x"] + z["r"] for z in zones] + \
         [z["x"] - z["r"] for z in zones]
    ys = [p["y"] for p in all_pts] + [z["y"] + z["r"] for z in zones] + \
         [z["y"] - z["r"] for z in zones]
    xspan = max(xs) - min(xs)
    yspan = max(ys) - min(ys)
    fig_h = 10.0
    fig_w = max(6.0, fig_h * xspan / yspan)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_case_routes(ax, case, case_data, arc3, zones)
    # ---- 图例与标注 ----
    ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    ax.set_xlabel("X 坐标（单位：100 m）", fontsize=12)
    ax.set_ylabel("Y 坐标（单位：100 m）", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)  # 去主网格，只留坐标轴刻度，减少线条干扰
    handles = [plt.Line2D([0], [0], ls="-", color=DIRECT_COLOR, lw=0.5,
                          label="直飞"),
               plt.Line2D([0], [0], ls="-", color=DETOUR_COLOR, lw=2.2,
                          label="绕行"),
               plt.Line2D([0], [0], ls="--", color=ZONE_EDGE, lw=1.0,
                          label="禁飞区"),
               plt.Line2D([0], [0], marker="o", ls="", color=PT_COLOR,
                          label="巡检点"),
               plt.Line2D([0], [0], marker="o", ls="", color=ANCHOR_COLOR,
                          label="绕行锚点"),
               plt.Line2D([0], [0], marker="*", ls="", color="black",
                          label="基地")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2,
              frameon=True, framealpha=0.85)
    fig.tight_layout()
    return fig


# ---------------- 图10：甘特图 ----------------
def fig10_gantt(archives: dict[str, dict], zones_all: dict[str, list]) -> plt.Figure:
    """图10：无人机任务甘特图（2×2，横轴为自 8:00 起的时间）。

    视觉规格（三轮定稿）：飞行段与巡检作业合并为单一浅蓝块「任务执行
    中」，各巡检作业时刻以深蓝细竖线标记（弱颗粒度：保留整块感、体现
    多巡检点串联节奏，密集/稀疏一目了然）；深灰实心块=必要等待（条上
    方标注时长，红色虚线自等待起点连至禁飞带顶部边缘——"边界整定等待
    =等禁区解除再出发"的图面证据，与报告 4.3 节机理绑定）；浅红背景
    带=禁飞时段（单区/重叠区深浅两档、顶部边缘 Zone 编号、时间窗入
    图注；零等待算例透明度再降并角注"无等待"结论）；行内非作业空白
    极浅灰底、每行右端标注单机完成时刻（呼应 δ 负载均衡）、横轴细虚线
    锚定禁飞起止时刻；整数小时刻度、行间分隔线；右上角右对齐 S_max 与
    总等待；整图底部居中 3 项图例。
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    for ax, case in zip(axes.flat, CASES):
        arc = archives[case]
        zones = zones_all[case]
        n = len(arc["schedules"])
        smax_h = arc["metrics"]["S_max_s"] / 3600.0
        total_wait = sum(seg.get("wait_s", 0)
                         for s in arc["schedules"] for seg in s["segments"])
        # ---- 无人机行底：非作业空白极浅灰（zorder 0，最底层） ----
        for i in range(n):
            ax.axhspan(n - i - 0.31, n - i + 0.31, color=ROW_BG, zorder=0)
        # ---- 禁飞时段：单区浅红 + 重叠区加深（深浅两档） ----
        spans = [(z["t0"], z["t1"]) for z in zones if z["t0"] < z["t1"]]
        over = []  # 空间重叠区间（两两交集再合并）
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])
                if a < b:
                    over.append([a, b])
        if over:
            over.sort()
            merged_o = [over[0]]
            for a, b in over[1:]:
                if a <= merged_o[-1][1]:
                    merged_o[-1][1] = max(merged_o[-1][1], b)
                else:
                    merged_o.append([a, b])
            for a, b in merged_o:
                ax.axvspan(a / 3600.0, b / 3600.0, color=ZONE_FACE,
                           alpha=0.14 if total_wait == 0 else 0.20,
                           zorder=0.5)
        for z in zones:
            if z["t0"] >= z["t1"]:  # 零时长窗口（如 Case4 Z8 单点）不渲染
                continue
            ax.axvspan(z["t0"] / 3600.0, z["t1"] / 3600.0, color=ZONE_FACE,
                       alpha=0.08 if total_wait == 0 else 0.12,
                       zorder=0.5)
            # 禁飞带顶部边缘 Zone 编号（时间窗统一入图注，不图面堆叠）
            ax.text(z["t0"] / 3600.0 + 0.04, n - 0.02, z["id"],
                    ha="left", va="top", fontsize=6.5, color=ZONE_EDGE,
                    fontweight="bold", zorder=4)
            # 横轴锚点：禁飞区起止时刻细虚线（时间参照）
            for t in (z["t0"], z["t1"]):
                ax.axvline(t / 3600.0, color="#B0B0B0", ls="--", lw=0.5,
                           alpha=0.8, zorder=0.6)
        for i, sched in enumerate(arc["schedules"]):
            y = n - i  # 自上而下：UAV 1 在最上
            # 任务执行时段 = 飞行段 ∪ 巡检作业（相接即合并，消除斑马纹）
            ivs = [(seg["depart_s"], seg["arrive_s"])
                   for seg in sched["segments"]]
            ivs += [(s, e) for s, e in sched["service_intervals"]]
            ivs.sort()
            merged = []
            for s, e in ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            for s, e in merged:
                ax.barh(y, (e - s) / 3600.0, left=s / 3600.0, height=0.62,
                        color=FLIGHT_COLOR, alpha=0.9, edgecolor="none",
                        zorder=2)
            # 巡检作业时刻：深蓝细竖线（弱颗粒度节奏标记，贯穿条带高度）
            for s, e in sched["service_intervals"]:
                ax.vlines((s + e) / 2 / 3600.0, y - 0.31, y + 0.31,
                          color=SERVICE_TICK, linewidth=0.5, zorder=2.5)
            for seg in sched["segments"]:
                if seg["wait_s"] > 0:
                    w = seg["wait_s"] / 3600.0
                    xw = seg["arrive_s"] / 3600.0
                    ax.barh(y, w, left=xw, height=0.62, color=WAIT_COLOR,
                            edgecolor="none", zorder=3)
                    # 等待时长标注（条上方，白底保证与飞行段重叠时可读）
                    ax.text(xw + w / 2, y + 0.42, f"{seg['wait_s']} s",
                            ha="center", va="bottom", fontsize=6.8,
                            color="#333333", zorder=5,
                            bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="none", pad=1.5))
                    # 边界整定等待锚点：红色虚线自等待起点连至禁飞带顶
                    # （等待段起点必落在某禁飞窗内，等禁区解除再出发）
                    if any(z["t0"] / 3600.0 <= xw <= z["t1"] / 3600.0
                           for z in zones if z["t0"] < z["t1"]):
                        ax.plot([xw, xw], [y + 0.31, n - 0.02], ls="--",
                                color=ZONE_EDGE, lw=0.6, alpha=0.75,
                                zorder=3.5)
            # 单机完成时刻标注（行最右端，与 δ 负载均衡指标呼应）
            end_h = merged[-1][1] / 3600.0
            x_t, ha_t = end_h + 0.05, "left"
            if x_t > smax_h - 0.05:  # 最重 UAV 完成时刻即 S_max，标注移入行内
                x_t, ha_t = smax_h - 0.05, "right"
            ax.text(x_t, y, f"{end_h:.2f} h", ha=ha_t, va="center",
                    fontsize=6.8, color="#333333", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=1))
        ax.text(0.015, 0.98, case, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        # 右上角核心指标（右对齐悬浮白底，与论文表4 对齐）
        ax.text(0.985, 0.98, f"S_max = {smax_h:.2f} h\n总等待 = {total_wait} s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3))
        if total_wait == 0:  # 零等待算例：主动点出结论
            ax.text(0.015, 0.045, "无等待：任务时序完全避开禁飞窗口",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.8, color="#888888", zorder=6)
        ax.set_xlabel("时刻（自 8:00 起，h）", fontsize=10)
        ax.set_yticks(range(n, 0, -1))
        ax.set_yticklabels([f"UAV {i + 1}" for i in range(n)])
        ax.set_xlim(0, smax_h)  # 终止点对齐各自 S_max
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 仅整数小时刻度
        ax.grid(axis="x", which="major", alpha=0.15, color="#CCCCCC")
        # 无人机行之间极浅灰分隔线（UAV 1~N 行边界清晰）
        for k in range(n + 1):
            ax.axhline(n - k + 0.5, color="#DDDDDD", lw=0.4, zorder=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FLIGHT_COLOR,
                             label="任务执行时段"),
               plt.Rectangle((0, 0), 1, 1, color=WAIT_COLOR,
                             label="必要等待（标注时长）"),
               plt.Rectangle((0, 0), 1, 1, color=ZONE_FACE, alpha=0.18,
                             label="禁飞时段")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    return fig


# ---------------- 图11：绕行代价对比 ----------------
def q2_distance_km(case: str, case_data: dict) -> float:
    """问题2 基线总飞行距离（km）：由 task_routes 按欧氏直线逐段重建。

    Q2 无禁飞约束，真实路径即任务点间直线；解档案只存任务序列
    （task_routes），距离无字段——按任务号→巡检点展开后逐段累加
    （含返回基地段），与 Q3 侧 total_distance_km 同口径（附件坐标
    ×0.1 转 km）。"""
    q2 = json.load(open(Q2_ARCHIVE / f"{case}.json", encoding="utf-8"))
    total = 0.0
    for route in q2["task_routes"]:
        prev = BASE
        for t in route:
            p = case_data["points"][case_data["task_of"][t]]
            total += math.hypot(p["x"] - prev[0], p["y"] - prev[1]) * KM_UNIT
            prev = (p["x"], p["y"])
        total += math.hypot(-prev[0], -prev[1]) * KM_UNIT  # 返回基地
    return total


def fig11_detour_cost(q3_all: dict[str, dict], data: dict[str, dict]) -> plt.Figure:
    """图11：绕行代价三维对比（S_max / 总距离 / 总等待），问题2 为基线。

    上两组为分组条形："问题2 无禁飞区方案"（橙）vs "问题3 方案"（绿）——
    S_max（h）与总飞行距离（km）各自成组；柱顶标数值、双柱上方标红色
    相对增幅（以问题2 为基准）。下子图：四算例总必要等待时长（s）——
    "时间代价-距离代价-等待代价"三维成本闭环，与报告 4.1 节数值对齐。
    """
    q2_tmax_h = []
    q2_dist_km = []
    for c in CASES:
        q2 = json.load(open(Q2_ARCHIVE / f"{c}.json", encoding="utf-8"))
        q2_tmax_h.append(q2["metrics"]["Tmax_s"] / 3600.0)
        q2_dist_km.append(q2_distance_km(c, data[c]))
    s3 = [q3_all[c]["metrics"]["S_max_s"] / 3600.0 for c in CASES]
    d3 = [q3_all[c]["metrics"]["total_distance_km"] for c in CASES]
    w3 = [sum(seg.get("wait_s", 0) for s in q3_all[c]["schedules"]
              for seg in s["segments"]) for c in CASES]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11.5), sharex=True)
    x = np.arange(len(CASES))
    width = 0.35

    def _grouped(ax, base_vals, q3_vals, unit):
        """一组分组条形：柱顶数值 + 上方红色增幅（以问题2 为基准）。"""
        b1 = ax.bar(x - width / 2, base_vals, width, color=Q2_BASE_COLOR,
                    edgecolor="white", label="问题 2 无禁飞区方案", zorder=3)
        b2 = ax.bar(x + width / 2, q3_vals, width, color=Q3_COLOR,
                    edgecolor="white", label="问题 3 方案", zorder=3)
        for b, v in zip(list(b1) + list(b2), base_vals + q3_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(base_vals) * 0.015,
                    f"{v:.3f}" if unit == "h" else f"{v:.1f}",
                    ha="center", fontsize=8, zorder=4)
        for i in range(len(CASES)):
            inc = (q3_vals[i] - base_vals[i]) / base_vals[i] * 100
            ax.text(x[i], max(base_vals[i], q3_vals[i]) * 1.07,
                    f"+{inc:.1f}%", ha="center", fontsize=10.5,
                    color=ZONE_EDGE, fontweight="bold", zorder=4)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    _grouped(ax1, q2_tmax_h, s3, "h")
    ax1.set_ylabel("总体完成时间（h）", fontsize=11)
    _grouped(ax2, q2_dist_km, d3, "km")
    ax2.set_ylabel("总飞行距离（km）", fontsize=11)

    # 下子图：总必要等待时长
    ax3.bar(x, w3, width * 1.1, color=Q3_COLOR, edgecolor="white", zorder=3)
    for i, w in enumerate(w3):
        ax3.text(x[i], w + max(w3) * 0.03, f"{w} s", ha="center",
                 fontsize=10, color=ZONE_EDGE, fontweight="bold", zorder=4)
    ax3.set_ylabel("总必要等待（s）", fontsize=11)
    ax3.set_xlabel("测试算例", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(CASES)
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {c: load_case(c) for c in CASES}
    zones_all = {c: load_zones(c) for c in CASES}
    q3_all = {c: load_q3(c) for c in CASES}

    jobs = [
        (fig9_routes(q3_all, data, zones_all), "fig9_禁飞区与绕行路径"),
        (fig10_gantt(q3_all, zones_all), "fig10_无人机任务甘特图"),
        (fig11_detour_cost(q3_all, data), "fig11_绕行代价对比"),
    ]
    for fig, name in jobs:
        for ext in ("png", "pdf"):
            save_fig(fig, FIG_DIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"完成：{name}")


if __name__ == "__main__":
    main()
