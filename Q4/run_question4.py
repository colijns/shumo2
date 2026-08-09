# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31

"""问题4 入口：介质A、B混合填充最低成本二维单调搜索。

口径（docs/问题4.md）：
- 决策变量 (N_A, N_B) 整数；成本 C = 1.05·N_A·V_A + 0.05·N_B·V_B（元，
  单件 c_A ≈ 1.4844e-2、c_B ≈ 1.6755e-3）；约束 P(N_A, N_B) ≥ 0.90；
- 共同随机数：trial_seed = BASE_SEED + trial_index，每试验生成固定顺序的
  A 序列（na_max 根）与 B 序列（nb_max 个），候选 (N_A, N_B) 取对应前缀；
  Y_s 对任一维 +1 单调不减 → 前缀子图边集合是全集子图子集；
- 外层成本、内层配比：给定成本上限 C_0，N_B^max(N_A; C_0) = floor((C_0 - c_A·N_A)/c_B)；
  可行性关于 C_0 单调 → 成本上二分 → 带下降 → c_B 粒度细化 → 最低可行层全扫；
- 统计判定：搜索阶段只认"点估计可行"（p_hat ≥ 0.90）；验证阶段从 C* 起
  向上分层推进（每层评估该成本层非支配边界点，Wilson 95% 区间：
  P_U < 0.90 判不足 / P_L ≥ 0.90 判可靠可行 / 跨线分级追加 M0 → M1 → M_FINAL），
  出现可靠点即停，层内全不足则抬层（细步进 8c_B / 粗步进 32c_B），直至纯A
  成本兜底上界；最终最优 = 验证后可靠最便宜点（验证推翻搜索层时向上取，
  无可靠点则保留点估计最优并警告）；点估计最优与置信约束保守最优分开报告；
- 独立复算：BASE_SEED + 12345 新种子，复算最优、邻点、纯A、纯B；
- 自检：单调性（逐 trial 违反数）、A 跨壁率 ≈ 60.08%、B 跨壁率 ≈ 11.53%。

环境变量（Q3 同风格）：
SHUMO_Q4_TRIALS_SEARCH（默认 100）、SHUMO_Q4_TRIALS_FINAL（默认 2000）、
SHUMO_Q4_TRIALS_BASELINE（默认 1500）、SHUMO_Q4_WORKERS（默认 min(8, cpu)）、
SHUMO_Q4_BASE_SEED（默认 42）、SHUMO_Q4_NA_MAX（默认 900）、
SHUMO_Q4_NB_BASELINE（默认 8000）。--smoke 走小样本全链路（M 缩到 1/20 级）。

输出 results/question4_{result,boundary,verify,points}.csv。
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Q3'))
import geometry as geo      # noqa: E402
import geometry_mix as gm   # noqa: E402
import first_passage as fp  # noqa: E402
from monte_carlo import wilson_ci  # noqa: E402

P_TARGET = 0.90             # 题目导通概率要求
VERIFY_SEED_OFFSET = 12345  # 独立复算种子偏移
MONO_SEED_OFFSET = 500000   # 单调性自检种子偏移
M_ANCHOR_DEFAULT = 8000     # 纯A 锚点兜底样本（验证带无可靠点时的终级判定）
CHUNK_T = 32                # worker job：试验批大小
CHUNK_P = 32                # worker job：候选点批大小
BATCH_FEA = 48              # fea 层扫描的评估块大小
BAND_STEP = 8               # 带下降/验证细步进步长（c_B 倍数）
BAND_STEP_COARSE = 32       # 验证粗步进（层内无跨线点时加速抬层）
CROSS_FRAC_FINE = 4         # 层内跨线点占比 ≤ 1/4 才视为接近真层（细步进）
CROSS_MAX_FINE = 8         # 跨线点 ≤ 8 个时同样细步进（绝对数兜底）
V2_CAP = 200                # 独立复算集合上限（成本升序取前 N，防兜底路径爆炸）
COST_TOL = 1e-9             # 整除修正浮点容差（元）

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CSV_RESULT = os.path.join(OUT_DIR, 'question4_result.csv')
CSV_BOUNDARY = os.path.join(OUT_DIR, 'question4_boundary.csv')
CSV_VERIFY = os.path.join(OUT_DIR, 'question4_verify.csv')
CSV_POINTS = os.path.join(OUT_DIR, 'question4_points.csv')


def fmt_pct(v):
    """百分比输出：保留百分号下两位小数。"""
    return f'{v * 100:.2f}%'


def b_lim(na, c0):
    """成本上限 c0（元）下、给定 na 时最多可加入的 B 数量（floor，容差保护）。"""
    return int(np.floor((c0 - gm.c_A * na) / gm.c_B + COST_TOL))


def wilson_verdict(x, m):
    """Wilson 95% 判定。返回 (verdict, lo, hi)；verdict ∈ reliable/insufficient/crossing。"""
    _, lo, hi = wilson_ci(int(x), int(m))
    if hi < P_TARGET:
        return 'insufficient', lo, hi
    if lo >= P_TARGET:
        return 'reliable', lo, hi
    return 'crossing', lo, hi


def n90_from_contacts(contacts, n_max, m):
    """首次导通数量样本 → 点估计 N90_hat 与保守 N_safe（Wilson 下界 ≥ 0.90）。

    与 Q3 run_question3.n_90_estimates 同构；contacts 元素 None = n_max 内未导通。
    返回 (n_hat, n_safe)，无解时 None。
    """
    arr = np.array([v if v is not None else n_max + 1 for v in contacts])
    Ns = np.arange(1, n_max + 1, dtype=int)
    x = np.array([int(np.sum(arr <= n)) for n in Ns])
    p = x / m
    hit = np.nonzero(p >= P_TARGET)[0]
    if not len(hit):
        return None, None
    n_hat = int(Ns[hit[0]])
    los = np.array([wilson_ci(xi, m)[1] for xi in x])
    safe = np.nonzero(los >= P_TARGET)[0]
    n_safe = int(Ns[safe[0]]) if len(safe) else None
    return n_hat, n_safe


def _baseline_job(args):
    """进程池 worker：trial 区间 [t0, t1) 的首次导通样本（与搜索同 seed 语义
    rng = default_rng(base_seed + t，每 trial 独立）。kind='a' 圆柱 / 'b' 球。"""
    kind, n_max, seed, t0, t1 = args
    out = []
    for t in range(t0, t1):
        rng = np.random.default_rng(seed + t)
        if kind == 'a':
            c, u, h = geo.generate_cylinders(n_max, rng)
            out.append(fp.first_contact_n(c, u, h)['n_contact'])
        else:
            balls = gm.generate_balls(n_max, rng)
            out.append(gm.first_contact_nb(balls))
    return out


def simulate_trials_par(kind, n_max, m, seed, workers):
    """纯A/纯B 基线并行：M 次试验拆成 workers 段，段内 trial 种子连续。"""
    seg = max(1, (m + workers - 1) // workers)
    jobs = [(kind, n_max, seed, t0, min(m, t0 + seg)) for t0 in range(0, m, seg)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_baseline_job, jobs))
    return [v for part in parts for v in part]


def crossing_checks(base_seed, n=2000):
    """A/B 跨壁比例自检，对照理论 60.08% / 11.53%。"""
    rng = np.random.default_rng(base_seed)
    c, u, h = geo.generate_cylinders(n, rng)
    n_cross, n_cyl = fp.crossing_count(c, u, h)
    r_a = n_cross / n_cyl
    balls = gm.generate_balls(n, rng)
    r_b = gm.ball_crossing_ratio(balls)
    return {'ratio_a': r_a, 'ratio_b': r_b,
            'n_cross_a': n_cross, 'n_cyl_a': n_cyl}


def _eval_job(args):
    """进程池 worker：一组 (trial 批, 点批)，逐 trial 共享 prepare 全图。

    pts 任序（逐点判定，无单调假设；曲线点 b_lim(na) 随 na 递减，不可排序）。
    """
    pts, na_max, nb_max, base_seed, t0, t1 = args
    x = np.zeros(len(pts), dtype=np.int64)
    for t in range(t0, t1):
        rng = np.random.default_rng(base_seed + t)
        c, u, h = geo.generate_cylinders(na_max, rng)
        balls = gm.generate_balls(nb_max, rng)
        pr = gm.prepare_trial(c, u, h, balls, nb_max)
        for k, (na, nb) in enumerate(pts):
            if gm.sample_prefix(pr, int(na), int(nb)):
                x[k] += 1
    return x


class EvalStore:
    """共同随机数点评估仓库：evaluated dict 跨阶段复用，分级追加试验。

    data: {(na, nb): [x, m]}。追加时 trial 从现有 m 继续（trial 索引全局一致），
    保证同种子下同一候选的计数可累加。
    """

    def __init__(self, workers, na_max, nb_max, base_seed):
        self.ex = ProcessPoolExecutor(max_workers=workers)
        self.na_max = na_max
        self.nb_max = nb_max
        self.base_seed = base_seed
        self.data = {}

    def eval(self, pts, m, trial_start=0):
        """评估 pts（(n,2) 任序）m 次试验，返回 x 数组（与 pts 同序）。

        序列长度按本批点的实际最大前缀 (na_max_used, nb_max_used) 自适应：
        _eval_job 只需生成对应前缀的介质与边。共同随机数语义仍成立——
        同一 trial 内候选点共享同一前缀，跨批追加时 default_rng(base+t)
        确定性保证同 trial 生成的前缀逐位一致（少生成的尾根不影响前缀）。
        搜索层（na≈300, nb≈1）prepare 从 ~150ms 降到 ~15ms。
        """
        pts = np.asarray(pts, dtype=np.int64).reshape(-1, 2)
        n = len(pts)
        if n == 0 or m <= 0:
            return np.zeros(n, dtype=np.int64)
        na_max_used = int(pts[:, 0].max())
        nb_max_used = int(pts[:, 1].max())
        jobs = []
        for t0 in range(trial_start, trial_start + m, CHUNK_T):
            t1 = min(trial_start + m, t0 + CHUNK_T)
            for p0 in range(0, n, CHUNK_P):
                jobs.append((pts[p0:min(n, p0 + CHUNK_P)], na_max_used,
                             nb_max_used, self.base_seed, t0, t1))
        x = np.zeros(n, dtype=np.int64)
        n_pc = (n + CHUNK_P - 1) // CHUNK_P
        for k, sub in enumerate(self.ex.map(_eval_job, jobs)):
            p0 = (k % n_pc) * CHUNK_P
            x[p0:p0 + len(sub)] += sub
        return x

    def ensure(self, pts, target):
        """把 pts 中样本不足 target 的点追加到 target 次。"""
        groups = {}
        for p in pts:
            d = self.data.get(p)
            cur = d[1] if d is not None else 0
            if cur < target:
                groups.setdefault(target - cur, []).append((p, cur))
        for gap, items in groups.items():
            need = [p for p, _ in items]
            xs = self.eval(need, gap, trial_start=items[0][1])
            for (p, cur), x in zip(items, xs):
                d = self.data.get(p, [0, 0])
                self.data[p] = [d[0] + int(x), d[1] + gap]

    def get(self, p):
        d = self.data.get(p)
        return (0, 0) if d is None else (int(d[0]), int(d[1]))

    def close(self):
        self.ex.shutdown()

    def fea(self, c0, n_a_hat, n_b_hat, m_search):
        """成本层可行性：c0 内是否存在点估计可行点（p_hat ≥ 0.90）。

        na 降序分块扫描 (na, b_lim(na; c0))，早停。剪枝（不评估直接判可行）：
        - na ≥ nA_hat：纯A 曲线已达 0.90（Q3 基线点估计），共同随机数下
          Y_s(na, b) ≥ Y_s(na, 0)（前缀嵌套），故必可行；
        - b_lim(na) ≥ nB_hat：纯B 曲线已达 0.90，同理必可行。
        返回 (exists, witness_point)。剪枝点未入 evaluated dict，验证阶段
        全面核查兜底。
        """
        na_max = int(np.floor(c0 / gm.c_A))
        if na_max < 0:
            return False, None
        if na_max >= n_a_hat:
            return True, (n_a_hat, b_lim(n_a_hat, c0))
        hi = na_max
        while True:
            lo = max(0, hi - BATCH_FEA + 1)
            pts = [(na, b_lim(na, c0)) for na in range(hi, lo - 1, -1)]
            need = []
            for na, nb in pts:
                if nb >= n_b_hat:      # 纯B 剪枝：b 足够多则纯B 前缀已可行
                    return True, (na, nb)
                if nb >= 0:
                    need.append((na, nb))
            self.ensure(need, m_search)
            for na, nb in pts:
                if nb < 0:
                    continue
                x, m = self.get((na, nb))
                if x / m >= P_TARGET:
                    return True, (na, nb)
            if lo == 0:
                return False, None
            hi = lo - 1


def band_points(c_star, na_cap):
    """最优成本 C* 下全部严格更便宜的非支配组合 (na, b_lim(na; C*))。

    整除修正：成本恰等于 C* 时 b_lim 减 1（严格低于 C*）。含 na=0 的纯B 端。
    """
    pts = []
    for na in range(min(int(np.floor(c_star / gm.c_A)), na_cap) + 1):
        b = b_lim(na, c_star)
        if b < 0:
            continue
        if abs((c_star - gm.c_A * na) / gm.c_B - b) < COST_TOL:
            b -= 1
        if b >= 0:
            pts.append((na, b))
    return pts


def point_cost(p):
    """候选组合成本：c_A*na + c_B*nb。"""
    return gm.c_A * p[0] + gm.c_B * p[1]


def compute_na_max(c_ub, n_a_hat, n_a_safe, na_max_env):
    """共同随机数 A 序列长度：覆盖可靠纯A 点与复算/单调性邻点（+2）。

    静默截断改为显式报错：序列不足则 (n_safe, 0) 锚点无样本可取，
    验证兜底失效。
    """
    base = max(n_a_hat, int(np.floor(c_ub / gm.c_A)),
               n_a_safe if n_a_safe is not None else 0)
    na_max = base + 2
    if na_max - 2 > na_max_env:
        raise SystemExit(
            f'序列需 {base} 根 A（n_safe={n_a_safe}），超过 '
            f'SHUMO_Q4_NA_MAX={na_max_env}，请提高环境变量上限')
    return na_max


def anchors_for(n_a_hat, n_a_safe):
    """纯A 验证锚点：点估计最优与可靠保守点（n_safe 无解时仅前者）。"""
    pts = [(n_a_hat, 0)]
    if n_a_safe is not None:
        pts.append((n_a_safe, 0))
    return pts


def seed_from_baseline(store, contacts_a, m_base, n_a_hat, n_a_safe):
    """把纯A 基线首次导通样本折算为锚点计数预置进 store。

    基线 _baseline_job 与搜索 _eval_job 同 seed 语义（default_rng(
    base_seed + t)）同 trial 索引 → 逐 trial 生成一致。x(n) =
    count(contacts_a <= n) 即主种子下 (n,0) 的真实导通计数，后续
    ensure() 从 trial_start=m_base 续跑补样本，无重复。
    """
    arr = np.array([v if v is not None else store.na_max + 1
                    for v in contacts_a])
    for n, _ in anchors_for(n_a_hat, n_a_safe):
        x = int(np.sum(arr <= n))
        store.data[(n, 0)] = [x, m_base]


def escalate(store, pend, stage, m_target, cost_cut=None):
    """跨线点按成本升序逐批追加样本至 m_target，批间重判并剪枝。

    任一时刻只要存在成本更低的可靠点，更高成本跨线点即便可靠也不更优
    → 停止追加（保留 crossing 状态如实报告，只省样本）。cost_cut 存在时
    先按 cost < cost_cut 预剪。返回 (更新后的 stage, 最新 K)。
    """
    pend = sorted(pend, key=point_cost)
    if cost_cut is not None:
        pend = [p for p in pend if point_cost(p) < cost_cut]
    while pend:
        chunk, pend = pend[:64], pend[64:]
        store.ensure(chunk, m_target)
        for p in chunk:
            stage[p] = wilson_verdict(*store.get(p))
        b = min((p for p, v in stage.items() if v[0] == 'reliable'),
                key=point_cost, default=None)
        if b is not None:
            cost_cut = point_cost(b)
            pend = [p for p in pend if point_cost(p) < cost_cut]
    return stage, cost_cut


def verify_band(store, c_min, c_ub_eff, na_cap, m0, m_final, anchor_m,
                anchors, K_init=None):
    """主种子验证带分层推进（统计口径：Wilson 下界 ≥ 0.90 才可靠）。

    从 c_min（最低可行层成本）起逐层评估非支配边界点 (na, b_lim(na; c_layer))：
    - 每层 m0 预筛，逐点 Wilson 判定入 stage；
    - 层内 crossing 点升级到 m1 再判（K 存在时只升成本 < K 的点）；
    - 层内 reliable 点更新 K（已知最便宜可靠点成本）；
    - 停止：K ≤ c_layer（单调引理：共同随机数下 Y_s 对 N_A、N_B 双
      单调，Wilson 上界随导通计数单调不减 ⇒ 成本更低层已全数判定不足、
      成本 ≥ c_layer 的点不可能优于 K）；或 c_layer 越过纯A 成本兜底
      c_ub_eff；
    - 全程无可靠点（K=None）时用纯A 锚点兜底：先补到 m_final，仍无
      可靠点再补到 anchor_m。
    返回 (stage, v_pts, K)。stage: {p: (verdict, lo, hi)}；v_pts 评估点序。
    """
    m1 = min(m_final, max(3 * m0, 600))
    c_layer = c_min
    v_pts, stage = [], {}
    K = K_init
    while True:
        band = [p for p in band_points(c_layer, na_cap=na_cap)
                if p not in stage]
        if band:
            store.ensure(band, m0)
            for p in band:
                stage[p] = wilson_verdict(*store.get(p))
            v_pts += band
        if m1 > m0:
            cross = [p for p in band if stage[p][0] == 'crossing'
                     and (K is None or point_cost(p) < K)]
            # 大面积跨线 = 样本量不足的统计噪声而非逼近真层：升级只会
            # 把样本浪费在整层点上。仅少量跨线（≤8 个或占比 ≤1/4）升级。
            if cross and (len(cross) <= CROSS_MAX_FINE
                          or len(cross) * CROSS_FRAC_FINE <= len(band)):
                stage, K = escalate(store, cross, stage, m1, K)
        for p in band:
            if stage[p][0] == 'reliable' \
                    and (K is None or point_cost(p) < K):
                K = point_cost(p)
        if K is not None and K <= c_layer:
            break
        if c_layer >= c_ub_eff - COST_TOL:
            break
        # 步进：少量跨线 = 接近真层 → 细步进；大面积跨线（噪声）或全不足
        # → 粗步进抬层。防止小样本冒烟下整层跨线导致细步进退化逐层爬。
        n_cross = sum(1 for p in band if stage[p][0] == 'crossing')
        if n_cross and (n_cross <= CROSS_MAX_FINE
                        or n_cross * CROSS_FRAC_FINE <= len(band)):
            step = BAND_STEP
        else:
            step = BAND_STEP_COARSE
        c_layer += step * gm.c_B
    if K is None:
        for p in anchors:
            if stage.get(p, (None,))[0] != 'reliable':
                store.ensure([p], m_final)
                if p not in stage:
                    v_pts.append(p)
                stage[p] = wilson_verdict(*store.get(p))
            if stage[p][0] == 'reliable' \
                    and (K is None or point_cost(p) < K):
                K = point_cost(p)
        if K is None:
            for p in anchors:
                if stage[p][0] != 'reliable':
                    store.ensure([p], anchor_m)
                    stage[p] = wilson_verdict(*store.get(p))
                    if stage[p][0] == 'reliable' \
                            and (K is None or point_cost(p) < K):
                        K = point_cost(p)
    return stage, v_pts, K


def monotonic_check(store, na, nb, m=50):
    """单调性自检：新种子 50 次试验，逐 trial 检查 Y(na,nb) ≤ Y(na+1,nb)
    且 ≤ Y(na,nb+1)。返回违反数（0 = 通过）。"""
    violations = 0
    base_seed = store.base_seed + MONO_SEED_OFFSET
    for t in range(m):
        rng = np.random.default_rng(base_seed + t)
        c, u, h = geo.generate_cylinders(store.na_max, rng)
        balls = gm.generate_balls(store.nb_max, rng)
        pr = gm.prepare_trial(c, u, h, balls, store.nb_max)
        y00 = gm.sample_prefix(pr, na, nb)
        if not y00:
            continue
        if not gm.sample_prefix(pr, na + 1, nb):
            violations += 1
        if not gm.sample_prefix(pr, na, nb + 1):
            violations += 1
    return violations


def main():
    ap = argparse.ArgumentParser(description='问题4 混合填充最低成本搜索')
    ap.add_argument('--smoke', action='store_true', help='小样本冒烟（M 缩至 1/20 级）')
    args = ap.parse_args()

    m_search = int(os.environ.get('SHUMO_Q4_TRIALS_SEARCH', '100'))
    m_final = int(os.environ.get('SHUMO_Q4_TRIALS_FINAL', '4000'))
    m_base = int(os.environ.get('SHUMO_Q4_TRIALS_BASELINE', '1500'))
    anchor_m = int(os.environ.get('SHUMO_Q4_TRIALS_ANCHOR',
                                  str(M_ANCHOR_DEFAULT)))
    workers = int(os.environ.get(
        'SHUMO_Q4_WORKERS', str(min(8, os.cpu_count() or 1))))
    base_seed = int(os.environ.get('SHUMO_Q4_BASE_SEED', '42'))
    na_max_env = int(os.environ.get('SHUMO_Q4_NA_MAX', '900'))
    nb_max_env = int(os.environ.get('SHUMO_Q4_NB_BASELINE', '8000'))
    if args.smoke:
        # 冒烟样本须足以让统计判定收敛（m=40 下整带全跨线 → 验证带无
        # 早停退化成全带扫描）。取生产 1/4~1/6：搜索/基线/验证/锚点四级。
        m_search = max(40, m_search // 2)
        m_base, m_final = 300, 600
        anchor_m = 1500
    m0 = max(200, m_search)          # 验证首级样本
    if args.smoke:
        m0 = 100

    os.makedirs(OUT_DIR, exist_ok=True)
    t_total0 = time.perf_counter()
    print(f'问题4：c_A={gm.c_A:.6e} 元, c_B={gm.c_B:.6e} 元, '
          f'M_search={m_search}, M_final={m_final}, M_base={m_base}, '
          f'workers={workers}, base_seed={base_seed}', flush=True)

    # ---------- 1. 纯A 基线（增量早停，并行；trial 种子与搜索同口径） ----------
    contacts_a = simulate_trials_par('a', na_max_env, m_base, base_seed, workers)
    n_a_hat, n_a_safe = n90_from_contacts(contacts_a, na_max_env, m_base)
    if n_a_hat is None:
        raise SystemExit(f'纯A 基线失败：{na_max_env} 根内 p_hat < 0.90，需提高 SHUMO_Q4_NA_MAX')
    cost_a = gm.c_A * n_a_hat
    print(f'纯A 基线：N90_hat={n_a_hat}, N_safe={n_a_safe}, '
          f'成本={cost_a:.4f} 元', flush=True)

    # ---------- 2. 纯B 基线（first_contact_nb 增量，并行） ----------
    nb_probe = max(nb_max_env, int(cost_a / gm.c_B) + 2)
    contacts_b = simulate_trials_par('b', nb_probe, m_base, base_seed, workers)
    n_b_hat, n_b_safe = n90_from_contacts(contacts_b, nb_probe, m_base)
    if n_b_hat is None:
        cost_b = None
        n_b_hat_used = float('inf')      # 纯B 剪枝禁用
        print(f'纯B 基线：{nb_probe} 个球内 p_hat < 0.90（不可行，'
              f'最高 p_hat = {max(np.mean([v is not None for v in contacts_b]), 0.0):.4f}）',
              flush=True)
    else:
        cost_b = gm.c_B * n_b_hat
        n_b_hat_used = n_b_hat
        print(f'纯B 基线：N90_hat={n_b_hat}, N_safe={n_b_safe}, '
              f'成本={cost_b:.4f} 元', flush=True)

    # ---------- 3. 成本上界与序列长度 ----------
    # 搜索/复算序列只需覆盖 C_ub 边界（+2 复算邻点）；nb_probe 是纯B 可行性
    # 探测上限，不进入共同随机数序列（否则 prepare 的 B-B 边数翻倍，白白拖慢）
    c_ub = cost_a if cost_b is None else min(cost_a, cost_b)
    na_max = compute_na_max(c_ub, n_a_hat, n_a_safe, na_max_env)
    nb_req = int(np.floor(c_ub / gm.c_B)) + 2
    if n_b_hat is not None:
        nb_req = max(nb_req, n_b_hat + 2)
    nb_max = min(nb_req, nb_probe)
    print(f'成本上界 C_ub = {c_ub:.4f} 元；序列 na_max={na_max}, nb_max={nb_max}',
          flush=True)

    # ---------- 4. 共同随机数二维搜索 ----------
    store = EvalStore(workers, na_max, nb_max, base_seed)
    # 纯A 锚点预置基线样本：同种子同 trial 同构，x(n) 由首次导通样本折算，
    # 验证阶段 ensure 从 m_base 续跑补差（无重复）
    anchors = anchors_for(n_a_hat, n_a_safe)
    seed_from_baseline(store, contacts_a, m_base, n_a_hat, n_a_safe)
    try:
        # 4a. 成本二分（可行性关于 C 单调）：[0, C_ub] 收敛到 c_B 粒度。
        # 二分只用粗样本定位（点估计定位允许更大噪声，收敛后由
        # 带下降/细化用全 m_search 校正），二分成本降 ~4×
        m_coarse = max(20, m_search // 4)
        lo, hi = 0.0, c_ub
        n_fea = 0
        while hi - lo > gm.c_B:
            mid = (lo + hi) / 2.0
            ok, _ = store.fea(mid, n_a_hat, n_b_hat_used, m_coarse)
            n_fea += 1
            if ok:
                hi = mid
            else:
                lo = mid
        # 4b. 带下降：从二分结果按 8c_B 步进下探
        while store.fea(hi - BAND_STEP * gm.c_B, n_a_hat, n_b_hat_used,
                        m_search)[0]:
            hi -= BAND_STEP * gm.c_B
            n_fea += 1
        # 4c. c_B 粒度细化：窗口内从低到高找最小可行层
        c_min = hi
        c_floor = hi - BAND_STEP * gm.c_B + gm.c_B
        for c0 in np.arange(c_floor, hi, gm.c_B):
            if store.fea(float(c0), n_a_hat, n_b_hat_used, m_search)[0]:
                c_min = float(c0)
                n_fea += 1
                break
        # 4d. 最低可行层全扫：点估计最优 [PE] 与层内下界达标候选 [RELIABLE@search]
        na_full = int(np.floor(c_min / gm.c_A))
        layer = [(na, b_lim(na, c_min)) for na in range(na_full + 1)]
        layer = [(na, nb) for na, nb in layer if nb >= 0]
        store.ensure(layer, m_search)
        best = None
        best_rel_search = None
        for na, nb in layer:
            x, m = store.get((na, nb))
            v, _, _ = wilson_verdict(x, m)
            cost = point_cost((na, nb))
            if x / m >= P_TARGET \
                    and (best is None or cost < best[0]):
                best = (cost, na, nb, x, m)
            if v == 'reliable' \
                    and (best_rel_search is None
                         or cost < best_rel_search[0]):
                best_rel_search = (cost, na, nb)
        if best is None:
            raise SystemExit(f'最低可行层 C_min={c_min:.4f} 全扫无可行点，搜索失败')
        c_star, na_star, nb_star, x_star, m_star = best
        # 搜索阶段点估计必须单独保存；后续 na_star 等会更新为主种子
        # Wilson 可靠解，不能反向覆盖 CSV 中“点估计最优”的含义。
        pe_cost, pe_na, pe_nb, pe_x, pe_m = (
            c_star, na_star, nb_star, x_star, m_star)
        k_init = best_rel_search[0] if best_rel_search is not None else None
        print(f'搜索：二分 {n_fea} 次 fea，最低可行层 C_min={c_min:.4f} 元，'
              f'点估计最优 [PE] = ({na_star}, {nb_star}) 成本 {c_star:.4f} 元 '
              f'p_hat={x_star / m_star:.4f} (M={m_star})', flush=True)
        if best_rel_search is not None:
            print(f'      层内下界达标 [RELIABLE@search] = '
                  f'({best_rel_search[1]}, {best_rel_search[2]}) '
                  f'成本 {best_rel_search[0]:.4f} 元', flush=True)

        # ---------- 5. 验证带：c_min 首层起分层推进 ----------
        # 搜索层点估计定位含蒙特卡洛噪声（m_search 样本 p_hat 标准差
        # sqrt(p(1-p)/m) ≈ 0.03），C* 可能低估真实可行层。验证带从最低
        # 可行层 c_min 起逐层推进（含搜索层全部点，boundary 判定列不缺失）：
        # 每层对非支配边界点做 Wilson 判定，层内跨线升级、出现可靠点即
        # 更新 K 并提前停止（单调引理见 verify_band docstring）；全程无
        # 可靠点时纯A 锚点兜底（m_final → anchor_m）。
        c_ub_eff = min(cost_a, cost_b) if cost_b is not None else cost_a
        stage, v_pts, k_best = verify_band(
            store, c_min, c_ub_eff, na_max, m0, m_final, anchor_m,
            anchors, K_init=k_init)
        if (na_star, nb_star) not in stage:
            # 整除修正可能把搜索层点剔出 band，单独补评
            store.ensure([(na_star, nb_star)], m0)
            stage[(na_star, nb_star)] = wilson_verdict(
                *store.get((na_star, nb_star)))
            v_pts.append((na_star, nb_star))
            if stage[(na_star, nb_star)][0] == 'reliable':
                k_best = (point_cost((na_star, nb_star)) if k_best is None
                          else min(k_best, point_cost((na_star, nb_star))))
        # phase B：残余跨线点（成本低于最便宜可靠点，潜在更优）升至 M_FINAL
        m1 = min(m_final, max(3 * m0, 600))
        cross = [p for p in v_pts if stage[p][0] == 'crossing'
                 and (k_best is None or point_cost(p) < k_best)]
        if cross and m_final > m1:
            stage, k_best = escalate(store, cross, stage, m_final, k_best)
        # 保守最优（主种子口径）：验证后 Wilson 下界 ≥ 0.90 中成本最小
        cons_best = min((p for p, v in stage.items() if v[0] == 'reliable'),
                        key=point_cost, default=None)
        if cons_best is not None:
            # 验证推翻搜索层：搜索只认点估计，验证认下界。最终最优以
            # 验证为准（独立复算再复核，见阶段6）。
            na_star, nb_star = cons_best
            c_star = point_cost(cons_best)
            x_star, m_star = store.get(cons_best)
            print(f'验证更新：保守可靠点成本 {c_star:.4f} 元，'
                  f'最终最优（主种子）= ({na_star}, {nb_star})', flush=True)
        v_opt = stage.get((na_star, nb_star))
        if v_opt is None or v_opt[0] != 'reliable':
            print(f'警告：最优 ({na_star},{nb_star}) 验证判定为 '
                  f'{v_opt[0] if v_opt else "未评估"}'
                  f'（p_hat={x_star / m_star:.4f}，M={m_star}；点估计最优'
                  f'与置信约束口径不一致，真实最优高于搜索层）', flush=True)
        n_insuff = sum(1 for v in stage.values() if v[0] == 'insufficient')
        n_reli = sum(1 for v in stage.values() if v[0] == 'reliable')
        n_cross = sum(1 for v in stage.values() if v[0] == 'crossing')
        print(f'验证：{len(v_pts)} 点 M0={m0}（首层 {c_min:.4f} 元）；'
              f'不足 {n_insuff}，可靠可行 {n_reli}，跨线 {n_cross}'
              f'（跨线点升至 M={m_final} 后仍未定者如实报告）', flush=True)

        # ---------- 6. 独立复算（新种子，最终判定数据源） ----------
        # 判定集合 = 主种子最终答案 + 邻点 + 主种子非不足且成本低于最便宜
        # 可靠点的全部点（潜在更优）+ 纯A 锚点 + 纯B 端点。最终答案取
        # verify 种子下可靠可行中成本最小者；无可靠点回退主种子保守最优，
        # 再回退点估计最优（显著警告）。
        verify_seed = base_seed + VERIFY_SEED_OFFSET
        main_ans = cons_best if cons_best is not None else (na_star, nb_star)
        v2_set = {main_ans}
        for d_na, d_nb in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (main_ans[0] + d_na, main_ans[1] + d_nb)
            if p[0] >= 0 and p[1] >= 0:
                v2_set.add(p)
        cand = [p for p, v in stage.items()
                if v[0] != 'insufficient'
                and (k_best is None or point_cost(p) < k_best)]
        # K 缺失的兜底路径（验证带无可靠点）会把整带跨线点全收进来，
        # 小样本冒烟下可达数千点。答案取 reliable 中成本最小者，截掉
        # 的更高成本点不可能更优 → 成本升序取前 V2_CAP 无信息损失。
        if len(cand) > V2_CAP:
            cand = sorted(cand, key=point_cost)[:V2_CAP]
        v2_set.update(cand)
        v2_set.update(anchors)
        if n_b_hat is not None:
            v2_set.add((0, n_b_hat))
        v2_pts = list(v2_set)
        store2 = EvalStore(workers, na_max, nb_max, verify_seed)
        try:
            xs2 = store2.eval(np.asarray(v2_pts, dtype=np.int64), m_final)
            v2 = [(p, int(x), m_final) for p, x in zip(v2_pts, xs2)]
        finally:
            store2.close()
        v2_verdict = {p: wilson_verdict(x, m) for p, x, m in v2}
        cons_best_v2 = min(
            (p for p, (v, _, _) in v2_verdict.items() if v == 'reliable'),
            key=point_cost, default=None)
        print(f'独立复算（新种子，M={m_final}，最终判定）：')
        for p, x, m in v2:
            v, lo, hi = v2_verdict[p]
            tag = {'reliable': '可靠', 'insufficient': '不足',
                   'crossing': '跨线'}[v]
            mv = stage.get(p)
            ms = f'主种子={mv[0]}(M={mv[1]})' if mv else '主种子=未评估'
            print(f'  ({p[0]:4d}, {p[1]:4d}) 成本 {point_cost(p):.4f}'
                  f'  p_hat={x / m:.4f} CI=[{lo:.4f}, {hi:.4f}] {tag} {ms}')
        # 最终答案：verify 种子 reliable 中成本最小
        if cons_best_v2 is not None:
            na_fin, nb_fin = cons_best_v2
            x_fin = next(x for p, x, _ in v2 if p == cons_best_v2)
            m_fin = m_final
            src = 'verify'
            print(f'最终答案（独立种子可靠）: ({na_fin}, {nb_fin}) '
                  f'成本 {point_cost(cons_best_v2):.4f} 元', flush=True)
        elif cons_best is not None:
            na_fin, nb_fin = cons_best
            x_fin, m_fin = store.get(cons_best)
            src = 'main'
            print('警告：独立复算无可靠点，回退主种子保守最优 '
                  f'({na_fin}, {nb_fin})', flush=True)
        else:
            na_fin, nb_fin = na_star, nb_star
            x_fin, m_fin = x_star, m_star
            src = 'search'
            print('警告：独立复算与主种子均无可靠可行点，最终答案回退'
                  '点估计最优（不满足置信约束，需扩大搜索或提高样本）',
                  flush=True)

        # ---------- 7. 自检 ----------
        v_extra = monotonic_check(store, na_star, nb_star)
        cr = crossing_checks(base_seed)
        print(f'单调性：违反 {v_extra} / 100（0 为通过）')
        print(f'A 跨壁率 {cr["ratio_a"]:.4f}（理论 ~60.08%），'
              f'B 跨壁率 {cr["ratio_b"]:.4f}（理论 ~11.53%）')

        # ---------- 8. 结果汇总与 CSV（最终判定源 = 独立复算） ----------
        v_fin, lo_fin, hi_fin = wilson_verdict(x_fin, m_fin)
        # 统一口径：纯A退化情形必须由Q3严格实体内/外界复核；这里的混合
        # 几何结果只保留为搜索候选，不能覆盖严格纯A推荐。
        strict_a_audit_required = (nb_fin == 0)
        phi_a = na_fin * gm.V_A_UM3 / 1000.0 * 100.0      # 体积分数 %
        phi_b = nb_fin * gm.V_B_UM3 / 1000.0 * 100.0
        c_fin = point_cost((na_fin, nb_fin))
        res = {
            '口径': '假设二(片段独立，完整球模型)',
            'c_A_元': gm.c_A, 'c_B_元': gm.c_B,
            'M_search': m_search, 'M_final': m_final, 'M_base': m_base,
            'M_anchor': anchor_m,
            'strict_A_audit_required': strict_a_audit_required,
            'strict_A_audit_command': ('python Q4/run_unified_a_audit.py'
                                       if strict_a_audit_required else ''),
            'strict_mixed_audit_command':
                'python Q4/run_strict_mixed_audit.py',
            'N_A_hat90': n_a_hat, 'N_A_safe': n_a_safe, '纯A成本_元': cost_a,
            'N_B_hat90': n_b_hat, 'N_B_safe': n_b_safe,
            '纯B成本_元': cost_b if cost_b is not None else '不可行',
            'C_ub_元': c_ub, 'C_min层_元': c_min,
            # 最终答案（独立种子可靠口径；答案种子=verify/main/search）
            'N_A_star': na_fin, 'N_B_star': nb_fin,
            'C_star_元': c_fin,
            'phi_A_pp': phi_a, 'phi_B_pp': phi_b, 'phi_total_pp': phi_a + phi_b,
            'P_hat_star': x_fin / m_fin, 'M_star': m_fin,
            'wilson95_CI': (float(lo_fin), float(hi_fin)),
            '答案判定': v_fin, '答案种子': src,
            # 对照口径
            '点估计最优_NA': pe_na, '点估计最优_NB': pe_nb,
            '点估计最优_C_元': pe_cost, '点估计最优_p_hat': pe_x / pe_m,
            '点估计最优_M': pe_m,
            '保守最优主种子': ((cons_best[0], cons_best[1],
                             point_cost(cons_best))
                          if cons_best is not None else None),
            '验证点数': len(v_pts),
            '验证不足数': n_insuff,
            '验证可靠数': n_reli,
            '验证跨线数': n_cross,
            '纯A保守兜底_元': gm.c_A * n_a_safe
            if n_a_safe is not None else None,
            '单调性违反': v_extra,
            'A跨壁率': cr['ratio_a'], 'B跨壁率': cr['ratio_b'],
            'elapsed_s': time.perf_counter() - t_total0,
        }
        print('\n===== 问题4 结果 =====')
        print(f'最终答案（{src} 种子，M={m_fin} 判定 {v_fin}）: '
              f'N_A*={na_fin}, N_B*={nb_fin}')
        print(f'体积分数         : φ_A*={fmt_pct(phi_a / 100)}, '
              f'φ_B*={fmt_pct(phi_b / 100)}, 总计 {fmt_pct((phi_a + phi_b) / 100)}')
        print(f'最低成本 C*      : {c_fin:.6f} 元')
        print(f'导通概率(判定源) : p_hat={x_fin / m_fin:.4f}, '
              f'Wilson CI = [{lo_fin:.4f}, {hi_fin:.4f}] (M={m_fin})')
        print(f'点估计最优 [PE]  : ({pe_na}, {pe_nb}) 成本 {pe_cost:.4f} 元 '
              f'p_hat={pe_x / pe_m:.4f} (M={pe_m})')
        print(f'纯A 对照         : ({n_a_hat}, 0) 成本 {cost_a:.4f} 元')
        if cost_b is not None:
            print(f'纯B 对照         : (0, {n_b_hat}) 成本 {cost_b:.4f} 元')
        else:
            print('纯B 对照         : 不可行（球连通效率不足）')
        if cons_best is not None:
            print(f'保守最优（主种子下界≥90%）: ({cons_best[0]}, {cons_best[1]}) '
                  f'成本 {point_cost(cons_best):.4f} 元')
        print(f'验证             : {len(v_pts)} 点，不足 {n_insuff}，'
              f'可靠 {n_reli}，跨线 {n_cross}')
        print(f'总耗时           : {time.perf_counter() - t_total0:.1f}s')
        if strict_a_audit_required:
            print('统一口径提醒：当前候选的 N_B=0，最终数量须以 '
                  'Q4/run_unified_a_audit.py 的Q3严格实体复核为准。')
        print('严格混合复核：低成本临界候选请运行 '
              'Q4/run_strict_mixed_audit.py。')

        # result.csv
        with open(CSV_RESULT, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(res.keys()))
            w.writeheader()
            w.writerow(res)

        # boundary.csv：验证层全部点（含 c_min 首层）+ 两种子判定
        rows = []
        for p in v_pts:
            x, m = store.get(p)
            _, lo, hi = wilson_ci(x, m)
            v = stage.get(p)
            vv = v2_verdict.get(p)
            rows.append({'N_A': p[0], 'N_B': p[1],
                         'cost_元': point_cost(p),
                         'x': x, 'm': m, 'p_hat': x / m,
                         'ci_lower': lo, 'ci_upper': hi,
                         'point_est_feasible': bool(x / m >= P_TARGET),
                         'main_seed_verdict': v[0] if v else '',
                         'main_seed_m': m,
                         'verify_seed_verdict': vv[0] if vv else '',
                         'verify_seed_m': m_final if vv else ''})
        rows.sort(key=lambda r: (r['cost_元'], r['N_A']))
        with open(CSV_BOUNDARY, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # verify.csv：独立复算（最终判定源）+ 主种子对照
        with open(CSV_VERIFY, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['N_A', 'N_B', 'cost_元', 'x', 'm', 'p_hat',
                        'ci_lower', 'ci_upper', 'verdict',
                        'main_seed_verdict'])
            for p, x, m in v2:
                v, lo, hi = v2_verdict[p]
                mv = stage.get(p)
                w.writerow([p[0], p[1], point_cost(p),
                            x, m, x / m, lo, hi, v,
                            mv[0] if mv else ''])

        # points.csv：全部评估点（供绘图热力图）
        with open(CSV_POINTS, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['N_A', 'N_B', 'cost_元', 'x', 'm', 'p_hat',
                        'ci_lower', 'ci_upper'])
            for p, d in sorted(store.data.items()):
                x, m = d
                _, lo, hi = wilson_ci(x, m)
                w.writerow([p[0], p[1], gm.c_A * p[0] + gm.c_B * p[1],
                            x, m, x / m, lo, hi])
    finally:
        store.close()

    print(f'\n结果已写入 {CSV_RESULT} / {CSV_BOUNDARY} / {CSV_VERIFY} / {CSV_POINTS}')


if __name__ == '__main__':
    main()
