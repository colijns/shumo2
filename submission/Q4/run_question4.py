# 本程序及代码是在AI工具辅助下完成的
# AI工具名称：DeepSeek‑V4‑Flash，版本 / 型号：DeepSeek‑V4‑Flash‑0731，开发机构 / 公司：深度求索（DeepSeek），版本发布日期：2026‑07‑31


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
import geometry as geo
import geometry_mix as gm
import first_passage as fp
from monte_carlo import wilson_ci

P_TARGET = 0.90
VERIFY_SEED_OFFSET = 12345
MONO_SEED_OFFSET = 500000
M_ANCHOR_DEFAULT = 8000
CHUNK_T = 32
CHUNK_P = 32
BATCH_FEA = 48
BAND_STEP = 8
BAND_STEP_COARSE = 32
CROSS_FRAC_FINE = 4
CROSS_MAX_FINE = 8
V2_CAP = 200
COST_TOL = 1e-9

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CSV_RESULT = os.path.join(OUT_DIR, 'question4_result.csv')
CSV_BOUNDARY = os.path.join(OUT_DIR, 'question4_boundary.csv')
CSV_VERIFY = os.path.join(OUT_DIR, 'question4_verify.csv')
CSV_POINTS = os.path.join(OUT_DIR, 'question4_points.csv')

def fmt_pct(v):
    return f'{v * 100:.2f}%'

def b_lim(na, c0):
    return int(np.floor((c0 - gm.c_A * na) / gm.c_B + COST_TOL))

def wilson_verdict(x, m):
    _, lo, hi = wilson_ci(int(x), int(m))
    if hi < P_TARGET:
        return 'insufficient', lo, hi
    if lo >= P_TARGET:
        return 'reliable', lo, hi
    return 'crossing', lo, hi

def n90_from_contacts(contacts, n_max, m):
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
    seg = max(1, (m + workers - 1) // workers)
    jobs = [(kind, n_max, seed, t0, min(m, t0 + seg)) for t0 in range(0, m, seg)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_baseline_job, jobs))
    return [v for part in parts for v in part]

def crossing_checks(base_seed, n=2000):
    rng = np.random.default_rng(base_seed)
    c, u, h = geo.generate_cylinders(n, rng)
    n_cross, n_cyl = fp.crossing_count(c, u, h)
    r_a = n_cross / n_cyl
    balls = gm.generate_balls(n, rng)
    r_b = gm.ball_crossing_ratio(balls)
    return {'ratio_a': r_a, 'ratio_b': r_b,
            'n_cross_a': n_cross, 'n_cyl_a': n_cyl}

def _eval_job(args):
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

    def __init__(self, workers, na_max, nb_max, base_seed):
        self.ex = ProcessPoolExecutor(max_workers=workers)
        self.na_max = na_max
        self.nb_max = nb_max
        self.base_seed = base_seed
        self.data = {}

    def eval(self, pts, m, trial_start=0):
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
                if nb >= n_b_hat:
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
    return gm.c_A * p[0] + gm.c_B * p[1]

def compute_na_max(c_ub, n_a_hat, n_a_safe, na_max_env):
    base = max(n_a_hat, int(np.floor(c_ub / gm.c_A)),
               n_a_safe if n_a_safe is not None else 0)
    na_max = base + 2
    if na_max - 2 > na_max_env:
        raise SystemExit(
            f'序列需 {base} 根 A（n_safe={n_a_safe}），超过 '
            f'SHUMO_Q4_NA_MAX={na_max_env}，请提高环境变量上限')
    return na_max

def anchors_for(n_a_hat, n_a_safe):
    pts = [(n_a_hat, 0)]
    if n_a_safe is not None:
        pts.append((n_a_safe, 0))
    return pts

def seed_from_baseline(store, contacts_a, m_base, n_a_hat, n_a_safe):
    arr = np.array([v if v is not None else store.na_max + 1
                    for v in contacts_a])
    for n, _ in anchors_for(n_a_hat, n_a_safe):
        x = int(np.sum(arr <= n))
        store.data[(n, 0)] = [x, m_base]

def escalate(store, pend, stage, m_target, cost_cut=None):
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

        m_search = max(40, m_search // 2)
        m_base, m_final = 300, 600
        anchor_m = 1500
    m0 = max(200, m_search)
    if args.smoke:
        m0 = 100

    os.makedirs(OUT_DIR, exist_ok=True)
    t_total0 = time.perf_counter()
    print(f'问题4：c_A={gm.c_A:.6e} 元, c_B={gm.c_B:.6e} 元, '
          f'M_search={m_search}, M_final={m_final}, M_base={m_base}, '
          f'workers={workers}, base_seed={base_seed}', flush=True)

    contacts_a = simulate_trials_par('a', na_max_env, m_base, base_seed, workers)
    n_a_hat, n_a_safe = n90_from_contacts(contacts_a, na_max_env, m_base)
    if n_a_hat is None:
        raise SystemExit(f'纯A 基线失败：{na_max_env} 根内 p_hat < 0.90，需提高 SHUMO_Q4_NA_MAX')
    cost_a = gm.c_A * n_a_hat
    print(f'纯A 基线：N90_hat={n_a_hat}, N_safe={n_a_safe}, '
          f'成本={cost_a:.4f} 元', flush=True)

    nb_probe = max(nb_max_env, int(cost_a / gm.c_B) + 2)
    contacts_b = simulate_trials_par('b', nb_probe, m_base, base_seed, workers)
    n_b_hat, n_b_safe = n90_from_contacts(contacts_b, nb_probe, m_base)
    if n_b_hat is None:
        cost_b = None
        n_b_hat_used = float('inf')
        print(f'纯B 基线：{nb_probe} 个球内 p_hat < 0.90（不可行，'
              f'最高 p_hat = {max(np.mean([v is not None for v in contacts_b]), 0.0):.4f}）',
              flush=True)
    else:
        cost_b = gm.c_B * n_b_hat
        n_b_hat_used = n_b_hat
        print(f'纯B 基线：N90_hat={n_b_hat}, N_safe={n_b_safe}, '
              f'成本={cost_b:.4f} 元', flush=True)

    c_ub = cost_a if cost_b is None else min(cost_a, cost_b)
    na_max = compute_na_max(c_ub, n_a_hat, n_a_safe, na_max_env)
    nb_req = int(np.floor(c_ub / gm.c_B)) + 2
    if n_b_hat is not None:
        nb_req = max(nb_req, n_b_hat + 2)
    nb_max = min(nb_req, nb_probe)
    print(f'成本上界 C_ub = {c_ub:.4f} 元；序列 na_max={na_max}, nb_max={nb_max}',
          flush=True)

    store = EvalStore(workers, na_max, nb_max, base_seed)

    anchors = anchors_for(n_a_hat, n_a_safe)
    seed_from_baseline(store, contacts_a, m_base, n_a_hat, n_a_safe)
    try:

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

        while store.fea(hi - BAND_STEP * gm.c_B, n_a_hat, n_b_hat_used,
                        m_search)[0]:
            hi -= BAND_STEP * gm.c_B
            n_fea += 1

        c_min = hi
        c_floor = hi - BAND_STEP * gm.c_B + gm.c_B
        for c0 in np.arange(c_floor, hi, gm.c_B):
            if store.fea(float(c0), n_a_hat, n_b_hat_used, m_search)[0]:
                c_min = float(c0)
                n_fea += 1
                break

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

        c_ub_eff = min(cost_a, cost_b) if cost_b is not None else cost_a
        stage, v_pts, k_best = verify_band(
            store, c_min, c_ub_eff, na_max, m0, m_final, anchor_m,
            anchors, K_init=k_init)
        if (na_star, nb_star) not in stage:

            store.ensure([(na_star, nb_star)], m0)
            stage[(na_star, nb_star)] = wilson_verdict(
                *store.get((na_star, nb_star)))
            v_pts.append((na_star, nb_star))
            if stage[(na_star, nb_star)][0] == 'reliable':
                k_best = (point_cost((na_star, nb_star)) if k_best is None
                          else min(k_best, point_cost((na_star, nb_star))))

        m1 = min(m_final, max(3 * m0, 600))
        cross = [p for p in v_pts if stage[p][0] == 'crossing'
                 and (k_best is None or point_cost(p) < k_best)]
        if cross and m_final > m1:
            stage, k_best = escalate(store, cross, stage, m_final, k_best)

        cons_best = min((p for p, v in stage.items() if v[0] == 'reliable'),
                        key=point_cost, default=None)
        if cons_best is not None:

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

        v_extra = monotonic_check(store, na_star, nb_star)
        cr = crossing_checks(base_seed)
        print(f'单调性：违反 {v_extra} / 100（0 为通过）')
        print(f'A 跨壁率 {cr["ratio_a"]:.4f}（理论 ~60.08%），'
              f'B 跨壁率 {cr["ratio_b"]:.4f}（理论 ~11.53%）')

        v_fin, lo_fin, hi_fin = wilson_verdict(x_fin, m_fin)

        strict_a_audit_required = (nb_fin == 0)
        phi_a = na_fin * gm.V_A_UM3 / 1000.0 * 100.0
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

            'N_A_star': na_fin, 'N_B_star': nb_fin,
            'C_star_元': c_fin,
            'phi_A_pp': phi_a, 'phi_B_pp': phi_b, 'phi_total_pp': phi_a + phi_b,
            'P_hat_star': x_fin / m_fin, 'M_star': m_fin,
            'wilson95_CI': (float(lo_fin), float(hi_fin)),
            '答案判定': v_fin, '答案种子': src,

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
              'Q4/run_strict_mixed_audit.py；只检验8个含B候选，'
              '纯A直接引用Q3，固定A序列长度为750，'
              '未完成大样本前不得把探索结果作为最终结论。'
              '可加 --status 仅查看检查点。')

        with open(CSV_RESULT, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(res.keys()))
            w.writeheader()
            w.writerow(res)

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
