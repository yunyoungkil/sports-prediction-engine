"""
evaluation/new_hypotheses_test.py
=================================
[7.11] 신규 가설 5종 워크포워드 시험 — 결과: 5종 모두 효과 없음 (TODO 7.11, 2026-11-01 재검증)
워크포워드 5구간(14,729경기) + 축구 핸디캡 3-way(2024-07 이후). 사전 등록 없이 Gemini가 탐색한 것이라 판정은 TODO 7.11 기준.

가설 목록:
  1. 휴식기/시즌 리셋(60일 공백) 및 지수 시간 감쇠(30일/60일) 폼
  2. 직전 1경기 대승·대패(±3골 차 이상) 착시 역보정
  3. 무승부 장인(최근 10경기 무승부율 & 저득점 잠그기 성향)
  4. 축구 핸디캡 ±1.0 '핸무' 및 대형 핸디캡 가격 왜곡
  5. 홈 전용 / 원정 전용 폼(Venue-specific Split)

주의 (3번): 결과별 피처가 있으면 MarketCorrector가 아닌 직접 짠 학습 코드(규제 0.5·λ, BFGS)를 쓴다.
기준(MarketCorrector)과 바로 비교하면 학습 코드 차이(약 −0.00011)가 피처 효과처럼 보이므로,
판정은 같은 코드에 0 피처만 넣은 대조군(per_zero)과 비교한 값으로 한다.

실행:
    python -m evaluation.new_hypotheses_test
"""

import sys
from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval
from evaluation.walk_forward import FOLD_STARTS, TUNE_MONTHS, make_folds
from models.betman_corrector import GROUP_NAMES, league_group, load_archive_df
from models.betman_markets import _fair, load_segments
from models.market_corrector import FEATURE_SETS, LAMBDA_GRID, MarketCorrector, _fair_probs

LAMS = LAMBDA_GRID + [1.0]


def build_enriched_soccer_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """킥오프 시간순(동시 킥오프 누수 방지)으로 5가지 실험용 피처를 포함한 행 목록 생성."""
    hist_all = defaultdict(lambda: deque(maxlen=15))
    hist_home = defaultdict(lambda: deque(maxlen=5))
    hist_away = defaultdict(lambda: deque(maxlen=5))

    rows = []
    for kickoff, batch in df.groupby("kickoff", sort=False):
        ko_sec = kickoff.timestamp()
        for r in batch.itertuples(index=False):
            odds = [float(r.odds_win), float(r.odds_draw), float(r.odds_lose)]
            p_mkt = _fair_probs(*odds)
            h_team, a_team = r.home_team, r.away_team
            h_all, a_all = hist_all[h_team], hist_all[a_team]

            # 0. 기본 폼 (최근 5경기 단순 평균)
            h5 = list(h_all)[-5:]
            a5 = list(a_all)[-5:]
            base_pts = (sum(x[1] for x in h5) / len(h5) - sum(x[1] for x in a5) / len(a5)) if (h5 and a5) else 0.0
            base_gd = (sum(x[2] for x in h5) / len(h5) - sum(x[2] for x in a5) / len(a5)) if (h5 and a5) else 0.0

            # 1-A. 시즌 리셋 폼 (직전 경기와 60일 이상 공백 시 이전 경기 제거)
            def _season_active(seq, cur_sec, gap_days=60.0):
                if not seq:
                    return []
                out = []
                nxt_t = cur_sec
                for item in reversed(seq):
                    if (nxt_t - item[0]) / 86400.0 > gap_days:
                        break
                    out.append(item)
                    nxt_t = item[0]
                return out[:5]

            hs_rst = _season_active(h5, ko_sec, 60.0)
            as_rst = _season_active(a5, ko_sec, 60.0)
            rst_pts = (sum(x[1] for x in hs_rst) / len(hs_rst) - sum(x[1] for x in as_rst) / len(as_rst)) if (hs_rst and as_rst) else 0.0
            rst_gd = (sum(x[2] for x in hs_rst) / len(hs_rst) - sum(x[2] for x in as_rst) / len(as_rst)) if (hs_rst and as_rst) else 0.0

            # 1-B. 지수 시간 감쇠 폼 (half-life tau = 30d, 60d)
            def _exp_form(seq, cur_sec, tau_days):
                if not seq:
                    return None
                ws = [np.exp(-max(0.0, (cur_sec - x[0]) / 86400.0) / tau_days) for x in seq]
                w_sum = sum(ws)
                if w_sum <= 1e-9:
                    return None
                return (
                    sum(w * x[1] for w, x in zip(ws, seq)) / w_sum,
                    sum(w * x[2] for w, x in zip(ws, seq)) / w_sum,
                )

            he30, ae30 = _exp_form(h5, ko_sec, 30.0), _exp_form(a5, ko_sec, 30.0)
            exp30_pts = (he30[0] - ae30[0]) if (he30 and ae30) else 0.0
            exp30_gd = (he30[1] - ae30[1]) if (he30 and ae30) else 0.0

            he60, ae60 = _exp_form(h5, ko_sec, 60.0), _exp_form(a5, ko_sec, 60.0)
            exp60_pts = (he60[0] - ae60[0]) if (he60 and ae60) else 0.0
            exp60_gd = (he60[1] - ae60[1]) if (he60 and ae60) else 0.0

            # 2. 직전 1경기 대승/대패(±3골 이상) 과잉반응 피처
            h_last_gd = h5[-1][2] if h5 else 0
            a_last_gd = a5[-1][2] if a5 else 0
            last1_gd_diff = float(h_last_gd - a_last_gd) if (h5 and a5) else 0.0
            h_blow_win = 1.0 if h_last_gd >= 3 else 0.0
            a_blow_win = 1.0 if a_last_gd >= 3 else 0.0
            h_blow_loss = 1.0 if h_last_gd <= -3 else 0.0
            a_blow_loss = 1.0 if a_last_gd <= -3 else 0.0
            blow_win_diff = h_blow_win - a_blow_win
            blow_loss_diff = h_blow_loss - a_blow_loss

            # 3. 무승부 장인 성향 (최근 10경기 무승부율 & 저득점)
            h10 = list(h_all)[-10:]
            a10 = list(a_all)[-10:]
            if len(h10) >= 5 and len(a10) >= 5:
                h_dr = sum(x[4] for x in h10) / len(h10)
                a_dr = sum(x[4] for x in a10) / len(a10)
                draw_rate_avg = 0.5 * (h_dr + a_dr) - 0.25
                tg_avg = 0.5 * (sum(x[3] for x in h10) / len(h10) + sum(x[3] for x in a10) / len(a10)) - 2.65
                raw_dr_avg = 0.5 * (h_dr + a_dr)
                raw_tg_avg = 0.5 * (sum(x[3] for x in h10) / len(h10) + sum(x[3] for x in a10) / len(a10))
            else:
                draw_rate_avg = 0.0
                tg_avg = 0.0
                raw_dr_avg = None
                raw_tg_avg = None

            # 5. 홈 전용 / 원정 전용 폼
            hh5 = list(hist_home[h_team])
            aa5 = list(hist_away[a_team])
            if len(hh5) >= 2 and len(aa5) >= 2:
                venue_pts_diff = sum(x[0] for x in hh5) / len(hh5) - sum(x[0] for x in aa5) / len(aa5)
                venue_gd_diff = sum(x[1] for x in hh5) / len(hh5) - sum(x[1] for x in aa5) / len(aa5)
            else:
                venue_pts_diff = base_pts
                venue_gd_diff = base_gd

            g = league_group(r.league)
            feats = {
                "form_pts_diff": base_pts,
                "form_gd_diff": base_gd,
                "rst_pts_diff": rst_pts,
                "rst_gd_diff": rst_gd,
                "exp30_pts_diff": exp30_pts,
                "exp30_gd_diff": exp30_gd,
                "exp60_pts_diff": exp60_pts,
                "exp60_gd_diff": exp60_gd,
                "last1_gd_diff": last1_gd_diff,
                "blow_win_diff": blow_win_diff,
                "blow_loss_diff": blow_loss_diff,
                "venue_pts_diff": venue_pts_diff,
                "venue_gd_diff": venue_gd_diff,
            }
            for name in GROUP_NAMES:
                feats[f"g_{name}"] = 1.0 if g == name else 0.0

            hs, as_ = int(r.home_score), int(r.away_score)
            y = 0 if hs > as_ else (1 if hs == as_ else 2)
            row = {
                "gm_ts": int(r.gm_ts),
                "match_seq": int(r.match_seq),
                "kickoff": kickoff,
                "date": kickoff.strftime("%Y-%m-%d"),
                "league": r.league,
                "group": g,
                "home_team": h_team,
                "away_team": a_team,
                "odds": np.array(odds, dtype=float),
                "p_market": p_mkt,
                "p_avg": None,
                "p_sharp": None,
                "actual": y,
                "y": y,
                "per_draw_rate": np.array([0.0, draw_rate_avg, 0.0]),
                "per_low_goals": np.array([0.0, -tg_avg, 0.0]),
                "per_zero": np.zeros(3),   # 3번 대조군: 정보 없는 피처 (같은 학습 코드끼리 비교)
                "raw_dr_avg": raw_dr_avg,
                "raw_tg_avg": raw_tg_avg,
                "h_last_gd": h_last_gd if h5 else None,
                "a_last_gd": a_last_gd if a5 else None,
            }
            row.update(feats)
            rows.append(row)

        for r in batch.itertuples(index=False):
            hs, as_ = int(r.home_score), int(r.away_score)
            hp, ap = (3, 0) if hs > as_ else ((1, 1) if hs == as_ else (0, 3))
            is_d = 1.0 if hs == as_ else 0.0
            tg = float(hs + as_)
            hist_all[r.home_team].append((ko_sec, hp, hs - as_, tg, is_d))
            hist_all[r.away_team].append((ko_sec, ap, as_ - hs, tg, is_d))
            hist_home[r.home_team].append((hp, hs - as_))
            hist_away[r.away_team].append((ap, as_ - hs))

    return rows


def eval_custom_wf(rows: List[Dict[str, Any]], folds: List[Dict[str, Any]], match_feats: List[str], extra_per_keys=(),
                   per_outcome=("log_p_market",)) -> Dict[str, Any]:
    """5구간 워크포워드 평가. per_outcome은 MarketCorrector 결과별 피처 이름 (예: 'x_diff' ← 행의 p_x, 7.12 경기 시각)"""
    g_cols = [f"g_{g}" for g in GROUP_NAMES]

    if not extra_per_keys:
        fs_name = "_tmp_exp"
        FEATURE_SETS[fs_name] = {
            "per_outcome": list(per_outcome),
            "match": list(match_feats) + g_cols,
        }
        all_rows, all_probs = [], []
        fold_lls = []
        for f in folds:
            tr, te = f["train"], f["test"]
            cut = pd.Timestamp(f["start"], tz="UTC") - pd.DateOffset(months=TUNE_MONTHS)
            in_tr = [r for r in tr if r["kickoff"] < cut]
            in_tu = [r for r in tr if r["kickoff"] >= cut]
            best_lam, best_ll = 1.0, float("inf")
            for lam in LAMS:
                m = MarketCorrector(feature_set=fs_name, lam=lam).fit(in_tr)
                p_tu = m.predict_rows(in_tu)
                y_tu = np.array([r["actual"] for r in in_tu])
                ll = float(-np.mean(np.log(np.clip(p_tu[np.arange(len(y_tu)), y_tu], 1e-15, None))))
                if ll < best_ll:
                    best_ll, best_lam = ll, lam
            m = MarketCorrector(feature_set=fs_name, lam=best_lam).fit(tr)
            p = m.predict_rows(te)
            y = np.array([r["actual"] for r in te])
            ll_te = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))
            fold_lls.append(ll_te)
            all_rows.extend(te)
            all_probs.append(p)
        probs = np.vstack(all_probs)
        y_all = np.array([r["y"] for r in all_rows])
        losses = -np.log(np.clip(probs[np.arange(len(y_all)), y_all], 1e-12, 1.0))
        return {"losses": losses, "fold_lls": fold_lls, "rows": all_rows, "probs": probs}

    # per-outcome 추가 피처(무승부 성향 등)가 있는 경우
    all_rows, all_probs = [], []
    fold_lls = []
    m_cols = list(match_feats) + g_cols

    def _fit_and_pred(tr_rows, eval_rows, lam):
        n_per = 1 + len(extra_per_keys)
        n_m = len(m_cols)

        def _matrices(rs):
            N = len(rs)
            log_p = np.log(np.clip(np.array([r["p_market"] for r in rs]), 1e-12, 1.0))
            P_extra = [np.array([r[k] for r in rs]) for k in extra_per_keys]
            M = np.array([[r[c] for c in m_cols] for r in rs], dtype=float) if n_m else np.zeros((N, 0))
            y = np.array([r["y"] for r in rs], dtype=int)
            return log_p, P_extra, M, y

        lp_tr, pe_tr, M_tr, y_tr = _matrices(tr_rows)
        mu = M_tr.mean(axis=0) if n_m else np.zeros(0)
        sd = np.where(M_tr.std(axis=0) > 1e-8, M_tr.std(axis=0), 1.0) if n_m else np.ones(0)
        Mz_tr = (M_tr - mu) / sd if n_m else M_tr

        dim = n_per + 2 * n_m + 2
        w0 = np.zeros(dim)

        def _logits(w, lp, pe, Mz):
            beta = w[0]
            z = (1.0 + beta) * lp
            for idx, arr in enumerate(pe):
                z = z + w[1 + idx] * arr
            wh = w[n_per:n_per + n_m]
            wa = w[n_per + n_m:n_per + 2 * n_m]
            ch, ca = w[-2], w[-1]
            z[:, 0] += Mz @ wh + ch
            z[:, 2] += Mz @ wa + ca
            z -= z.max(axis=1, keepdims=True)
            exp_z = np.exp(z)
            return exp_z / exp_z.sum(axis=1, keepdims=True)

        def _loss(w):
            p = _logits(w, lp_tr, pe_tr, Mz_tr)
            nll = -np.mean(np.log(np.clip(p[np.arange(len(y_tr)), y_tr], 1e-12, 1.0)))
            reg = 0.5 * lam * np.sum(w[:-2] ** 2)
            return nll + reg

        res = minimize(_loss, w0, method="BFGS", options={"maxiter": 200})
        lp_ev, pe_ev, M_ev, _ = _matrices(eval_rows)
        Mz_ev = (M_ev - mu) / sd if n_m else M_ev
        return _logits(res.x, lp_ev, pe_ev, Mz_ev)

    for f in folds:
        tr, te = f["train"], f["test"]
        cut = pd.Timestamp(f["start"], tz="UTC") - pd.DateOffset(months=TUNE_MONTHS)
        in_tr = [r for r in tr if r["kickoff"] < cut]
        in_tu = [r for r in tr if r["kickoff"] >= cut]
        y_tu = np.array([r["y"] for r in in_tu])
        best_lam, best_ll = 1.0, float("inf")
        for lam in LAMS:
            p_tu = _fit_and_pred(in_tr, in_tu, lam)
            ll = float(-np.mean(np.log(np.clip(p_tu[np.arange(len(y_tu)), y_tu], 1e-12, 1.0))))
            if ll < best_ll:
                best_ll, best_lam = ll, lam
        p_te = _fit_and_pred(tr, te, best_lam)
        y_te = np.array([r["y"] for r in te])
        fold_lls.append(float(-np.mean(np.log(np.clip(p_te[np.arange(len(y_te)), y_te], 1e-12, 1.0)))))
        all_rows.extend(te)
        all_probs.append(p_te)

    probs = np.vstack(all_probs)
    y_all = np.array([r["y"] for r in all_rows])
    losses = -np.log(np.clip(probs[np.arange(len(y_all)), y_all], 1e-12, 1.0))
    return {"losses": losses, "fold_lls": fold_lls, "rows": all_rows, "probs": probs}


def compare_to_base(name: str, cand: Dict[str, Any], base: Dict[str, Any], ref: str = "base"):
    diff = cand["losses"] - base["losses"]
    clusters = [r["gm_ts"] for r in base["rows"]]
    lo, hi = cluster_bootstrap_ci(diff.tolist(), clusters, n_bootstrap=1000)
    mean_d = float(np.mean(diff))
    wins = sum(1 for c_ll, b_ll in zip(cand["fold_lls"], base["fold_lls"]) if c_ll < b_ll)
    print(f"  {name:<26}: vs {ref} {mean_d:+.5f} [{lo:+.5f}, {hi:+.5f}] | 구간 승 {wins}/5 | 구간별 {[round(c - b, 5) for c, b in zip(cand['fold_lls'], base['fold_lls'])]}")


def run_all():
    print("=" * 72)
    print("🧪 [신규 가설 5종 워크포워드 테스트] 데이터 로딩 중...")
    df = load_archive_df()
    rows = build_enriched_soccer_rows(df)
    folds = make_folds(rows, FOLD_STARTS)
    print(f"전체 아카이브: {len(rows):,}경기 | 워크포워드 5구간 평가: {sum(len(f['test']) for f in folds):,}경기")

    base = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff"])
    mkt_losses = np.array([-np.log(max(r["p_market"][r["y"]], 1e-12)) for r in base["rows"]])
    b_vs_m = base["losses"] - mkt_losses
    m_lo, m_hi = cluster_bootstrap_ci(b_vs_m.tolist(), [r["gm_ts"] for r in base["rows"]], n_bootstrap=1000)
    m_d = float(np.mean(b_vs_m))
    print(f"  기준(base) vs 시장         : {m_d:+.5f} [{m_lo:+.5f}, {m_hi:+.5f}]")

    print("\n" + "=" * 72)
    print("1️⃣ [테스트 1] 휴식기/시즌 리셋(60일 공백) 및 지수 시간 감쇠(30일/60일) 폼")
    c_rst = eval_custom_wf(rows, folds, ["rst_pts_diff", "rst_gd_diff"])
    compare_to_base("season_reset_60d (교체)", c_rst, base)
    c_e30 = eval_custom_wf(rows, folds, ["exp30_pts_diff", "exp30_gd_diff"])
    compare_to_base("exp_decay_30d (교체)", c_e30, base)
    c_e60 = eval_custom_wf(rows, folds, ["exp60_pts_diff", "exp60_gd_diff"])
    compare_to_base("exp_decay_60d (교체)", c_e60, base)

    print("\n" + "=" * 72)
    print("2️⃣ [테스트 2] 직전 1경기 대승·대패(±3골 차 이상) 착시 역보정 (Blowout Overreaction)")
    c_last1 = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff", "last1_gd_diff"])
    compare_to_base("+last1_gd_diff", c_last1, base)
    c_blow = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff", "blow_win_diff", "blow_loss_diff"])
    compare_to_base("+blowout_win_loss_3g", c_blow, base)

    te_rows = base["rows"]
    hw_blow = [r for r in te_rows if r["h_last_gd"] is not None and r["h_last_gd"] >= 3]
    aw_blow = [r for r in te_rows if r["a_last_gd"] is not None and r["a_last_gd"] >= 3]
    hl_blow = [r for r in te_rows if r["h_last_gd"] is not None and r["h_last_gd"] <= -3]
    print(f"  [직접 집계] 직전 3골+ 대승 홈팀(n={len(hw_blow)}): 시장 예상 승률 {np.mean([r['p_market'][0] for r in hw_blow])*100:.1f}% vs 실제 승률 {np.mean([1 if r['y']==0 else 0 for r in hw_blow])*100:.1f}% (단통 ROI {(np.mean([r['odds'][0] if r['y']==0 else 0.0 for r in hw_blow])-1)*100:+.1f}%)")
    print(f"  [직접 집계] 직전 3골+ 대승 원정팀(n={len(aw_blow)}): 시장 예상 승률 {np.mean([r['p_market'][2] for r in aw_blow])*100:.1f}% vs 실제 승률 {np.mean([1 if r['y']==2 else 0 for r in aw_blow])*100:.1f}% (단통 ROI {(np.mean([r['odds'][2] if r['y']==2 else 0.0 for r in aw_blow])-1)*100:+.1f}%)")
    print(f"  [직접 집계] 직전 3골+ 대패 홈팀(n={len(hl_blow)}): 시장 예상 승률 {np.mean([r['p_market'][0] for r in hl_blow])*100:.1f}% vs 실제 승률 {np.mean([1 if r['y']==0 else 0 for r in hl_blow])*100:.1f}% (단통 ROI {(np.mean([r['odds'][0] if r['y']==0 else 0.0 for r in hl_blow])-1)*100:+.1f}%)")

    print("\n" + "=" * 72)
    print("3️⃣ [테스트 3] '무승부 장인(최근 10경기 무승부율 & 저득점 잠그기)' 성향 테스트")
    c_ctrl = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff"], extra_per_keys=("per_zero",))
    compare_to_base("대조군 (같은 코드 + 0 피처)", c_ctrl, base)
    c_dr = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff"], extra_per_keys=("per_draw_rate",))
    compare_to_base("+per_draw_rate (10경기)", c_dr, base)
    compare_to_base("+per_draw_rate (10경기)", c_dr, c_ctrl, ref="대조군")
    c_dr_tg = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff"], extra_per_keys=("per_draw_rate", "per_low_goals"))
    compare_to_base("+draw_rate + low_goals", c_dr_tg, base)
    compare_to_base("+draw_rate + low_goals", c_dr_tg, c_ctrl, ref="대조군")
    print("  → 판정은 '대조군' 줄로 한다 (base와의 차이는 학습 코드 차이가 섞임)")

    dr_high = [r for r in te_rows if r["raw_dr_avg"] is not None and r["raw_dr_avg"] >= 0.35]
    dr_lock = [r for r in te_rows if r["raw_dr_avg"] is not None and r["raw_dr_avg"] >= 0.35 and r["raw_tg_avg"] <= 2.3]
    all_draw_roi = (np.mean([r["odds"][1] if r["y"] == 1 else 0.0 for r in te_rows]) - 1.0) * 100
    print(f"  [직접 집계] 전체 경기 무승부(n={len(te_rows)}): 시장 예상 {np.mean([r['p_market'][1] for r in te_rows])*100:.1f}% vs 실제 {np.mean([1 if r['y']==1 else 0 for r in te_rows])*100:.1f}% | ROI {all_draw_roi:+.1f}%")
    if dr_high:
        rets = [(r["odds"][1] - 1.0) if r["y"] == 1 else -1.0 for r in dr_high]
        roi, lo, hi = cluster_roi_confidence_interval(rets, [r["gm_ts"] for r in dr_high])
        print(f"  [직접 집계] 양팀 무승부율 ≥35%(n={len(dr_high)}): 시장 예상 {np.mean([r['p_market'][1] for r in dr_high])*100:.1f}% vs 실제 {np.mean([1 if r['y']==1 else 0 for r in dr_high])*100:.1f}% | ROI {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%]")
    if dr_lock:
        rets = [(r["odds"][1] - 1.0) if r["y"] == 1 else -1.0 for r in dr_lock]
        roi, lo, hi = cluster_roi_confidence_interval(rets, [r["gm_ts"] for r in dr_lock])
        print(f"  [직접 집계] 무승부율 ≥35% & 평균골 ≤2.3(n={len(dr_lock)}): 시장 예상 {np.mean([r['p_market'][1] for r in dr_lock])*100:.1f}% vs 실제 {np.mean([1 if r['y']==1 else 0 for r in dr_lock])*100:.1f}% | ROI {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%]")

    print("\n" + "=" * 72)
    print("4️⃣ [테스트 4] 축구 핸디캡 ±1.0 '핸무(정확히 1골 차 승리)' 가격 왜곡 분석")
    segs = load_segments()
    h_df = segs.get("축구 핸디캡|3")
    if h_df is not None:
        h_df = h_df[h_df["kickoff"] >= pd.Timestamp("2024-07-01", tz="UTC")]
        h_minus1 = h_df[np.isclose(h_df["line"], -1.0)]
        h_plus1 = h_df[np.isclose(h_df["line"], 1.0)]
        h_big = h_df[h_df["line"].abs() >= 2.0]
        for label, sub in [("홈 -1.0 핸디캡 (홈 강팀)", h_minus1), ("홈 +1.0 핸디캡 (원정 강팀)", h_plus1), ("|기준선| ≥ 2.0 (대형 핸디캡)", h_big)]:
            if sub.empty:
                continue
            odds_cols = ["odds_win", "odds_draw", "odds_lose"]
            odds_s = sub[odds_cols].to_numpy(dtype=float)
            p_m = _fair(odds_s)
            y_s = sub["result"].to_numpy(dtype=int)
            clusters = sub["gm_ts"].astype(str).tolist()
            print(f"  [{label}] n={len(sub):,}경기:")
            for idx, nm in enumerate(["핸승", "핸무(정확히 기준차)", "핸패"]):
                exp_p = np.mean(p_m[:, idx]) * 100
                act_p = np.mean(y_s == idx) * 100
                rets = [(odds_s[i, idx] - 1.0) if y_s[i] == idx else -1.0 for i in range(len(sub))]
                roi, lo, hi = cluster_roi_confidence_interval(rets, clusters)
                print(f"    · {nm:<18}: 시장 예상 {exp_p:5.1f}% vs 실제 {act_p:5.1f}% (차이 {act_p - exp_p:+.2f}%p) | 단통 ROI {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%]")
    else:
        print("  축구 핸디캡|3 세그먼트 데이터 없음")

    print("\n" + "=" * 72)
    print("5️⃣ [테스트 5] 홈 전용 / 원정 전용 강팀(Venue-specific Form Split) 테스트")
    c_v_add = eval_custom_wf(rows, folds, ["form_pts_diff", "form_gd_diff", "venue_pts_diff", "venue_gd_diff"])
    compare_to_base("+venue_form (통합+홈원정)", c_v_add, base)
    c_v_only = eval_custom_wf(rows, folds, ["venue_pts_diff", "venue_gd_diff"])
    compare_to_base("venue_only (홈원정 폼만)", c_v_only, base)


if __name__ == "__main__":
    run_all()

