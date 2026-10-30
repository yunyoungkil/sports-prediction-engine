"""
evaluation/scorecard.py
=======================
[Phase 5.3] AI 성적표 — 경기 전에 저장한 예측(prediction_log)만 실제 결과로 채점합니다.
학습·튜닝에 전혀 쓰이지 않은 "진짜 앞으로의 성적"이라 가장 정직한 지표입니다.

- Log loss: AI(실전 확률) vs 베트맨 시장(마진 제거), 경기별 차이의 부트스트랩 95% CI
- ROI: AI 확률 기준 EV > 2%인 결과에 베트맨 배당으로 1단위씩 걸었을 때
- 실행할 때마다 data/model_scorecard.json 의 history 에 날짜별로 한 줄 누적
- [5.11] 축구 승무패 외 유형(핸디캡·언더오버·농구·야구 등)은 베트맨 정답 코드(game_result)로 채점, 유형별 내역 포함
- [6.5] 신뢰구간은 회차 단위 부트스트랩 (같은 회차 경기끼리는 독립이 아닐 수 있음)
- [6.5] 사전 등록 가설(HYPOTHESES): 등록 이후 킥오프한 경기로만 판정하고, 기준은 결과를 보고 바꾸지 않는다
- [6.6] 보정 곡선(calibration): AI·배당이 '○○%'라고 한 선택지들이 실제로 몇 % 일어났는지 10%p 구간별로

실행:
    python -m evaluation.scorecard
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff
from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval
from models.league_groups import league_group_v2
from tracker.prediction_log import LOG_DIR

SCORECARD_PATH = Path("data/model_scorecard.json")
KEYS = ("home_win", "draw", "away_win")
EV_THRESHOLD = 0.02
MIN_SAMPLES_FOR_VERDICT = 300

# [6.5] 사전 등록 가설 (2026-10-27 등록). 워크포워드에서 AI 확률 우위가 컸던 두 그룹이 앞으로도 그런지 —
# 등록 다음 날(2026-10-28 00:00 KST) 이후 킥오프한 축구 승무패만, 미리 정한 기준으로 판정한다
HYPOTHESES_SINCE = datetime(2026, 10, 27, 15, 0, tzinfo=timezone.utc)
HYPOTHESES = [
    {"id": "H1", "name": "국가대표 경기에서 AI 확률 우위가 계속된다", "group": "national"},
    {"id": "H2", "name": "UEFA 대회에서 AI 확률 우위가 계속된다", "group": "uefa_cup"},
]
HYPOTHESIS_MIN, HYPOTHESIS_MAX = 300, 600


def _vec(probs: Dict[str, float]) -> np.ndarray:
    v = np.array([float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0) for k in KEYS])
    return v / v.sum() if v.sum() > 0 else v


def load_graded_results(path: Path = Path("data/history/betman_recent_results.json")) -> Dict[tuple, int]:
    """(회차, 경기번호) → 정답 0/1/2. 베트맨 정답 코드(game_result) 우선, 없으면 축구 승무패만 스코어로 계산"""
    res: Dict[tuple, int] = {}
    for it in json.loads(path.read_text(encoding="utf-8")):
        key = (int(it["gm_ts"]), str(it["match_seq"]))
        code = "" if it.get("game_result") is None else str(it.get("game_result"))
        if code in ("0", "1", "2"):
            res[key] = int(code)
        elif it.get("game_type") == "축구 승무패" and it.get("home_score") is not None and it.get("away_score") is not None:
            hs, as_ = it["home_score"], it["away_score"]
            res[key] = 0 if hs > as_ else (1 if hs == as_ else 2)
    return res


def load_graded_by_line(path: Path = Path("data/history/betman_recent_results.json")) -> Dict[tuple, List[tuple]]:
    """(회차, 경기번호) → [(기준값 handi_win, 정답 0/1/2), …]. 핸디캡·언더오버는 같은 경기번호에 기준값별 결과 줄이 여럿 (REVIEW R19)"""
    out: Dict[tuple, List[tuple]] = {}
    for it in json.loads(path.read_text(encoding="utf-8")):
        key = (int(it["gm_ts"]), str(it["match_seq"]))
        code = "" if it.get("game_result") is None else str(it.get("game_result"))
        y = None
        if code in ("0", "1", "2"):
            y = int(code)
        elif it.get("game_type") == "축구 승무패" and it.get("home_score") is not None and it.get("away_score") is not None:
            hs, as_ = it["home_score"], it["away_score"]
            y = 0 if hs > as_ else (1 if hs == as_ else 2)
        if y is not None:
            out.setdefault(key, []).append((float(it.get("handi_win") or 0.0), y))
    return out


def pick_result(cands: Optional[List[tuple]], line: Any) -> Optional[int]:
    """결과 줄이 하나면 그대로, 여럿이면 예측 로그의 기준값과 같은 줄만. 맞는 줄이 없으면 추측하지 않고 None (채점 제외)"""
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][1]
    ln = float(line or 0.0)
    for handi, y in cands:
        if abs(handi - ln) < 1e-9:
            return y
    return None


def _ll_diffs(rows: List[Dict[str, Any]]) -> np.ndarray:
    """경기별 Log loss 차이 (AI − 시장, 음수 = AI가 더 정확)"""
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(y))
    return (-np.log(np.clip(np.array([r["model"] for r in rows])[idx, y], 1e-15, None))
            + np.log(np.clip(np.array([r["market"] for r in rows])[idx, y], 1e-15, None)))


def _round_ci(values, rows: List[Dict[str, Any]], digits: int = 4) -> List[float]:
    lo, hi = cluster_bootstrap_ci(values, [r["gm_ts"] for r in rows])
    return [round(lo, digits), round(hi, digits)]


def _ll_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    d = _ll_diffs(rows)
    return {"n": len(rows), "ll_diff": round(float(d.mean()), 4), "ll_diff_ci": _round_ci(d, rows)}


def _ll_block_fixtures(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """_ll_block + 서로 다른 대진 수와 대진 단위 부트스트랩 범위 (같은 대진의 여러 게임번호는 독립이 아님 — REVIEW R20)"""
    blk = _ll_block(rows)
    d = _ll_diffs(rows)
    lo, hi = cluster_bootstrap_ci(d, [r["fixture"] for r in rows])
    blk.update({"fixtures": len({r["fixture"] for r in rows}), "ll_diff_ci_fixture": [round(lo, 4), round(hi, 4)]})
    return blk


def _value_bets(rows: List[Dict[str, Any]], threshold: float) -> List[tuple]:
    """AI 확률 기준 EV가 threshold를 넘는 최고 선택지에 1단위 → [(순손익 비율, 회차)]"""
    out = []
    for r in rows:
        ev = r["model"] * r["odds"] - 1.0
        k = int(ev.argmax())
        if ev[k] > threshold:
            out.append((r["odds"][k] - 1.0 if k == r["y"] else -1.0, r["gm_ts"]))
    return out


def calibration(rows: List[Dict[str, Any]], key: str, n_bins: int = 10) -> List[Dict[str, Any]]:
    """[6.6] 보정 곡선: '○○%'라고 말한 선택지들이 실제로 몇 % 일어났나 (10%p 구간, 발매된 선택지만)"""
    said, happened = [], []
    for r in rows:
        for k in range(3):
            if r["odds"][k] > 1.0:          # 2지선다 유형의 빈 무승부 칸은 제외
                said.append(float(r[key][k]))
                happened.append(1.0 if k == r["y"] else 0.0)
    said_a, happened_a = np.array(said), np.array(happened)
    out = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (said_a >= lo) & ((said_a < hi) if i < n_bins - 1 else (said_a <= hi))
        if m.any():
            out.append({"range": f"{int(lo * 100)}~{int(hi * 100)}%", "n": int(m.sum()),
                        "said_pct": round(float(said_a[m].mean()) * 100, 1),
                        "happened_pct": round(float(happened_a[m].mean()) * 100, 1)})
    return out


def hypothesis_verdict(n: int, ll_diff: float, ci: List[float]) -> str:
    """미리 정한 판정 (2026-10-27): 300경기부터 판정, 600경기에서도 확인 안 되면 중단"""
    if n < HYPOTHESIS_MIN:
        return f"진행 중 ({n}/{HYPOTHESIS_MIN}경기)"
    if ci[1] < 0:
        return "재현됨 (CI 상한 < 0) → 수익 검증 단계로"
    if ll_diff >= 0:
        return "사라짐 (AI가 더 정확하지 않음) → 이 가설 중단"
    if n >= HYPOTHESIS_MAX:
        return f"{HYPOTHESIS_MAX}경기에서도 확인 안 됨 → 이 가설 중단"
    return f"판정 보류 (평균은 AI 우세, CI가 0 포함 · {HYPOTHESIS_MAX}경기까지 계속)"


def build_hypotheses(rows: List[Dict[str, Any]], since: datetime = HYPOTHESES_SINCE) -> List[Dict[str, Any]]:
    out = []
    for h in HYPOTHESES:
        sub = [r for r in rows if r["game_type"] == "축구 승무패" and league_group_v2(r["league"]) == h["group"]
               and r["kickoff"] is not None and r["kickoff"] >= since]
        item = {**h, "since": since.isoformat(), "n": len(sub)}
        if sub:
            blk = _ll_block(sub)
            bets = _value_bets(sub, EV_THRESHOLD)
            roi, lo, hi = cluster_roi_confidence_interval([b[0] for b in bets], [b[1] for b in bets])
            item.update({"ll_diff": blk["ll_diff"], "ll_diff_ci": blk["ll_diff_ci"],
                         "value_bets": len(bets), "roi_pct": roi, "roi_ci": [lo, hi],
                         "verdict": hypothesis_verdict(len(sub), blk["ll_diff"], blk["ll_diff_ci"])})
        else:
            item.update({"ll_diff": None, "ll_diff_ci": None, "value_bets": 0, "roi_pct": None, "roi_ci": None,
                         "verdict": hypothesis_verdict(0, 0.0, [0.0, 0.0])})
        out.append(item)
    return out


SHADOW_MIN_N = 100   # 이만큼 쌓이기 전에는 '나쁨'으로 판단하지 않음 (models/shadow_models.py 교체 규칙)


def shadow_compare(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """그림자 모델마다: 같은 경기에서 (그림자 − 지금 AI) Log loss 차이 · 회차 단위 범위. 음수면 그림자가 더 정확"""
    out: Dict[str, Any] = {}
    for name in sorted({k for r in rows for k in r.get("shadow", {})}):
        sub = [r for r in rows if name in r.get("shadow", {})]
        y = np.array([r["y"] for r in sub])
        i = np.arange(len(y))
        ps, pm = np.array([r["shadow"][name] for r in sub]), np.array([r["model"] for r in sub])
        d = -np.log(np.clip(ps[i, y], 1e-15, None)) + np.log(np.clip(pm[i, y], 1e-15, None))
        ci = _round_ci(d, sub, digits=5)
        if len(sub) < SHADOW_MIN_N:
            verdict = f"모으는 중 ({len(sub)}/{SHADOW_MIN_N}경기)"
        elif ci[0] > 0:
            verdict = "지금 AI보다 확실히 나쁨 → 교체하지 않음"
        elif ci[1] < 0:
            verdict = "지금 AI보다 확실히 좋음"
        else:
            verdict = "차이가 확실하지 않음 (과거 시험 결과대로 300경기 뒤 교체)"
        out[name] = {"n": len(sub), "ll_diff_vs_model": round(float(d.mean()), 5), "ci": ci, "verdict": verdict}
    return out


def build_scorecard(log_dir: Path = LOG_DIR, results_path: Path = Path("data/history/betman_recent_results.json")) -> Dict[str, Any]:
    results = load_graded_by_line(results_path)
    rows: List[Dict[str, Any]] = []
    for f in sorted(log_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        gm_ts = int(data.get("gm_ts"))
        for seq, e in data.get("matches", {}).items():
            y = pick_result(results.get((gm_ts, str(seq))), e.get("line"))
            if y is None or not e.get("model_probs") or not e.get("market_probs"):
                continue
            home, away = e.get("home_team"), e.get("away_team")
            rows.append({"gm_ts": gm_ts, "y": y,
                         # 같은 대진의 승무패·핸디캡 등은 결과가 같이 움직이는 한 묶음 (REVIEW R20)
                         "fixture": f"{gm_ts}|{home}|{away}" if home and away else f"{gm_ts}|{seq}", "model": _vec(e["model_probs"]), "market": _vec(e["market_probs"]),
                         "odds": np.array([float(e["betman_odds"][k]) for k in KEYS]),
                         "source": e.get("prob_source"), "game_type": e.get("game_type") or "축구 승무패",
                         "league": e.get("league") or "", "kickoff": parse_betman_kickoff(e.get("match_time")),
                         "shadow": {k: _vec(v) for k, v in (e.get("shadow_probs") or {}).items() if v}})

    card: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "graded_matches": len(rows), "rounds": sorted({r["gm_ts"] for r in rows}),
                            "ci_method": "회차 단위 부트스트랩"}
    if not rows:
        card["verdict"] = "채점할 경기 없음 (경기 전 예측 기록이 결과와 아직 만나지 않음)"
        card["hypotheses"] = build_hypotheses(rows)
        return card

    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(y))
    pm = np.array([r["model"] for r in rows])
    pk = np.array([r["market"] for r in rows])
    nll_m = -np.log(np.clip(pm[idx, y], 1e-15, None))
    nll_k = -np.log(np.clip(pk[idx, y], 1e-15, None))
    d = nll_m - nll_k

    bets = _value_bets(rows, EV_THRESHOLD)
    roi, lo, hi = cluster_roi_confidence_interval([b[0] for b in bets], [b[1] for b in bets])
    # 💎 경기 (대시보드 표시와 같은 기준: AI 확률 기준 EV > 0 인 최고 EV 선택지에 1단위)
    gems = _value_bets(rows, 0.0)
    g_roi, g_lo, g_hi = cluster_roi_confidence_interval([b[0] for b in gems], [b[1] for b in gems])

    card.update({
        "model_ll": round(float(nll_m.mean()), 4),
        "market_ll": round(float(nll_k.mean()), 4),
        "ll_diff": round(float(d.mean()), 4),
        "ll_diff_ci": _round_ci(d, rows),
        "hit_rate_top_pick": round(float((pm.argmax(axis=1) == y).mean() * 100), 1),
        "value_bets": len(bets), "roi_pct": roi, "roi_ci_low": lo, "roi_ci_high": hi,
        "gem_bets": len(gems), "gem_hits": sum(1 for g in gems if g[0] > 0),
        "gem_roi_pct": g_roi, "gem_roi_ci_low": g_lo, "gem_roi_ci_high": g_hi,
        "prob_sources": {s: sum(1 for r in rows if r["source"] == s) for s in {r["source"] for r in rows}},
        # 유형별 LL 차이 (어느 유형에서 AI가 이기고 지는지)
        "by_game_type": {gt: _ll_block_fixtures([r for r in rows if r["game_type"] == gt])
                         for gt in sorted({r["game_type"] for r in rows})},
        "fixtures": len({r["fixture"] for r in rows}),   # '300경기'는 게임번호 기준 — 독립적인 대진 수를 옆에 병기
        "hypotheses": build_hypotheses(rows),
        # [Phase 7.1] 그림자 모델 vs 지금 AI — 같은 실전 경기만 (판정에는 안 씀 · 300경기 뒤 교체 결정용)
        "shadow": shadow_compare(rows),
        # [6.6] 보정 곡선 — AI와 배당이 각각 말한 확률 vs 실제로 일어난 비율
        "calibration": {"ai": calibration(rows, "model"), "market": calibration(rows, "market")},
    })
    if len(rows) < MIN_SAMPLES_FOR_VERDICT:
        card["verdict"] = f"검증 중 (표본 {len(rows)}/{MIN_SAMPLES_FOR_VERDICT})"
    elif card["ll_diff_ci"][1] < 0:
        card["verdict"] = "AI가 시장보다 확실히 정확함 (CI 상한 < 0)"
    elif card["ll_diff"] < 0:
        card["verdict"] = "AI가 시장보다 약간 정확하지만 확실하지 않음 (CI가 0 포함)"
    else:
        card["verdict"] = "AI가 시장보다 정확하지 않음"
    return card


def update_scorecard(path: Path = SCORECARD_PATH) -> Dict[str, Any]:
    card = build_scorecard()
    try:
        model_report = json.loads(Path("data/market_corrector.json").read_text(encoding="utf-8")).get("report", {})
    except Exception:
        model_report = {}
    card["model"] = {
        "trained_at": model_report.get("trained_at"),
        "trained_on": model_report.get("production_trained_on"),
        "passes_gate": model_report.get("passes_gate"),
        "backtest_test": model_report.get("test"),
        "backtest_betman": model_report.get("betman"),
    }

    history: List[Dict[str, Any]] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            history = []
    today = card["updated_at"][:10]
    entry = {"date": today, **{k: card.get(k) for k in ("graded_matches", "model_ll", "market_ll", "ll_diff", "ll_diff_ci", "roi_pct", "gem_bets", "gem_roi_pct")}}
    history = [h for h in history if h.get("date") != today] + [entry]
    card["history"] = history
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    c = update_scorecard()
    print("=" * 70)
    print("📋 [Phase 5.3] AI 성적표 (경기 전 예측만 채점)")
    print("=" * 70)
    print(f"채점 경기 {c['graded_matches']}건 | 회차 {c['rounds']}")
    if c.get("model_ll") is not None:
        print(f"LL: AI {c['model_ll']} vs 시장 {c['market_ll']} (차이 {c['ll_diff']:+.4f}, CI {c['ll_diff_ci']})")
        print(f"1순위 적중률 {c['hit_rate_top_pick']}% | EV>{EV_THRESHOLD:.0%} 베팅 {c['value_bets']}건 ROI {c['roi_pct']:+.2f}% [{c['roi_ci_low']:+.2f}, {c['roi_ci_high']:+.2f}]")
        print(f"💎 EV>0 경기 {c['gem_bets']}건 (적중 {c['gem_hits']}) ROI {c['gem_roi_pct']:+.2f}% [{c['gem_roi_ci_low']:+.2f}, {c['gem_roi_ci_high']:+.2f}]")
    print(f"판정: {c['verdict']}")
    for h in c.get("hypotheses", []):
        print(f"📌 {h['id']} {h['name']}: {h['n']}경기 · {h['verdict']}")
