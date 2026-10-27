"""
dashboard/app.py
================
Streamlit 기반 스포츠 승부 예측 & 가치 픽(Value Bet) 실시간 웹 대시보드
[Tab 1: 🛒 맞춤조건구매 (STEP 01 맞춤조건설정 & STEP 02 조합결과보기 시뮬레이터)]
[Tab 2: 🔍 AI 경기 분석 (조회 전용 - 배당 확률 vs AI 확률, 기대가치, 최근 폼)]
[Tab 3: 📊 성과 검증 (실제 경기 결과 1:1 대조, 누적 ROI, 95% 신뢰구간 & CLV 통계)]
"""

import sys
import os
import itertools
from pathlib import Path
import json
import re
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# 프로젝트 루트 경로를 sys.path에 추가 (dashboard 하위 폴더에서 실행 시 models/tracker 패키지 인식)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib

# Streamlit 세션 유지 중 코드 변경 시 서브모듈 핫 리로딩 보장 — 프로젝트 패키지는 전부 함께 지운다.
# 일부만 지우면 새로 읽은 모듈이 메모리에 남은 옛 모듈을 import해서 ImportError가 난다
# (2026-10-27: 새 models.paper_trading → 옛 evaluation.metrics, cluster_roi_confidence_interval 없음)
_PROJECT_PKGS = ("models", "tracker", "evaluation", "collector", "notifier")
for mod_name in list(sys.modules.keys()):
    if mod_name.split(".")[0] in _PROJECT_PKGS:
        del sys.modules[mod_name]

importlib.invalidate_caches()

try:
    from models.combo_builder import (
        calculate_combined_odds,
        calculate_payout,
        is_tax_free,
        validate_combo,
        expand_combinations
    )
except ImportError:
    import models.combo_builder as _cb
    importlib.reload(_cb)
    calculate_combined_odds = getattr(_cb, "calculate_combined_odds")
    calculate_payout = getattr(_cb, "calculate_payout")
    validate_combo = getattr(_cb, "validate_combo")
    expand_combinations = getattr(_cb, "expand_combinations")
    if hasattr(_cb, "is_tax_free"):
        is_tax_free = getattr(_cb, "is_tax_free")
    else:
        def is_tax_free(odds: float, payout: int) -> bool:
            if payout <= 100_000:
                return True
            if odds <= 100.0 and payout <= 2_000_000:
                return True
            return False

from models.strategy_backtest import backtest, describe_rule, extract_rule, outcome_kind
from models.combo_finder import candidate_legs, find_combos
from collector.betman_sharp import parse_betman_kickoff
from models.paper_trading import STRATEGIES as STRATEGY_LABELS, BACKFILL_LABELS as BACKFILL_LABEL_MAP
from models.betman_markets import SPORT_NAMES
from models.form_analyzer import KOREAN_NAME
from tracker.my_slips import (KIND_LABEL, MAX_SAVED_COMBOS, add_slip, build_slip, combo_results, delete_slip, display_score,
                              draw_is_void, grade_slip, line_row, load_json, load_slips, outcome_from_rows, save_json,
                              slip_kind, update_slip)

# 내 선택 저장 위치 (개인 기록이라 저장소에 올리지 않음 · .gitignore). 테스트는 SPE_USER_DATA_DIR 로 다른 폴더를 씀
MY_DATA_DIR = Path(os.environ.get("SPE_USER_DATA_DIR") or (project_root / "user_data"))
MY_CURRENT_PATH = MY_DATA_DIR / "current_selections.json"
MY_SLIPS_PATH = MY_DATA_DIR / "my_slips.json"

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="스포츠 승부예측 & 맞춤조건구매 대시보드",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto"   # PC에서는 🛒 내 선택 사이드바를 펼쳐 두고, 휴대폰에서는 접어 둠
)

# 2. 커스텀 CSS 스타일링 (번역 방지 및 반응형 배지 UI, 라이트/다크 테마 호환)
st.markdown("""
<meta name="google" content="notranslate">
<style>
    .notranslate { translate: no !important; }
    .step-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    /* 버튼 묶음(날짜·빠른 필터·배당 선택)이 휴대폰에서 옆으로 잘리지 않고 다음 줄로 넘어가도록 */
    .stButtonGroup > div { flex-wrap: wrap; row-gap: 6px; }
</style>
""", unsafe_allow_html=True)


# 3. 데이터 로딩 헬퍼
@st.cache_data(ttl=60)
def load_data():
    pred_path = project_root / "data" / "predictions.json"
    if not pred_path.exists():
        return []
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


predictions = load_data()


@st.cache_data(ttl=3600)
def load_archive_for_backtest():
    from models.betman_corrector import load_archive_df
    return load_archive_df()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest(rule_key: tuple) -> dict:
    """((kind, ref_odds), ...) 축만으로 된 규칙의 과거 반복 결과 (조합 찾기 후보용)"""
    legs = [{"type": "single", "kind": k, "ref_odds": o} for k, o in rule_key]
    return backtest(legs, load_archive_for_backtest())


@st.cache_data(ttl=60)
def load_model_status():
    """
    현재 실전 확률을 만드는 AI의 검증 결과 (run_pipeline.py와 같은 우선순위).
    베트맨 학습 보정(5.6) → 유럽 학습 보정(5.1) → 없음(시장 단독)
    """
    for path, name, keys in [
        (project_root / "data" / "betman_corrector.json", "베트맨 학습 보정 AI",
         [("test", "TEST 2026-01~10 (베트맨)"), ("holdout", "최근 실전 (베트맨)")]),
        (project_root / "data" / "market_corrector.json", "유럽 학습 보정 AI",
         [("test", "TEST 2025/26 (유럽)"), ("betman", "베트맨 실전")]),
    ]:
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8")).get("report", {})
        if rep.get("passes_gate"):
            return {"name": name, "trained_at": rep.get("trained_at"),
                    "evals": [(label, rep[k]) for k, label in keys if k in rep]}
    return None


@st.cache_data(ttl=55, show_spinner=False)
def fetch_live_round_results(gm_ts: str) -> list:
    """베트맨 결과 페이지에서 해당 회차의 끝난 경기 결과를 바로 조회 (대시보드 실시간 표시용, 저장하지 않음)"""
    try:
        from collector.betman_results_scraper import BetmanResultsScraper
        return BetmanResultsScraper().fetch_round_results(int(gm_ts))
    except Exception:
        return []


def started_without_result(preds, results_idx, gm_ts) -> bool:
    """이미 시작했는데 결과가 아직 없는 경기가 있는지 (있을 때만 실시간 조회)"""
    now = datetime.now().astimezone()
    for m in preds:
        ko = parse_betman_kickoff(m.get("match_time"))
        if ko and ko <= now and not results_idx.get((str(gm_ts), str(m.get("match_seq")))):
            return True
    return False


@st.cache_data(ttl=60)
def load_paper_backfill():
    path = project_root / "data" / "paper_trading_backfill.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_paper_trading():
    path = project_root / "data" / "paper_trading.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_markets_model():
    """[5.11] 전 종목·전 유형 보정 모델의 유형(세그먼트)별 검증 결과"""
    path = project_root / "data" / "betman_markets.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_clv():
    path = project_root / "data" / "clv_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def markets_applied_types() -> list:
    mm = load_markets_model() or {}
    return sorted(k.split("|")[0] for k, v in mm.get("segments", {}).items() if v.get("passes_gate"))


@st.cache_data(ttl=60)
def load_scorecard():
    path = project_root / "data" / "model_scorecard.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_market_scan():
    """[6.4·6.6] 유형별 마진·AI 우위 표 + 배당 구간별 실제 환급률 (매주 재학습 때 갱신)"""
    path = project_root / "data" / "market_scan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_stale_odds():
    """[6.2] 지연 배당 측정 (결과 갱신 때 자동 갱신)"""
    path = project_root / "data" / "stale_odds_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# 3-1. 경기 결과 · 경기 전 AI 예측 기록 (맞춤조건구매 목록의 적중 여부 표시용)
PICK_KEYS = ("home_win", "draw", "away_win")
SEL_KEY = {"home_win": "home", "draw": "draw", "away_win": "away"}
RESULT_CODE_TO_KEY = {"0": "home_win", "1": "draw", "2": "away_win"}   # 2-way(언더/오버, 홀/짝)도 0=앞, 2=뒤
AI_SOURCES = ("betman_corrector", "market_corrector", "betman_markets")


@st.cache_data(ttl=60)
def load_recommendation_status():
    """[Phase 6.5] 추천 잠금 상태 — 검증 3단계(과거 · 실시간 확률 · 수익)를 통과한 기능만 추천 표시 (evaluation/gates.py)"""
    from evaluation.gates import recommendation_status
    d = project_root / "data"
    return recommendation_status(d / "walk_forward.json", d / "model_scorecard.json", d / "paper_trading.json")


REC_STATUS = load_recommendation_status()
GEM_ON = REC_STATUS["features"]["gem"]["on"]          # 💎 표시 · 가치 픽 · 💎 필터/빠른 선택
FINDER_ON = REC_STATUS["features"]["finder"]["on"]    # 🧩 조합 찾기
LOCK_NOTE = "검증 3단계 통과 전 · ③ 성과 검증 탭"


def lock_notice(feature: dict) -> None:
    """잠긴 기능 자리에: 왜 잠겼는지와 단계별 진행 상황 (쉬운 말)"""
    items = [(s["ok"], s["name"], s.get("plain", s["detail"])) for s in REC_STATUS["stages"]]
    p = feature["profit"]
    items.append((p["ok"], p["name"], p.get("plain", p["detail"])))
    st.info("🔒 **추천 잠금 중** — AI가 아래 시험 3개를 모두 통과해야 이 기능이 켜져요. "
            "그동안에도 AI는 뒤에서 계속 예측하고 채점받아요.\n\n"
            + "\n".join(f"- {'✅' if ok else '⏳'} **{name}** — {text}" for ok, name, text in items))


# ------------------------------------------------------------------
# [6.6] 쉬운 말 화면 조각 — 성과 탭에서 "그래서 어떻게 판단하나"
# ------------------------------------------------------------------
def won(pct: float) -> str:
    """환급률 % → 1만 원당 돌아오는 돈"""
    return f"{pct * 100:,.0f}원"


def bucket_summary(scan):
    """축구 승무패 배당대별 환급률에서 정배(1.30 이하)와 고배당(5.0 이상) 요약"""
    rows = ((scan or {}).get("buckets") or {}).get("축구 승무패") or []
    low = [b for b in rows if b["range"] in ("1.15 이하", "1.15~1.30")]
    high = [b for b in rows if b["range"] == "5.0 이상"]
    if not low or not high:
        return None
    n = sum(b["n"] for b in low)
    return {"low": sum(b["return_pct"] * b["n"] for b in low) / n, "high": high[0]["return_pct"]}


def eta_text(card, target: int = 300) -> str:
    """성적표 history의 쌓이는 속도로 목표까지 남은 기간 (근거가 부족하면 빈 문자열)"""
    hist = [h for h in (card or {}).get("history", []) if h.get("graded_matches")]
    n = (card or {}).get("graded_matches") or 0
    if n >= target or len(hist) < 2:
        return ""
    days = (datetime.fromisoformat(hist[-1]["date"]) - datetime.fromisoformat(hist[0]["date"])).days
    gained = hist[-1]["graded_matches"] - hist[0]["graded_matches"]
    if days < 3 or gained <= 0:
        return ""
    return f" 지금 속도면 약 {max(1, round((target - n) / (gained / days) / 7))}주 뒤예요."


def render_conclusion(card, scan, stale) -> None:
    """🧭 지금 결론 — 질문 3개에 쉬운 말로 답"""
    s1 = REC_STATUS["stages"][0]
    on = [f["short"] for f in REC_STATUS["features"].values() if f["on"]]
    if on:
        a1 = (f"**시험을 통과한 기능만 켜져 있어요:** {', '.join(on)}. 그래도 매번 이긴다는 뜻은 아니에요. "
              "오래 했을 때 평균이 이익이라는 뜻이에요.")
    else:
        a1 = ("**아직 아니에요 (🔒 추천 잠금).** "
              + ("AI가 배당보다 확률을 *아주 조금* 더 잘 맞히는 건 과거 시험에서 확인됐어요. 하지만 그 차이가 베트맨이 떼어가는 몫보다 "
                 "훨씬 작아서 돈으로는 이어지지 않아요. " if s1["ok"] else "AI가 배당보다 낫다는 게 아직 확인되지 않았어요. ")
              + "그래서 💎·추천 픽·조합 찾기를 보여 주지 않아요.")
    bs = bucket_summary(scan)
    a2 = ((f"**배당 1.3 이하 정배**에 걸면 1만 원당 평균 약 **{won(bs['low'])}**, **5배 이상 고배당**은 약 **{won(bs['high'])}**이 "
           "돌아왔어요 (축구 승무패, 2024년 7월 이후 실제 결과). 고배당일수록, 여러 경기를 묶을수록 떼이는 몫이 커져요. "
           "어느 쪽이든 평균적으로는 잃는 구조라, 잃어도 괜찮은 금액만 쓰세요.")
          if bs else "배당대별 표가 아직 없어요 (매주 재학습 때 만들어져요).")
    n = (card or {}).get("graded_matches") or 0
    a3 = (f"AI 실전 채점이 **{n:,} / 300경기**예요. " + ("300경기가 모이면 1차 판정이 나와요." + eta_text(card)
                                                    if n < 300 else "판정이 나왔어요 — 아래 🔒 표를 보세요."))
    if stale and stale.get("value"):
        a3 += (f" '베트맨이 해외 배당을 늦게 따라가는지' 확인은 **{stale['value'].get('matches', 0)} / 100경기**예요"
               f" (지금까지: {stale.get('verdict', '-')}).")
    with st.container(border=True):
        st.markdown("#### 🧭 지금 결론 — 이것만 보면 돼요")
        st.markdown(f"**Q1. AI 추천대로 사도 되나요?**  \n{a1}")
        st.markdown(f"**Q2. 그래도 산다면 뭐가 덜 손해예요?**  \n{a2}")
        st.markdown(f"**Q3. 다음엔 언제 다시 보면 되나요?**  \n{a3}")
        st.caption("볼 때 순서: ① 이 상자에서 결론 확인 → ② 아래 🔒 표에서 AI 시험이 얼마나 진행됐는지 확인 → "
                   "③ 사고 싶은 경기가 있으면 💰 표에서 덜 손해 보는 배당대 확인")


def render_glossary() -> None:
    with st.expander("📖 용어 쉽게 보기"):
        st.markdown(
            "- **배당 확률**: 배당을 '일어날 확률'로 바꾼 것(베트맨 몫은 뺀 값). 배당 2.0이면 대략 45~50%예요.\n"
            "- **AI 확률**: 배당 확률을 과거 경기 결과로 조금 고친 값이에요.\n"
            "- **정배 / 역배**: 이길 가능성이 높은 쪽(낮은 배당) / 낮은 쪽(높은 배당).\n"
            "- **환급률 / 마진**: 1만 원 걸면 평균 돌아오는 돈 / 베트맨이 떼어가는 몫. 환급률 90%면 9,000원이 돌아오고 1,000원은 베트맨 몫이에요.\n"
            "- **기대가치 (EV)**: 똑같은 걸 아주 많이 샀을 때의 평균 손익. +면 이득, −면 손해예요.\n"
            "- **정확도 점수 (Log loss)**: 확률이 실제 결과와 얼마나 맞았는지의 점수. **낮을수록 좋아요.** "
            "'AI − 배당'이 마이너스면 AI가 더 잘 맞힌 거예요.\n"
            "- **95% 범위 (운 감안 범위)**: 운까지 생각했을 때 진짜 값이 있을 만한 범위예요. 범위 **전체가** 한쪽(0보다 작거나 큼)에 "
            "있어야 '확실'이고, 0에 걸쳐 있으면 '아직 모름'이에요.\n"
            "- **과거 시험 (워크포워드)**: 지난 경기들을 시간 순서대로, '그때까지 알던 정보만으로' 다시 맞혀 본 시험이에요.\n"
            "- **가상 베팅**: 실제 돈 없이 '이렇게 샀다면'을 기록만 하는 거예요.\n"
            "- **추천 잠금 (섀도 모드)**: AI는 뒤에서 계속 예측하고 채점받지만, 시험 3개를 통과하기 전에는 추천을 보여 주지 않는 상태예요.\n"
            "- **사전 가설**: 결과를 보기 전에 미리 적어 둔 예상이에요. 나중에 결과에 맞춰 말을 바꾸지 않으려고 미리 적어요.\n"
            "- **회차 단위**: 같은 회차 경기끼리는 비슷하게 움직일 수 있어서, 회차를 한 덩어리로 보고 계산한 거예요. "
            "회차가 적을 때는 범위가 너무 좁게 나올 수 있어서 경기 하나씩 본 범위와 비교해 넓은 쪽을 써요.\n"
            "- **왜 300경기?**: 축구는 운이 커서 몇십 경기로는 실력과 운을 구분할 수 없어요. 그래서 기준을 미리 정해 뒀어요.\n"
            "- **마감 배당 비교**: AI가 '배당보다 높다'고 본 쪽 배당이 마감 때 내려갔다면, 다른 사람들의 돈이 나중에 그쪽으로 몰렸다는 "
            "뜻이에요 (AI가 먼저 맞게 봤다는 신호).\n"
            "- **적중률**: 참고용이에요. 배당 1순위만 골라도 꽤 맞아요. 돈이 되는지는 **돌아온 돈(환급률·수익률)**으로 봐야 해요.")


def render_margin_tables(scan) -> None:
    """💰 베트맨이 떼어가는 몫 — 배당대별 실제 환급률 + 유형별 표"""
    st.subheader("💰 베트맨이 떼어가는 몫")
    if not scan:
        st.info("아직 표가 없어요 (매주 재학습 때 만들어져요).")
        return
    st.caption("1만 원을 걸었을 때 평균적으로 **실제로 돌아온 돈**이에요. 1만 원보다 적은 만큼이 평균 손해예요. "
               f"정배(낮은 배당)일수록 덜 떼고, 고배당일수록 많이 떼요. 갱신: {str(scan.get('updated_at', '-'))[:10]}")
    buckets = scan.get("buckets") or {}
    for col, (name, rows) in zip(st.columns(max(1, len(buckets))), buckets.items()):
        with col:
            st.markdown(f"**배당대별 — {name}** (2024년 7월 이후)")
            st.dataframe(pd.DataFrame([{
                "배당": b["range"], "1만 원당 돌아온 돈": won(b["return_pct"]),
                "운 감안 범위": f"{won(b['return_ci'][0])} ~ {won(b['return_ci'][1])}",
                "적중률": f"{b['hit_pct']:.0f}%", "기록 수": f"{b['n']:,}"} for b in rows]),
                hide_index=True, width="stretch")
    plain = {"AI 우위 없음": "차이 없음", "AI 우위 있음 · 마진 못 넘음": "조금 나음 (그래도 손해)",
             "마진 넘김 후보": "이익 가능성 (확인 필요)"}
    seg_rows = []
    for r in scan.get("rows", []):
        gt, k = r["segment"].split("|")
        m = r.get("margin_pct")
        seg_rows.append({
            "유형": gt + (" (무 있음)" if k == "3" and "승무패" not in gt else ""),
            "1만 원당 돌아오는 돈 (배당 기준)": won(100 / (1 + m / 100)) if m is not None else "-",
            "AI가 배당보다 나은가": ("판단 불가 (경기 수 부족)" if str(r["verdict"]).startswith("판정 불가")
                               else plain.get(r["verdict"], r["verdict"])),
            "2026년 시험 경기 수": r.get("n_test", 0)})
    with st.expander("유형별 표 (승무패 · 핸디캡 · 언더오버 · 홀짝 …)"):
        st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width="stretch")
        cands = scan.get("candidates") or []
        st.caption("'1만 원당 돌아오는 돈'은 배당에 들어 있는 베트맨 몫으로 계산한 평균이에요. "
                   "'AI가 배당보다 나은가'는 2026년 시험에서 AI 확률이 배당 확률보다 더 잘 맞았는지예요. "
                   + ("AI 기준으로 이익이 나는 유형은 없었어요." if not cands else f"이익 가능성을 확인할 유형: {', '.join(cands)}"))


def render_calibration(card) -> None:
    """🎯 보정 곡선 — AI가 '○○%'라고 한 선택지들이 실제로 몇 % 일어났나"""
    cal = (card or {}).get("calibration") or {}
    ai, mk = cal.get("ai") or [], {b["range"]: b for b in (cal.get("market") or [])}
    if not ai:
        return
    st.markdown("**🎯 AI가 말한 확률, 실제로 맞았나?**")
    st.caption("AI가 '60~70%'라고 한 선택지들이 실제로도 60~70% 정도 일어났다면, AI가 말하는 확률을 믿을 만하다는 뜻이에요. "
               "선택지가 30개 미만인 줄은 운이 커서 참고만 하세요.")
    rows = []
    for b in ai:
        diff = b["happened_pct"] - b["said_pct"]
        judge = ("표본 적음" if b["n"] < 30 else "✅ 비슷" if abs(diff) <= 10
                 else "⬆️ 실제로 더 자주 일어남" if diff > 0 else "⬇️ 실제로 덜 일어남")
        m = mk.get(b["range"])
        rows.append({"AI가 말한 확률": b["range"], "선택지 수": b["n"], "실제로 일어난 비율": f"{b['happened_pct']:.0f}%",
                     "판단": judge, "배당이 같은 확률이라 한 것은": f"{m['happened_pct']:.0f}% ({m['n']}개)" if m else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


@st.cache_data(ttl=60)
def load_results_index():
    """(회차, 경기번호) → 결과 목록. 같은 번호에 기준선이 여러 개인 경우가 있어 리스트로 보관"""
    path = project_root / "data" / "history" / "betman_recent_results.json"
    idx = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                idx.setdefault((str(r.get("gm_ts")), str(r.get("match_seq"))), []).append(r)
    return idx


@st.cache_data(ttl=60)
def load_prediction_log(gm_ts: str):
    """경기 시작 전에 고정 저장된 AI 예측 (사후 정보가 섞이지 않은 채점용 기록)"""
    path = project_root / "data" / "history" / "prediction_log" / f"{gm_ts}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("matches", {})


@st.cache_data(ttl=60)
def load_past_round_matches(gm_ts: str) -> list:
    """
    과거 회차를 결과 파일(betman_recent_results.json)로부터 카드 형식으로 재구성 (조회 전용).
    배당은 결과 파일에 기록된 값(발매 최종 배당)이며, 배당 예상은 그 배당의 마진 제거 확률.
    """
    from models.odds_normalizer import remove_margin
    out = []
    for rows in load_results_index().values():
        for r in rows:
            if str(r.get("gm_ts")) != str(gm_ts):
                continue
            gt = str(r.get("game_type", ""))
            hw = r.get("handi_win") or 0
            if "핸디캡" in gt:
                cat, label = ("전반 핸디캡" if "전반" in gt else "핸디캡"), f"핸디캡 ({float(hw):+.1f})"
            elif "언더오버" in gt:
                cat, label = ("전반 언더오버" if "전반" in gt else "언더오버"), f"언더/오버 (기준 {float(hw):g})"
            elif "SUM" in gt:
                cat, label = "홀짝", "홀/짝"
            else:
                cat = "전반 승무패" if "전반" in gt else "일반 승무패"
                label = cat
            odds = {"home_win": float(r.get("odds_win") or 0), "draw": float(r.get("odds_draw") or 0),
                    "away_win": float(r.get("odds_lose") or 0)}
            mkt = None
            if odds["home_win"] > 1 and odds["away_win"] > 1:
                mkt = remove_margin(odds["home_win"], odds["draw"] if odds["draw"] > 1 else 0, odds["away_win"])["true_probs"]
            out.append({
                "gm_ts": int(gm_ts), "match_seq": str(r.get("match_seq")), "match_time": r.get("match_time", ""),
                "league": r.get("league", ""), "sport": r.get("sport", ""), "game_type": gt,
                "category": cat, "category_label": label, "line": float(hw),
                "home_team": r.get("home_team", ""), "away_team": r.get("away_team", ""),
                "betman_odds": odds, "market_implied_probs": mkt, "is_single": False,
                "prob_source": None, "evaluation": {}, "_result": r,
                # 같은 경기번호에 기준선이 여러 개 있을 수 있어 카드 고유 키를 따로 둔다
                "_key": f"{r.get('match_seq')}_{r.get('bettyp_cd', '')}_{r.get('handi_win', '')}_{r.get('handi_lose', '')}",
            })
    out.sort(key=lambda m: (m["match_time"], int(m["match_seq"]) if m["match_seq"].isdigit() else 0))
    return out


def detect_current_gm_ts(preds) -> str:
    for m in preds:
        if m.get("gm_ts"):
            return str(m["gm_ts"])
    snap_dir = project_root / "data" / "history" / "odds_history"
    names = sorted(p.stem for p in snap_dir.glob("*.json") if p.stem.isdigit()) if snap_dir.exists() else []
    return names[-1] if names else ""


def find_result(m, results_idx, gm_ts):
    if m.get("_result") is not None:      # 과거 회차 카드는 결과 행에서 직접 만들어짐
        return m["_result"]
    cands = results_idx.get((str(gm_ts), str(m.get("match_seq"))), [])
    if len(cands) > 1:
        # 같은 경기번호에 기준값(핸디캡·언더오버)마다 결과 줄이 있음 → 기준값(handi_win)이 같은 줄.
        # handi_val 은 유형 코드(2, 23 …)라 비교하면 H +2.0 같은 기준값과 잘못 맞을 수 있어 쓰지 않는다
        line = float(m.get("line") or 0.0)
        for c in cands:
            if abs(float(c.get("handi_win") or 0.0) - line) < 1e-9:
                return c
    return cands[0] if cands else None


def pick_labels(m) -> dict:
    gt, cat = str(m.get("game_type", "")), str(m.get("category", ""))
    if "SUM" in gt:
        return {"home_win": "홀", "draw": "-", "away_win": "짝"}
    if "언더오버" in cat or "언더오버" in gt:
        return {"home_win": "언더", "draw": "-", "away_win": "오버"}
    if "핸디캡" in cat or "핸디캡" in gt:
        return {"home_win": "핸승", "draw": "핸무", "away_win": "핸패"}
    # 베트맨 규정: 승①패 = 1점차 이내 승부(무승부 포함), 승⑤패 = 5점차 이내 승부 — 가운데는 '무'가 아님
    if "승1패" in gt:
        return {"home_win": "승", "draw": "①", "away_win": "패"}
    if "승5패" in gt:
        return {"home_win": "승", "draw": "⑤", "away_win": "패"}
    return {"home_win": "승", "draw": "무", "away_win": "패"}


def category_display(m) -> str:
    """표시·필터용 게임 유형 (홀/짝이 '일반 승무패'로 잘못 분류되는 문제(R14) 보정)"""
    if "SUM" in str(m.get("game_type", "")):
        return "홀/짝"
    return m.get("category_label", m.get("game_type", "승무패"))


def category_group(m) -> str:
    """필터용 유형 묶음 (기준선 값은 무시): 일반 승무패 / 핸디캡 / 언더오버 / 홀/짝 / 전반 …"""
    if "SUM" in str(m.get("game_type", "")):
        return "홀/짝"
    return m.get("category") or m.get("game_type", "승무패")


SPORT_ICONS = {"SC": "⚽", "BK": "🏀", "BS": "⚾", "VL": "🏐"}
TYPE_BADGES = ("승무패", "핸디캡", "언더오버", "홀짝", "전반")


def type_badge(m) -> str:
    """유형 배지 필터 묶음: 전반 / 홀짝 / 핸디캡 / 언더오버 / 승무패(승패·승5패·승1패 포함)"""
    gt = f"{m.get('game_type', '')} {m.get('category', '')}"
    if "전반" in gt:
        return "전반"
    if "SUM" in gt or "홀" in gt:
        return "홀짝"
    if "핸디캡" in gt:
        return "핸디캡"
    if "언더오버" in gt:
        return "언더오버"
    return "승무패"


def type_name(m) -> str:
    """줄에 크게 보여 줄 유형 이름 (종목 이름 빼고): '축구 핸디캡' → '핸디캡', '야구 SUM' → '홀짝'"""
    gt = str(m.get("game_type", "")).strip()
    if not gt:
        return category_display(m)
    words = gt.split(" ")
    if words[0] in SPORT_NAMES.values() and len(words) > 1:
        words = words[1:]
    return " ".join(words).replace("SUM", "홀짝")


def line_chip(m) -> str:
    """기준선 칩: 핸디캡 H -1.0 (파랑) · 언더오버 U/O 2.5 (초록)"""
    gt = str(m.get("game_type", "")) + str(m.get("category", ""))
    try:
        line = float(m.get("line") or 0.0)
    except (TypeError, ValueError):
        return ""
    style = "padding:1px 7px;border-radius:5px;font-size:12px;font-weight:700"
    if "핸디캡" in gt and line:
        return f"<span style='{style};background:rgba(59,130,246,.15);color:#60a5fa'>H {line:+.1f}</span>"
    if "언더오버" in gt and line:
        return f"<span style='{style};background:rgba(16,185,129,.15);color:#34d399'>U/O {line:g}</span>"
    return ""


ONE_GAME_TAG = ("<span style='padding:1px 6px;border-radius:5px;font-size:11px;font-weight:700;"
                "background:rgba(245,158,11,.18);color:#f59e0b'>한경기</span>")


def league_chip(league: str) -> str:
    """리그: 보라 칩 (핸디캡 파랑 · 언더오버 초록 · 한경기 주황과 겹치지 않는 대비색)"""
    return ("<span style='padding:1px 8px;border-radius:5px;font-size:12px;font-weight:700;"
            f"background:rgba(168,85,247,.16);color:#c084fc'>{league}</span>")


def short_when(m) -> str:
    """'26.10.26 (월) 08:30' → '10.26(월) 08:30'"""
    tm = re.search(r"\d{2}\.(\d{2}\.\d{2})\s*\(([^)]*)\)\s*(\d{1,2}:\d{2})", str(m.get("match_time", "")))
    return f"{tm.group(1)}({tm.group(2)}) {tm.group(3)}" if tm else str(m.get("match_time", ""))


def render_tiles(tiles):
    """작은 숫자 칸 한 줄: [(제목, 값, 보조 설명), ...] — 좁은 화면에서는 다음 줄로 넘어감
    (아래 padding: Streamlit 글 블록의 -1rem 아래 여백 때문에 다음 줄 글과 붙지 않게)"""
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:2px;padding-bottom:14px'>" + "".join(
            "<div style='flex:1 1 160px;padding:8px 12px;border:1px solid rgba(148,163,184,.3);border-radius:8px'>"
            f"<div style='font-size:12px;opacity:.75'>{label}</div>"
            f"<div style='font-size:20px;font-weight:700;line-height:1.4'>{value}</div>"
            f"<div style='font-size:12px;opacity:.65'>{sub}</div></div>"
            for label, value, sub in tiles) + "</div>",
        unsafe_allow_html=True)


def keep_valid(key, options):
    """목록이 바뀌어(회차·종목 변경) 예전 선택값이 없어졌으면 '전체'로 되돌림 (위젯을 그리기 전에만 호출)"""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def keep_valid_multi(key, options):
    """배지(여러 개 선택)에서 지금 목록에 없는 값만 빼기 (위젯을 그리기 전에만 호출)"""
    cur = st.session_state.get(key) or []
    if any(v not in options for v in cur):
        st.session_state[key] = [v for v in cur if v in options]


def is_sale_closed(m, now_ms: float) -> bool:
    """베트맨 발매 마감 시각(close_time, ms)이 지났는지. 마감 정보가 없으면(지난 회차) 마감으로 본다."""
    ct = m.get("close_time")
    return True if not ct else float(ct) <= now_ms


def available_picks(m) -> list:
    odds = m.get("betman_odds", {})
    return [k for k in PICK_KEYS if float(odds.get(k) or 0.0) > 1.0]


def top_pick(probs: dict, opts: list):
    """확률 1순위 선택 (probs 키는 home_win/draw/away_win 또는 home/draw/away)"""
    if not probs or not opts:
        return None, 0.0
    get = lambda k: float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0)
    best = max(opts, key=get)
    return best, get(best)


def prob_of(probs, k) -> float:
    """확률 사전에서 선택지 확률 (키가 home_win/draw/away_win 또는 home/draw/away 어느 쪽이든)"""
    return float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0) if probs else 0.0


def time_key(m) -> str:
    """경기 시작 시각 정렬 키 'YYMMDDHHMM'"""
    t = re.search(r"(\d{2})\.(\d{2})\.(\d{2}).*?(\d{1,2}):(\d{2})", str(m.get("match_time", "")))
    return "".join(f"{int(x):02d}" for x in t.groups()) if t else ""


def combo_key(pairs) -> str:
    """조합의 고유 이름 '경기번호:선택지|…' (순서 무관).
    조합별 금액·선택 체크를 조합 번호(#1, #2 …)가 아니라 내용에 묶는다 — 번호에 묶으면 고른 경기가 바뀌었을 때
    예전 조합의 금액·체크가 같은 번호의 새 조합에 그대로 남는다."""
    return "|".join(f"{seq}:{sel}" for seq, sel in sorted((str(s), k) for s, k in pairs))


def selection_status() -> dict:
    """담은 경기 → 조합 수 · 예상 배당 범위 · 베트맨 규정 검사 (사이드바 🛒 내 선택에서 사용)"""
    active = {}
    for seq, sel in st.session_state["selections"].items():
        m = pred_map.get(seq)
        if not m:
            continue
        odds, labels = m.get("betman_odds", {}), pick_labels(m)
        picks = [{"selection": k, "selection_name": labels[k], "odds": float(odds[k])}
                 for k in PICK_KEYS if sel.get(SEL_KEY[k]) and float(odds.get(k) or 0.0) > 1.0]
        if picks:
            active[seq] = {"match": m, "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}",
                           "is_single": bool(m.get("is_single")), "picks": picks}
    n = len(active)
    combos = 0
    if active:
        combos = 1
        for v in active.values():
            combos *= len(v["picks"])
    odds_lo = calculate_combined_odds([min(p["odds"] for p in v["picks"]) for v in active.values()]) if active else 0.0
    odds_hi = calculate_combined_odds([max(p["odds"] for p in v["picks"]) for v in active.values()]) if active else 0.0

    valid = False
    if n == 0:
        msg = "ℹ️ 경기를 선택해 주세요 (조합은 최소 2경기, 한경기 표시 경기는 1경기도 가능)."
    elif n == 1:
        if next(iter(active.values()))["is_single"]:
            valid, msg = True, "✅ 한경기 구매가 가능한 경기입니다. 조합 결과 보기가 가능합니다."
        else:
            msg = "⚠️ 1경기만 선택되었습니다. 이 경기는 한경기 구매가 안 되므로 2경기 이상 선택해야 합니다."
    elif n > 10:
        msg = f"⚠️ 베트맨 규정상 조합은 최대 10경기까지만 가능합니다 (현재 {n}경기)."
    elif len({v["match_key"] for v in active.values()}) != n:
        msg = "⚠️ 같은 대진의 다른 유형(예: 승무패와 핸디캡)은 한 조합에 묶을 수 없습니다."
    elif any("더블찬스" in str(v["match"].get("game_type", "")) for v in active.values()):
        msg = "⚠️ 더블찬스는 한경기 전용 상품이라 다른 경기와 묶을 수 없어요 (베트맨 안내)."
    else:
        valid, msg = True, f"✅ 베트맨 프로토 규정 통과! ({n}경기 선택 / 총 {combos:,}개 조합)"
    return {"active": active, "n": n, "combos": combos, "odds_lo": odds_lo, "odds_hi": odds_hi,
            "valid": valid, "msg": msg}


# 4. 세션 상태 초기화 (단일 상태: selections)
if "selections" not in st.session_state:
    st.session_state["selections"] = {}  # { match_seq: {"home": bool, "draw": bool, "away": bool} }
    # 새로고침·대시보드 재시작 후에도 이번 회차에 고르던 선택을 그대로 복원 (자동 저장 파일, 회차가 바뀌면 버림)
    _cur = load_json(MY_CURRENT_PATH, {})
    if _cur.get("gm_ts") and str(_cur["gm_ts"]) == detect_current_gm_ts(predictions):
        _seqs = {str(m["match_seq"]) for m in predictions}
        st.session_state["selections"] = {s: v for s, v in _cur.get("selections", {}).items() if s in _seqs}
        for _ck, _cs in _cur.get("combo_state", {}).items():
            if _cs.get("stake") is not None:
                st.session_state[f"stake_{_ck}"] = int(_cs["stake"])
            st.session_state[f"combo_chk_{_ck}"] = bool(_cs.get("picked", True))
        if _cur.get("batch_stake"):
            st.session_state["batch_stake"] = _cur["batch_stake"]

if "custom_step" not in st.session_state:
    st.session_state["custom_step"] = 1  # 1: STEP 01 맞춤조건설정, 2: STEP 02 조합결과보기

if "editor_version" not in st.session_state:
    st.session_state["editor_version"] = 0

if "batch_stake" not in st.session_state:
    st.session_state["batch_stake"] = 1000

pred_map = {str(m["match_seq"]): m for m in predictions}

# 타이틀 헤더
st.title("⚽ 스포츠 승부 예측 & 맞춤조건구매 대시보드")
st.caption("베트맨 프로토 승부식 공식 계산 규칙 준수 · 복수 선택 조합 시뮬레이터 · 베트맨 과거 데이터로 학습한 AI 보정 확률")

# 5. 메인 3대 탭 구성 (중복 요소 제거 및 직관적 분류)
# key + on_change: 열린 탭을 코드에서 바꿀 수 있게 (사이드바 '조합 결과 보기' → ① 탭으로 이동).
# 탭을 바꾸면 앱이 한 번 다시 실행되지만, 세 탭 내용은 그대로 모두 그려 필터·선택 상태를 유지한다
MAIN_TABS = ["🛒 ① 맞춤조건구매 (시뮬레이터)", "🔍 ② AI 경기 분석 (조회 전용)", "📊 ③ 성과 검증 (실제 경기 대조 & 통계)",
             "📒 ④ 내 베팅 기록"]
tab1, tab2, tab3, tab4 = st.tabs(MAIN_TABS, key="main_tab", on_change="rerun")

# ==============================================================================
# TAB 1: 🛒 ① 맞춤조건구매 (시뮬레이터 전용)
# ==============================================================================
with tab1:
    st.markdown("""
    <div class='step-banner'>
        <b>💡 맞춤조건구매 계산기</b><br>
        경기마다 승·무·패를 하나 이상 고르면, 만들어지는 모든 조합의 배당·환급금·비과세 여부를 자동으로 계산해요.<br>
        <span style='font-size:12px; color:#64748b;'>※ 계산 전용입니다. 실제 구매는 베트맨에서 직접 하세요.</span>
    </div>
    """, unsafe_allow_html=True)

    # 상단 단계 전환 버튼 (① 경기 고르기 ↔ ② 조합 결과 보기)
    STEP_LABELS = ["① 경기 고르기", "② 조합 결과 보기"]

    def _go_step(n: int):
        """버튼 콜백: 단계 값과 단계 버튼 상태를 함께 바꾸고 ① 탭을 연다
        (다른 탭·사이드바에서 눌러도 바로 그 단계 화면으로 이동)"""
        st.session_state["custom_step"] = n
        st.session_state["step_nav"] = STEP_LABELS[n - 1]
        st.session_state["main_tab"] = MAIN_TABS[0]

    # 선택된 단계를 다시 누르면 선택이 풀리므로(None), 그리기 전에 지금 단계로 되돌려 둔다
    if st.session_state.get("step_nav") not in STEP_LABELS:
        st.session_state["step_nav"] = STEP_LABELS[st.session_state["custom_step"] - 1]
    step_col1, step_col2 = st.columns([2.5, 1.5], vertical_alignment="bottom")
    with step_col1:
        step_choice = st.segmented_control("진행 단계", STEP_LABELS, key="step_nav")
        st.session_state["custom_step"] = STEP_LABELS.index(step_choice) + 1

    with step_col2:
        # 빠른 초기화 버튼 (칸 전체 폭: 반 칸에 두면 좁은 화면에서 '🗑️ …'로 잘림)
        if st.button("🗑️ 선택 전체 초기화", width="stretch"):
            st.session_state["selections"] = {}
            st.session_state["editor_version"] += 1
            st.rerun()

    st.divider()

    # -------------------------------------------------------------
    # STEP 01: 맞춤 조건 설정 화면
    # -------------------------------------------------------------
    if st.session_state["custom_step"] == 1:
        st.markdown("### 📋 STEP 01. 대상 경기 검색 및 승/무/패 다중 선택")
        st.caption("배당 버튼을 눌러 선택하세요. 한 경기에서 여러 개를 고르면 가능한 모든 조합이 자동으로 계산됩니다. "
                   "고른 경기는 왼쪽 🛒 내 선택에 담겨요 (휴대폰은 왼쪽 위 › 를 누르면 열려요).")
        if not GEM_ON:   # [6.6] 판단 요령 한 줄 (쉬운 말)
            _bs = bucket_summary(load_market_scan())
            st.info("🔒 지금은 **AI 추천이 잠겨 있어요** (AI가 시험 통과 전). "
                    + (f"사고 싶다면 **배당 1.3 이하 정배**가 덜 손해예요 — 1만 원당 평균 약 {won(_bs['low'])} 돌아왔고, "
                       f"5배 이상 고배당은 약 {won(_bs['high'])}이었어요. " if _bs else "")
                    + "자세한 건 ③ 성과 검증 탭 맨 위 '🧭 지금 결론'을 보세요.")
        _chip = "display:inline-block;margin:0 14px 6px 0;font-size:13px;color:#cbd5e1"
        st.markdown(
            "<div style='margin:-4px 0 8px 0'>"
            f"<span style='{_chip}'>⭐ AI 확률 1순위 <span style='color:#94a3b8'>(AI 없으면 배당 1순위)</span></span>"
            + (f"<span style='{_chip}'>💎 AI 기준 기대가치 플러스 <span style='color:#f59e0b'>(검증 통과)</span></span>" if GEM_ON else
               f"<span style='{_chip}'>🔒 💎 추천 잠금 <span style='color:#94a3b8'>({LOCK_NOTE})</span></span>") +
            f"<span style='{_chip}'>⚠️ AI 예상이 배당과 다름</span>"
            f"<span style='{_chip}'>✅ ❌ 끝난 경기의 적중 여부</span>"
            "</div>",
            unsafe_allow_html=True)

        # 🧩 조합 찾기 — 데이터로 확인한 원칙(저배당 정배 축, AI ▲, 💎 우선, 커버 없음)으로 2~3폴더 후보
        if not FINDER_ON:
            with st.expander("🔒 조합 찾기 — 추천 잠금 중 (검증 3단계 통과 전)", expanded=False):
                lock_notice(REC_STATUS["features"]["finder"])
        else:
            with st.expander("🧩 조합 찾기 — 과거 데이터로 확인한 원칙으로 이번 회차 2~3폴더 후보 만들기", expanded=False):
                st.caption("원칙: ① 축은 저배당 정배(1.3배 이하가 가장 손실이 작음) ② AI가 배당보다 높게 본(▲) 정배만, AI와 배당 1순위가 다르면(⚠️) 제외 "
                           "③ 💎 선택지 우선 ④ 전배·커버 없이 2~3폴더. 정렬은 AI 기준 기대 손익이 덜 나쁜 순입니다.")
                fc1, fc2, fc3, fc4 = st.columns([1.6, 1, 1, 1], vertical_alignment="bottom")
                with fc1:
                    fav_rng = st.slider("축 배당 범위", 1.01, 2.0, (1.01, 1.5), 0.01, key="finder_range")
                with fc2:
                    sizes_opt = st.selectbox("폴더 수", ["2~3폴더", "2폴더", "3폴더"], key="finder_sizes")
                with fc3:
                    only_up = st.checkbox("AI ▲ 정배만", value=True, key="finder_up")
                with fc4:
                    with_gems = st.checkbox("💎 포함", value=True, key="finder_gems")
                sizes = {"2~3폴더": (2, 3), "2폴더": (2,), "3폴더": (3,)}[sizes_opt]
                f_legs = candidate_legs(predictions, fav_rng, only_ai_up=only_up, include_gems=with_gems)
                f_combos = find_combos(f_legs, sizes=sizes, top=8)
                if not f_combos:
                    st.info(f"조건에 맞는 후보가 없습니다 (후보 선택지 {len(f_legs)}개). 축 배당 범위를 넓히거나 'AI ▲ 정배만'을 해제해 보세요. "
                            "후보가 없는 회차는 무리해서 조합하지 않는 것이 원칙에 맞습니다.")
                else:
                    st.caption(f"후보 선택지 {len(f_legs)}개 → 상위 {len(f_combos)}개 조합 (아직 시작하지 않은 축구 승무패만)")

                def _pick_combo(legs_):
                    st.session_state["selections"] = {
                        leg["seq"]: {SEL_KEY[k]: (k == leg["key"]) for k in PICK_KEYS} for leg in legs_}
                    st.session_state["editor_version"] += 1
                    _go_step(2)

                for ci, c in enumerate(f_combos):
                    with st.container(border=True):
                        cc1, cc2, cc3 = st.columns([4.2, 2.6, 1.2], vertical_alignment="center")
                        with cc1:
                            parts = []
                            for leg in c["legs"]:
                                m = leg["match"]
                                if leg["key"] == "home_win":
                                    what = f"<b>{m['home_team']}</b> 승"
                                elif leg["key"] == "away_win":
                                    what = f"<b>{m['away_team']}</b> 승(원정)"
                                else:
                                    what = f"<b>{m['home_team']} vs {m['away_team']}</b> 무승부"
                                parts.append(f"{leg['role']} {what} {leg['odds']:.2f} "
                                             f"<span style='color:#94a3b8;font-size:12px'>({m.get('league', '')}, AI {leg['p_ai'] * 100:.0f}% · 배당 {leg['p_mk'] * 100:.0f}%)</span>")
                            st.markdown("<div style='font-size:14px;line-height:1.6'>" + "<br>".join(parts) + "</div>", unsafe_allow_html=True)
                        with cc2:
                            ev_col = "#10b981" if c["ev_ai"] > 0 else "#f59e0b"
                            st.markdown(
                                f"<div style='font-size:13px;line-height:1.6'><b style='font-size:16px'>{c['odds']:.1f}배</b> · 적중 확률(AI) {c['p_ai'] * 100:.1f}%<br>"
                                f"기대 손익 AI <b style='color:{ev_col}'>{c['ev_ai'] * 100:+.1f}%</b> · 배당 기준 {c['ev_mk'] * 100:+.1f}%</div>",
                                unsafe_allow_html=True)
                            rule = [{"type": "single", "kind": outcome_kind([float(leg["match"]["betman_odds"][k]) for k in PICK_KEYS],
                                                                             PICK_KEYS.index(leg["key"])),
                                     "ref_odds": leg["odds"]} for leg in c["legs"]]
                            bt = cached_backtest(tuple((r["kind"], round(r["ref_odds"], 2)) for r in rule))
                            if bt.get("rounds"):
                                st.caption(f"과거 반복: {bt['rounds']}회차 · 적중 {bt['hit_rate_pct']:.0f}% · 실제 수익률 {bt['roi_pct']:+.1f}%")
                        with cc3:
                            st.button("🛒 담기", key=f"finder_pick_{ci}", width="stretch",
                                      on_click=_pick_combo, args=(c["legs"],))
                st.caption("⚠️ 대부분의 조합은 기대 손익이 마이너스입니다. AI 확률은 검증 중이며, 과거 반복 결과는 비슷한 배당 경기로 같은 방식을 적용한 실제 결과입니다.")

        # 필터 컨트롤: ① 회차·날짜 ② 검색·종목·리그·유형 ③ 빠른 필터 (리그·유형 목록은 고른 종목에 맞춰 줄어듦)
        current_gm_ts = detect_current_gm_ts(predictions)
        QUICK_FILTERS = {"open": "⏳ 마감 전만", "single": "⚡ 한경기만", "gem": "💎 AI 플러스만", "picked": "✔️ 내가 고른 경기만"}
        if not GEM_ON:   # [6.5] 추천 잠금 중에는 💎 필터 없음
            QUICK_FILTERS.pop("gem")
        FILTER_DEFAULTS = {"step1_search": "", "step1_sports": [], "step1_league": "전체 리그",
                           "step1_types": [], "step1_quick": []}

        def _reset_filters():
            for k, v in FILTER_DEFAULTS.items():
                st.session_state[k] = v
            for k in [k for k in st.session_state if str(k).startswith("step1_day_")]:
                del st.session_state[k]

        with st.container(border=True):
            # 회차 선택: 현재 판매 회차 + 결과가 있는 과거 회차 (과거는 조회 전용)
            past_rounds = sorted({ts for ts, _ in load_results_index().keys() if ts and ts != str(current_gm_ts)}, reverse=True)
            round_labels = {"ALL": f"전체 회차 (현재 + 지난 {len(past_rounds)}개)"}
            if current_gm_ts:
                round_labels[str(current_gm_ts)] = f"{current_gm_ts}회차 (현재 판매)"
            round_labels.update({ts: f"{ts}회차 (종료·조회 전용)" for ts in past_rounds})
            # 날짜(r2)는 다른 필터를 적용한 뒤 경기 수와 함께 채운다
            r1, r2, r3 = st.columns([1.4, 3.3, 1.0], vertical_alignment="bottom")
            with r1:
                sel_round = st.selectbox("🗓️ 회차", list(round_labels), index=1 if current_gm_ts else 0,
                                         format_func=lambda x: round_labels[x], key="step1_round")
            with r3:
                live_on = st.toggle("🔴 실시간 결과", value=True, key="step1_live",
                                    help="대시보드를 열어 둔 동안 1분마다 베트맨 결과를 확인해, 끝난 경기의 결과·적중 여부를 바로 표시합니다.")
            is_all = sel_round == "ALL"
            is_past = (not is_all) and sel_round != str(current_gm_ts)
            if is_all:
                round_matches = list(predictions) + [m for ts in past_rounds for m in load_past_round_matches(ts)]
            else:
                round_matches = load_past_round_matches(sel_round) if is_past else predictions

            # 종목·유형 배지: 아무것도 안 누르면 전체, 여러 개 누르면 그중 하나라도 맞는 경기
            b1, b2 = st.columns([1.5, 2.6])
            with b1:
                sport_opts = [s for s in ("SC", "BK", "BS", "VL") if any(m.get("sport") == s for m in round_matches)]
                keep_valid_multi("step1_sports", sport_opts)
                sel_sports = st.pills("종목", sport_opts, selection_mode="multi", key="step1_sports",
                                      format_func=lambda s: f"{SPORT_ICONS.get(s, '')} {SPORT_NAMES.get(s, s)}") or []
            in_sport = [m for m in round_matches if not sel_sports or m.get("sport") in sel_sports]
            with b2:
                type_opts = [t for t in TYPE_BADGES if any(type_badge(m) == t for m in in_sport)]
                keep_valid_multi("step1_types", type_opts)
                sel_types = st.pills("게임 유형", type_opts, selection_mode="multi", key="step1_types") or []

            f1, f2 = st.columns([1.5, 2.6])
            with f1:
                search_kw = st.text_input("🔎 팀 이름 검색", placeholder="예: 아스널, 뮌헨, 인천", key="step1_search")
            with f2:
                league_opts = ["전체 리그"] + sorted({m.get("league", "기타") for m in in_sport})
                keep_valid("step1_league", league_opts)
                sel_league = st.selectbox("리그", league_opts, key="step1_league")

            q1, q2 = st.columns([5, 1], vertical_alignment="center")
            with q1:
                keep_valid_multi("step1_quick", list(QUICK_FILTERS))
                quick = st.pills("빠른 필터", list(QUICK_FILTERS), format_func=QUICK_FILTERS.get,
                                 selection_mode="multi", key="step1_quick", label_visibility="collapsed") or []
            with q2:
                st.button("↺ 필터 초기화", key="step1_reset", on_click=_reset_filters, width="stretch")
            count_slot = st.empty()
            only_open, only_single_opt, only_rec_opt, only_picked = (k in quick for k in ("open", "single", "gem", "picked"))

        # 필터링 적용
        now_ts = datetime.now().timestamp() * 1000
        filtered_matches = []
        for m in round_matches:
            if search_kw:
                q = search_kw.lower()
                if q not in m.get("home_team", "").lower() and q not in m.get("away_team", "").lower():
                    continue
            if sel_league != "전체 리그" and m.get("league") != sel_league:
                continue
            if sel_types and type_badge(m) not in sel_types:
                continue
            if sel_sports and m.get("sport") not in sel_sports:
                continue
            if only_open and is_sale_closed(m, now_ts):
                continue
            if only_single_opt and not m.get("is_single", False):
                continue
            filtered_matches.append(m)

        # 경기 카드 목록: 배당 버튼을 눌러 바로 선택 (가로 스크롤 없이 휴대폰에서도 사용)
        results_idx = load_results_index()
        live_ts = str(current_gm_ts)
        if live_on and current_gm_ts and sel_round in (live_ts, "ALL") and started_without_result(predictions, results_idx, live_ts):
            live_rows = fetch_live_round_results(live_ts)
            if live_rows:
                results_idx = dict(results_idx)
                fresh = {}
                for r in live_rows:
                    fresh.setdefault((live_ts, str(r.get("match_seq"))), []).append(r)
                results_idx.update(fresh)
        # 경기마다 자기 회차 기준으로 결과·경기 전 AI 기록을 찾는다 (전체 회차 보기 지원)
        def _ts(m):
            return str(m.get("gm_ts") or current_gm_ts)

        def _res(m):
            return find_result(m, results_idx, _ts(m))

        def _log(m):
            return load_prediction_log(_ts(m)).get(str(m.get("match_seq")))

        def _past(m):
            return _ts(m) != str(current_gm_ts)

        if is_past:
            has_log = bool(load_prediction_log(sel_round))
            st.info(f"🗂️ {sel_round}회차는 종료된 회차입니다 (조회 전용). 배당은 최종 발매 배당이며, "
                    + ("AI 예상은 경기 전에 기록된 경기만 표시됩니다." if has_log else "이 회차에는 경기 전 AI 기록이 없어 배당 예상만 표시됩니다."))
        if is_all:
            logged = [ts for ts in [str(current_gm_ts)] + past_rounds if load_prediction_log(ts)]
            st.info(f"🗂️ 전체 회차 보기: 현재 회차는 선택 가능, 지난 회차는 조회 전용입니다. "
                    f"AI 예상·💎는 경기 전 AI 기록이 있는 회차({', '.join(logged) or '없음'})에서만 나옵니다.")

        def _ai_probs(m, res, log):
            """AI 예상 확률: AI가 적용된 유형만. 끝난 경기는 경기 전 고정 기록으로만 (경기 후 재계산 값은 결과가 섞일 수 있음)"""
            if res is not None:
                return log.get("model_probs") if log else None
            return m.get("blended_probabilities") if m.get("prob_source") in AI_SOURCES else None

        def _gems(m, ai_probs, opts):
            """💎 = AI 확률 기준 기대가치(EV) 플러스 선택지 (1순위가 아니어도 표시). [6.5] 추천 잠금 중에는 표시 안 함"""
            out = {}
            if ai_probs and GEM_ON:
                for k in opts:
                    p_k = float(ai_probs.get(k, ai_probs.get(k.replace("_win", ""), 0.0)) or 0.0)
                    ev_k = p_k * float(m.get("betman_odds", {}).get(k, 0.0)) - 1.0
                    if ev_k > 0:
                        out[k] = ev_k
            return out

        def _match_day(m):
            mt = re.search(r"\d{2}\.(\d{2})\.(\d{2})\s*(\([^)]*\))?", str(m.get("match_time", "")))
            return f"{mt.group(1)}.{mt.group(2)} {mt.group(3) or ''}".strip() if mt else "날짜 미정"

        # 날짜: 며칠뿐이면 버튼으로 바로 전환, 많으면(전체 회차 보기) 목록. 날짜 옆 숫자 = 그날 경기 수
        day_counts = {}
        for m in filtered_matches:
            day_counts[_match_day(m)] = day_counts.get(_match_day(m), 0) + 1
        day_opts = ["전체 날짜"] + sorted(day_counts)
        # 기본값: 아직 결과가 안 나온 경기가 있는 첫 날짜 (바로 고를 수 있는 경기부터 보이도록)
        open_days = sorted({_match_day(m) for m in filtered_matches if _res(m) is None})
        day_default = open_days[0] if (open_days and not is_all) else "전체 날짜"
        day_buttons = len(day_opts) <= 6
        day_key = f"step1_day_{sel_round}_{'seg' if day_buttons else 'sel'}"
        if st.session_state.get(day_key) not in day_opts + [None]:
            st.session_state.pop(day_key, None)

        def _day_fmt(d):
            return d if d == "전체 날짜" else f"{d} · {day_counts[d]}경기"

        with r2:
            if day_buttons:
                # 선택된 날짜를 다시 누르면 선택이 풀림(None) → 전체 날짜
                sel_day = st.segmented_control("📅 경기 날짜", day_opts, default=day_default, format_func=_day_fmt,
                                               key=day_key) or "전체 날짜"
            else:
                sel_day = st.selectbox("📅 경기 날짜", day_opts, index=day_opts.index(day_default),
                                       format_func=_day_fmt, key=day_key)
        if sel_day != "전체 날짜":
            filtered_matches = [m for m in filtered_matches if _match_day(m) == sel_day]
        if only_rec_opt:
            filtered_matches = [m for m in filtered_matches if _gems(
                m, _ai_probs(m, _res(m), _log(m)),
                available_picks(m))]
        if only_picked:
            filtered_matches = [m for m in filtered_matches if str(m.get("match_seq")) in st.session_state["selections"]]

        def _time_key(m):
            t = re.search(r"(\d{2})\.(\d{2})\.(\d{2}).*?(\d{1,2}):(\d{2})", str(m.get("match_time", "")))
            return "".join(f"{int(x):02d}" for x in t.groups()) if t else ""

        def _section(m):
            """open = 지금 고를 수 있음 · wait = 발매 마감, 결과 대기 · done = 결과 나옴"""
            if _res(m) is not None:
                return "done"
            return "wait" if (_past(m) or is_sale_closed(m, now_ts)) else "open"

        def _fixture(m):
            """같은 대진(같은 회차·팀·시작 시각) = 한 묶음"""
            return (_ts(m), m.get("home_team", ""), m.get("away_team", ""), _time_key(m))

        def _seq_num(m):
            s = str(m.get("match_seq", ""))
            return int(s) if s.isdigit() else 0

        # 같은 대진의 게임번호가 흩어지지 않도록: 대진 첫 번호 → 게임번호 순으로 먼저 정렬한 뒤 시간으로 정렬 (정렬은 안정적)
        fx_first = {}
        for m in filtered_matches:
            fx_first[_fixture(m)] = min(fx_first.get(_fixture(m), 10 ** 9), _seq_num(m))

        def _ordered(ms, newest_first=False):
            base = sorted(ms, key=lambda m: (fx_first[_fixture(m)], _seq_num(m)))
            return sorted(base, key=_time_key, reverse=newest_first)

        # 고를 수 있는 경기(시간순) → 결과 대기(시간순) → 끝난 경기(최근부터)
        open_ms = _ordered([m for m in filtered_matches if _section(m) == "open"])
        wait_ms = _ordered([m for m in filtered_matches if _section(m) == "wait"])
        done_ms = _ordered([m for m in filtered_matches if _section(m) == "done"], newest_first=True)
        filtered_matches = open_ms + wait_ms + done_ms

        LOAD_STEP = 25
        # 필터가 바뀌면 처음 25경기부터 (회차·날짜·검색·필터·결과 수가 같을 때만 지금까지 불러온 만큼 유지)
        list_sig = (sel_round, sel_day, search_kw, sel_league, tuple(sel_types), tuple(sel_sports), only_open,
                    only_single_opt, only_rec_opt, only_picked, len(filtered_matches))
        if st.session_state.get("step1_list_sig") != list_sig:
            st.session_state["step1_list_sig"] = list_sig
            st.session_state["step1_shown"] = LOAD_STEP
        shown = min(int(st.session_state.get("step1_shown", LOAD_STEP)), len(filtered_matches))
        page_matches = filtered_matches[:shown]
        count_slot.markdown(
            f"<div style='font-size:13px;opacity:.75'>조건에 맞는 경기 <b>{len(filtered_matches):,}</b>개</div>",
            unsafe_allow_html=True)

        def _load_more():
            st.session_state["step1_shown"] = st.session_state.get("step1_shown", LOAD_STEP) + LOAD_STEP

        def _load_more_footer():
            """목록 끝의 '더 보기' 버튼 + 화면에 보이면 자동으로 누르는 스크립트 (무한 스크롤)"""
            if shown >= len(filtered_matches):
                if len(filtered_matches) > LOAD_STEP:
                    st.caption(f"— 마지막 경기까지 모두 표시했습니다 ({len(filtered_matches):,}경기) —")
                return
            left = len(filtered_matches) - shown
            st.button(f"▼ 더 보기 ({min(LOAD_STEP, left)}경기 더 · 남은 {left:,}경기)", key="step1_load_more",
                      width="stretch", on_click=_load_more)
            # 같은 출처 iframe에서 부모 문서의 '더 보기' 버튼을 감시 → 화면에 들어오면 클릭. 실패해도 버튼은 수동으로 동작
            components.html("""
<script>
(function () {
  const doc = window.parent.document;
  if (doc.__spInfScroll) { doc.__spInfScroll.disconnect(); }
  let busy = false;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting && !busy) {
        busy = true;
        setTimeout(() => { e.target.click(); }, 150);
      }
    });
  }, { rootMargin: "300px" });
  doc.__spInfScroll = observer;
  const find = () => Array.from(doc.querySelectorAll("button")).find((b) => b.innerText.includes("더 보기 ("));
  let tries = 0;
  const timer = setInterval(() => {
    const btn = find();
    if (btn) { observer.observe(btn); clearInterval(timer); }
    if (++tries > 40) clearInterval(timer);
  }, 250);
})();
</script>
""", height=0)

        # 회차 채점 요약 (현재 필터 기준). 배당 1순위는 모든 종료 경기, AI 1순위는 경기 전 기록이 있는 경기만
        mk_hits, ai_pairs = [], []
        for m in filtered_matches:
            res = _res(m)
            actual = RESULT_CODE_TO_KEY.get(str(res.get("game_result"))) if res is not None else None
            if not actual:
                continue
            opts = available_picks(m)
            log = _log(m)
            mk_k, _ = top_pick((log or {}).get("market_probs") or m.get("market_implied_probs"), opts)
            if mk_k:
                mk_hits.append(mk_k == actual)
            if log and log.get("model_probs"):
                ai_k, _ = top_pick(log["model_probs"], opts)
                ai_pairs.append((ai_k == actual, mk_k == actual))
        if filtered_matches:
            n_mk, n_ai = len(mk_hits), len(ai_pairs)
            mk_hit, ai_hit, ai_mk_hit = sum(mk_hits), sum(a for a, _ in ai_pairs), sum(b for _, b in ai_pairs)
            tiles = [
                ("🟢 지금 고를 수 있음", f"{len(open_ms):,}경기", "발매 마감 전" if open_ms else "발매 중인 경기 없음"),
                ("⏳ 결과 대기", f"{len(wait_ms):,}경기", "발매 마감 · 진행 중"),
                ("🏁 결과 나옴", f"{len(done_ms):,}경기", "목록 맨 뒤, 최근 경기부터" if done_ms else "아직 없음"),
                ("📈 배당 1순위 적중", f"{mk_hit}/{n_mk}" if n_mk else "—",
                 f"{mk_hit / n_mk * 100:.0f}% · 배당이 가장 낮은 쪽" if n_mk else "결과가 나오면 표시"),
                ("🧠 AI 1순위 적중", f"{ai_hit}/{n_ai}" if n_ai else "—",
                 f"{ai_hit / n_ai * 100:.0f}% · 같은 경기 배당 1순위 {ai_mk_hit}/{n_ai}" if n_ai
                 else ("결과 나온 경기에 경기 전 AI 기록 없음" if n_mk else "결과가 나오면 표시")),
            ]
            render_tiles(tiles)
            if n_mk:
                st.caption("💡 적중률은 참고용이에요. 배당 1순위만 찍어도 꽤 맞으므로 AI는 '같은 경기'의 배당 1순위와 비교하세요. "
                           "AI 실력의 공식 지표는 ③ 성과 검증 탭의 성적표(Log loss)입니다.")

        if not page_matches:
            open_any = [m for m in predictions if not is_sale_closed(m, now_ts)]
            if only_open and not open_any:
                closes = [float(m["close_time"]) for m in predictions if m.get("close_time")]
                last = datetime.fromtimestamp(max(closes) / 1000).strftime("%m/%d %H:%M") if closes else "-"
                st.info(f"⏳ 지금은 발매 중인(마감 전) 경기가 없습니다. 현재 회차 발매가 {last}에 모두 마감되었고 다음 회차는 아직 열리지 않았습니다. "
                        "다음 회차가 열리면 하루 2번(09:30·18:30) 자동 수집 때 표시됩니다.")
            else:
                st.info("조건에 맞는 경기가 없습니다. 날짜를 '전체 날짜'로 바꾸거나 필터를 초기화해 보세요.")
                st.button("↺ 필터 초기화", key="step1_reset_empty", on_click=_reset_filters)

        def _prob(probs, k):
            return float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0) if probs else 0.0

        def _mark(pick, actual):
            if not pick or not actual:
                return ""
            ok = pick == actual
            return f"<span style='color:{'#10b981' if ok else '#ef4444'}'>{'✅' if ok else '❌'}</span>"

        cap = "<div style='font-size:11px;color:#94a3b8'>{}</div>"
        section_shown = set()

        # 대진 머리줄의 최종 점수: 핸디캡이 더해지지 않은 유형(승무패 → 홀짝 → 언더오버)의 점수
        SCORE_PRIORITY = {"승무패": 0, "홀짝": 1, "언더오버": 2}
        fx_score = {}
        for m in sorted((m for m in round_matches if type_badge(m) in SCORE_PRIORITY and _res(m) is not None),
                        key=lambda m: SCORE_PRIORITY[type_badge(m)]):
            if _res(m).get("score"):
                fx_score.setdefault(_fixture(m), _res(m)["score"])

        def _fixture_header(m, section):
            when = short_when(m)
            rnd = f"{_ts(m)}회 · " if is_all else ""
            if section == "open":
                ct = m.get("close_time")
                status = f"🟢 {datetime.fromtimestamp(float(ct) / 1000).strftime('%m.%d %H:%M')} 마감" if ct else "🟢 발매 중"
            elif section == "wait":
                status = "⏳ 발매 마감 · 결과 대기"
            else:
                status = f"🏁 최종 {fx_score[_fixture(m)]}" if _fixture(m) in fx_score else "🏁 종료"
            # 바깥 padding: Streamlit 글 블록은 아래 여백을 -1rem 당겨서, 여백이 없으면 첫 줄 배당 버튼이 머리줄 선에 붙는다
            return ("<div style='padding-bottom:14px'>"
                    "<div style='display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;padding-bottom:6px;"
                    "border-bottom:1px solid rgba(148,163,184,.25)'>"
                    f"<span style='font-size:12px;opacity:.75'>{SPORT_ICONS.get(m.get('sport'), '')} {rnd}{when}</span>"
                    + league_chip(m.get("league", "")) +
                    f"<span style='font-size:16px'><b>{m.get('home_team', '')}</b> <span style='opacity:.6'>vs</span> "
                    f"<b>{m.get('away_team', '')}</b></span>"
                    f"<span style='margin-left:auto;font-size:12px;opacity:.85'>{status}</span></div></div>")

        cur_fx, fx_box, fx_rows = None, None, 0

        for m in page_matches:
            seq = str(m.get("match_seq", ""))
            odds = m.get("betman_odds", {})
            labels = pick_labels(m)
            opts = available_picks(m)
            if not opts:
                continue
            log = _log(m)
            res = _res(m)

            # 목록 구간 제목: 지금 고를 수 있음 → 결과 대기 → 결과 나옴
            section = _section(m)
            if section not in section_shown:
                section_shown.add(section)
                title, n_sec, hint = {
                    "open": ("🟢 지금 고를 수 있는 경기", len(open_ms), "배당 버튼을 눌러 고르세요 · 시간순"),
                    "wait": ("⏳ 결과 대기", len(wait_ms), "발매 마감 · 결과가 나오면 적중 여부 표시"),
                    "done": ("🏁 결과 나온 경기", len(done_ms), "최근 경기부터 · ✅❌ = 적중 여부"),
                }[section]
                st.markdown(f"<div style='margin:14px 0 4px;font-size:15px'><b>{title}</b> "
                            f"<span style='opacity:.7'>{n_sec:,}경기 · {hint}</span></div>", unsafe_allow_html=True)

            # 배당 예상: 경기 전에 확정된 배당(마진 제거) 기준 → 사후 정보 없음
            mk_probs = (log or {}).get("market_probs") or m.get("market_implied_probs")
            mk_k, mk_p = top_pick(mk_probs, opts)
            ai_probs = _ai_probs(m, res, log)
            ai_k, ai_p = top_pick(ai_probs, opts)
            gems = _gems(m, ai_probs, opts)

            cur = st.session_state["selections"].get(seq, {})
            default = [k for k in opts if cur.get(SEL_KEY[k])]
            star_k = ai_k or mk_k
            disp = {k: f"{'💎' if k in gems else ''}{'⭐' if k == star_k else ''}{labels[k]} {float(odds[k]):.2f}" for k in opts}
            closed = is_sale_closed(m, now_ts) and not _past(m)
            actual = RESULT_CODE_TO_KEY.get(str(res.get("game_result"))) if res is not None else None

            # 대진 묶음: 같은 대진이 이어지면 한 상자 안에 머리줄(시간·리그·팀·상태) 한 번 + 게임번호 줄들
            if (section, _fixture(m)) != cur_fx:
                cur_fx = (section, _fixture(m))
                fx_box = st.container(border=True)
                fx_box.markdown(_fixture_header(m, section), unsafe_allow_html=True)
                fx_rows = 0

            with fx_box:
                if fx_rows:   # 같은 대진 안의 게임번호 줄 사이 얇은 구분선
                    st.markdown("<div style='border-top:1px solid rgba(148,163,184,.12);margin:-6px 0'></div>",
                                unsafe_allow_html=True)
                c1, c2, c3 = st.columns([2.2, 3.0, 4.6], vertical_alignment="center")
                with c1:
                    # 게임번호 · 유형 · 기준선 칩 · 한경기 (마우스를 올리면 리그·유형 전체 표시)
                    tags = line_chip(m) + (ONE_GAME_TAG if m.get("is_single") else "")
                    st.markdown(
                        f"<div title='#{seq} · {m.get('league', '')} · {category_display(m)}' style='display:flex;"
                        "align-items:center;gap:8px;white-space:nowrap;overflow:hidden'>"
                        f"<span style='font-size:15px;font-weight:800;min-width:34px'>{seq}</span>"
                        f"<span style='font-size:15px;font-weight:600'>{type_name(m)}</span>{tags}</div>",
                        unsafe_allow_html=True)
                with c2:
                    picked = st.pills(
                        f"선택 #{seq}", options=opts, format_func=lambda k, d=disp: d[k], selection_mode="multi",
                        default=default, key=f"pick_{_ts(m)}_{m.get('_key', seq)}_{st.session_state['editor_version']}",
                        label_visibility="collapsed", disabled=(res is not None or _past(m) or closed))
                    ev = m.get("evaluation", {})
                    if res is None and GEM_ON and ev.get("is_recommended"):
                        st.markdown(f"<div style='font-size:12px;margin-top:-6px'>🔥 가치 픽 {ev.get('best_ev_pct', '')}</div>",
                                    unsafe_allow_html=True)
                # 배당 예상 · AI 예상 · 점수 · 결과는 한 칸 안에 가로 4칸으로 (휴대폰에서도 세로로 길게 쌓이지 않게)
                mk_body = f"<b>{labels[mk_k]}</b> {mk_p * 100:.0f}% {_mark(mk_k, actual)}" if mk_k else "-"
                if ai_k and ai_k == mk_k:
                    delta = (ai_p - _prob(mk_probs, ai_k)) * 100
                    arrow = "▲" if delta >= 0.5 else ("▼" if delta <= -0.5 else "")
                    ai_body = f"<b>{labels[ai_k]}</b> {ai_p * 100:.0f}% <span style='font-size:11px;color:#94a3b8'>{arrow}{abs(delta):.0f}</span>" if arrow \
                        else f"<b>{labels[ai_k]}</b> {ai_p * 100:.0f}%"
                elif ai_k:
                    ai_body = f"<span style='color:#f59e0b'>⚠️ <b>{labels[ai_k]}</b> {ai_p * 100:.0f}% 다름</span>"
                elif res is not None and m.get("game_type") == "축구 승무패":
                    ai_body = "<span style='color:#94a3b8;font-size:12px'>경기 전 기록 없음</span>"
                else:
                    ai_body = "<span style='color:#94a3b8'>-</span>"
                if ai_k:
                    ai_body += f" {_mark(ai_k, actual)}"
                for gk, gev in sorted(gems.items(), key=lambda x: -x[1]):
                    ai_body += (f"<div style='font-size:12px;color:#38bdf8'>💎 <b>{labels[gk]}</b> {_prob(ai_probs, gk) * 100:.0f}% · "
                                f"EV {gev * 100:+.1f}% {_mark(gk, actual)}</div>")
                score = res.get("score", "") if res is not None else "-"
                note = " (핸디)" if res is not None and "핸디캡" in str(m.get("category", "")) else ""
                if actual:
                    res_body = f"<div style='font-size:15px'><b>{labels.get(actual, '?')}</b></div>"
                else:
                    res_body = ("<div style='font-size:13px;color:#94a3b8'>마감 · 결과 대기</div>" if closed
                                else "<div style='font-size:13px;color:#94a3b8'>진행 전</div>")
                cell = "<div style='flex:{} 1 0;min-width:{}px'>{}{}</div>"
                c3.markdown(
                    "<div style='display:flex;gap:10px;align-items:flex-start'>"
                    + cell.format(1.0, 62, cap.format("배당 예상"), f"<div style='font-size:14px'>{mk_body}</div>")
                    + cell.format(1.5, 92, cap.format("AI 예상"), f"<div style='font-size:14px'>{ai_body}</div>")
                    + cell.format(0.9, 48, cap.format("점수" + note), f"<div style='font-size:16px;font-weight:700'>{score}</div>")
                    + cell.format(1.2, 62, cap.format("결과"), res_body)
                    + "</div>", unsafe_allow_html=True)
                fx_rows += 1

            if res is None and not _past(m) and not closed:
                picked = picked or []
                if picked:
                    st.session_state["selections"][seq] = {SEL_KEY[k]: (k in picked) for k in PICK_KEYS}
                elif seq in st.session_state["selections"]:
                    del st.session_state["selections"][seq]

        _load_more_footer()

        if live_on and current_gm_ts:
            @st.fragment(run_every="60s")
            def _live_tick():
                rows_ = fetch_live_round_results(str(current_gm_ts)) if started_without_result(
                    predictions, load_results_index(), str(current_gm_ts)) else []
                n = len(rows_)
                prev = st.session_state.get("live_result_count")
                st.session_state["live_result_count"] = n
                st.caption(f"🔴 실시간 결과 확인 {datetime.now().strftime('%H:%M:%S')} · 현재 회차 결과 {n}건"
                           if rows_ else f"🔴 실시간 결과 대기 중 ({datetime.now().strftime('%H:%M')}) — 진행 중이거나 결과 대기 중인 경기가 없으면 조회하지 않습니다")
                if prev is not None and n > prev:
                    st.rerun(scope="app")
            _live_tick()
        # 선택 요약·규정 검사·STEP 02 이동은 사이드바 🛒 내 선택에서 (목록 맨 아래 요약은 중복이라 없앰)

    # -------------------------------------------------------------
    # STEP 02: 조합 결과 및 시뮬레이션 화면
    # -------------------------------------------------------------
    else:
        # 선택 경기 데이터 → 직교 전개
        active_selections = {}
        for seq, sel in st.session_state["selections"].items():
            if seq not in pred_map:
                continue
            m = pred_map[seq]
            odds = m.get("betman_odds", {})
            probs = m.get("blended_probabilities") or m.get("market_implied_probs") or {}
            picks = []
            for k in PICK_KEYS:
                if sel.get(SEL_KEY[k]) and float(odds.get(k, 0) or 0) > 1.0:
                    picks.append({
                        "selection": k,
                        "selection_name": pick_labels(m)[k],
                        "odds": odds[k],
                        "prob": float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0),
                    })
            if picks:
                active_selections[seq] = {
                    "match_seq": seq,
                    "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}",
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "league": m.get("league", ""),
                    "game_type": category_display(m),
                    "is_single": m.get("is_single", False),
                    "picks": picks,
                }

        combos = expand_combinations(active_selections)

        h1, h2 = st.columns([4, 1.2], vertical_alignment="center")
        with h1:
            st.markdown("### 🎯 STEP 02. 조합 결과 & 베팅 시뮬레이션")
        with h2:
            st.button("⬅️ 경기 다시 고르기", width="stretch", on_click=_go_step, args=(1,))

        if not combos:
            st.warning("고른 경기가 없어요. ① 경기 고르기에서 배당 버튼을 눌러 주세요.")
        else:
            is_single_only = len(active_selections) == 1
            unit = 1000 if is_single_only else 100
            sel_sig = "|".join(f"{s}:{''.join(p['selection'][0] for p in v['picks'])}" for s, v in sorted(active_selections.items()))

            # 1) 고른 경기: 번호 · 유형 · 기준선 · 리그 · 팀 · 고른 선택지(배당 · 확률) — 한 덩어리로 그려 줄 간격을 고르게
            unit_txt = "한경기 1,000원" if is_single_only else "조합 100원"
            chip_css = "display:inline-block;background:#1e3a5f;border-radius:12px;padding:2px 10px;margin-left:4px;font-size:13px"
            match_rows = []
            for seq, v in active_selections.items():
                m = pred_map[seq]
                chips = "".join(
                    f"<span style='{chip_css}'><b>{p['selection_name']}</b> {float(p['odds']):.2f} "
                    f"<span style='opacity:.7'>· {p['prob'] * 100:.0f}%</span></span>" for p in v["picks"])
                match_rows.append(
                    "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;"
                    "border:1px solid rgba(148,163,184,.25);border-radius:8px;margin-bottom:6px'>"
                    f"<span style='font-size:15px;font-weight:800'>{seq}</span>"
                    f"<span style='font-size:14px;font-weight:600'>{type_name(m)}</span>{line_chip(m)}"
                    f"{ONE_GAME_TAG if m.get('is_single') else ''}{league_chip(m.get('league', ''))}"
                    f"<span style='font-size:15px;word-break:keep-all'><b>{v['home_team']}</b> "
                    f"<span style='opacity:.6'>vs</span> <b>{v['away_team']}</b></span>"
                    f"<span style='margin-left:auto'>{chips}</span></div>")
            st.markdown(
                f"<div style='padding-bottom:12px'><div style='margin-bottom:6px'><b>고른 경기 {len(active_selections)}개</b> · "
                f"조합 {len(combos)}개 <span style='opacity:.65;font-size:13px'>({unit_txt} 단위 · % = 그 선택지의 확률)</span></div>"
                + "".join(match_rows) + "</div>", unsafe_allow_html=True)
            # 베트맨에서 살 수 없는 구성이면 여기서도 알림 (휴대폰에서는 사이드바 장바구니가 접혀 있음)
            rule_check = selection_status()
            if not rule_check["valid"]:
                st.warning(rule_check["msg"] + " 계산은 볼 수 있지만 베트맨에서는 이대로 살 수 없어요.")

            def _bump_editor():
                st.session_state["combo_editor_ver"] = st.session_state.get("combo_editor_ver", 0) + 1

            # 조합 번호 → 조합 내용 이름 (금액·체크 저장 키). 사이드바 장바구니도 같은 키를 읽는다
            ckey_of = {c["combo_index"]: combo_key((p["match_seq"], p["selection"]) for p in c["picks"]) for c in combos}

            # 2) 금액 · 정렬 · 빠른 선택 — 금액은 바꾸는 즉시 모든 조합에 적용 (표에서 조합별로 따로 바꿀 수도 있음)
            def _apply_stake(v):
                v = max(unit, min(100000, (int(v) // unit) * unit))
                st.session_state["batch_stake"] = v
                st.session_state["batch_stake_input"] = v
                for c in combos:
                    st.session_state[f"stake_{ckey_of[c['combo_index']]}"] = v
                _bump_editor()

            def _quick_select(mode):
                for c in combos:
                    st.session_state[f"combo_chk_{ckey_of[c['combo_index']]}"] = (
                        True if mode == "all" else False if mode == "none" else c["ev"] > 0)
                _bump_editor()

            # 한경기(1,000원 단위) ↔ 조합(100원 단위)이 바뀌어도 입력칸 값이 범위를 벗어나지 않게 (그리기 전에)
            st.session_state["batch_stake_input"] = max(unit, min(100000, (int(st.session_state.get(
                "batch_stake_input", st.session_state["batch_stake"])) // unit) * unit))
            presets = [1000, 5000, 10000, 50000] if is_single_only else [100, 1000, 5000, 10000]
            with st.container(border=True):
                a1, a2, a3 = st.columns([1.2, 2.3, 1.3], vertical_alignment="bottom")
                with a1:
                    st.number_input("조합당 금액 (원)", min_value=unit, max_value=100000, step=unit, key="batch_stake_input",
                                    on_change=lambda: _apply_stake(st.session_state["batch_stake_input"]),
                                    help="바꾸면 모든 조합에 바로 적용돼요. 아래 표에서 조합별로 따로 바꿀 수도 있어요.")
                with a2:
                    for col, v in zip(st.columns(len(presets)), presets):
                        col.button(f"{v:,}원", key=f"stake_preset_{v}", width="stretch", on_click=_apply_stake, args=(v,))
                with a3:
                    sort_option = st.selectbox("정렬", ["기본 순서", "배당 높은순", "배당 낮은순", "적중 확률 높은순", "기대가치 높은순"],
                                               key="step2_sort")
                q1, q2, q3, _ = st.columns([1, 1, 1.5, 1.3])
                q1.button("☑️ 전체 선택", width="stretch", on_click=_quick_select, args=("all",))
                q2.button("⬜ 전체 해제", width="stretch", on_click=_quick_select, args=("none",))
                if GEM_ON:   # [6.5] AI 기준 기대가치로 고르는 버튼은 추천 잠금 중 숨김
                    q3.button("💎 기대가치 플러스만", width="stretch", on_click=_quick_select, args=("ev",),
                              help="기대가치가 플러스인 조합만 선택")

            if sort_option == "배당 높은순":
                combos.sort(key=lambda x: x["combined_odds"], reverse=True)
            elif sort_option == "배당 낮은순":
                combos.sort(key=lambda x: x["combined_odds"])
            elif sort_option == "적중 확률 높은순":
                combos.sort(key=lambda x: x["prob_product"], reverse=True)
            elif sort_option == "기대가치 높은순":
                combos.sort(key=lambda x: x["ev"], reverse=True)

            # 3) 요약 자리 (표 편집 결과를 반영하려고 표 다음에 채운다)
            summary_box = st.container()

            # 4) 조합 표 — 경기마다 한 열, 선택·금액 직접 수정
            match_cols = {seq: f"#{seq} {v['home_team'][:8]}" for seq, v in active_selections.items()}
            default_stake = max(unit, (int(st.session_state["batch_stake"]) // unit) * unit)
            rows = []
            for c in combos:
                idx, ck = c["combo_index"], ckey_of[c["combo_index"]]
                st.session_state.setdefault(f"combo_chk_{ck}", True)
                st.session_state.setdefault(f"stake_{ck}", default_stake)
                row = {"선택": bool(st.session_state[f"combo_chk_{ck}"]), "#": idx}
                for p in c["picks"]:
                    row[match_cols[str(p["match_seq"])]] = f"{p['selection_name']} {float(p['odds']):.2f}"
                stake = max(unit, (int(st.session_state[f"stake_{ck}"]) // unit) * unit)
                payout = calculate_payout(c["combined_odds"], stake)
                row.update({
                    "배당": round(c["combined_odds"], 2),
                    "투표금액": stake,
                    "적중 시 환급": payout,
                    "적중 확률": round(c["prob_product"] * 100, 2),
                    "기대가치": round(c["ev"] * 100, 1),
                    "비과세": "✅" if is_tax_free(c["combined_odds"], payout) else "과세",
                })
                rows.append(row)

            col_cfg = {
                "선택": st.column_config.CheckboxColumn("선택", width="small"),
                "#": st.column_config.NumberColumn("#", width="small"),
                "배당": st.column_config.NumberColumn("배당", format="%.2f배", width="small"),
                "투표금액": st.column_config.NumberColumn("투표금액(원) ✏️", min_value=unit, max_value=100000, step=unit,
                                                     format="localized", help=f"{unit:,}원 단위로 직접 수정할 수 있습니다"),
                "적중 시 환급": st.column_config.NumberColumn("적중 시 환급(원)", format="localized"),
                "적중 확률": st.column_config.NumberColumn("적중 확률", format="%.2f%%", width="small"),
                "기대가치": st.column_config.NumberColumn("기대가치", format="%+.1f%%", width="small"),
                "비과세": st.column_config.TextColumn("비과세", width="small"),
            }
            for name in match_cols.values():
                col_cfg[name] = st.column_config.TextColumn(name, width="small")

            editor_key = f"combo_editor_{sel_sig}_{sort_option}_{st.session_state.get('combo_editor_ver', 0)}"
            row_ids = [ckey_of[c["combo_index"]] for c in combos]

            def _sync_editor(key=editor_key, ids=row_ids, unit_=unit):
                """편집 직후(재실행 전) 세션에 반영하고 표를 새로 그린다.
                이렇게 하지 않으면 '적중 시 환급' 같은 계산 열이 이전 금액 기준으로 한 번 늦게 표시된다."""
                for pos, ch in st.session_state.get(key, {}).get("edited_rows", {}).items():
                    ck = ids[int(pos)]
                    if "선택" in ch:
                        st.session_state[f"combo_chk_{ck}"] = bool(ch["선택"])
                    if ch.get("투표금액") is not None:
                        v = int(ch["투표금액"])
                        st.session_state[f"stake_{ck}"] = max(unit_, min(100000, (v // unit_) * unit_))
                _bump_editor()

            edited = st.data_editor(
                pd.DataFrame(rows), hide_index=True, width="stretch", column_config=col_cfg,
                disabled=[k for k in rows[0] if k not in ("선택", "투표금액")],
                key=editor_key, on_change=_sync_editor,
                height=min(38 + 35 * len(rows), 460),
            )

            # 편집 결과를 세션에 반영하고 합계 계산
            by_idx = {c["combo_index"]: c for c in combos}
            chosen = []
            for _, r in edited.iterrows():
                idx = int(r["#"])
                stake = int(r["투표금액"]) if pd.notna(r["투표금액"]) else unit
                stake = max(unit, min(100000, (stake // unit) * unit))
                st.session_state[f"combo_chk_{ckey_of[idx]}"] = bool(r["선택"])
                st.session_state[f"stake_{ckey_of[idx]}"] = stake
                if bool(r["선택"]):
                    c = by_idx[idx]
                    chosen.append((c, stake, calculate_payout(c["combined_odds"], stake)))

            total_stake = sum(s for _, s, _ in chosen)
            exp_return = sum(c["prob_product"] * pay for c, _, pay in chosen)
            hit_any = sum(c["prob_product"] for c, _, _ in chosen)   # 조합끼리는 동시에 맞을 수 없음(배타적)
            pays = [pay for _, _, pay in chosen]
            exp_pnl = exp_return - total_stake

            with summary_box:
                pay_txt = "-" if not pays else (f"{pays[0]:,}원" if min(pays) == max(pays) else f"{min(pays):,} ~ {max(pays):,}원")
                pnl_pct = exp_pnl / total_stake * 100 if total_stake else 0.0
                pnl_color = "#f87171" if exp_pnl < 0 else "#34d399"
                render_tiles([
                    ("💰 총 투표금액", f"{total_stake:,}원", f"{len(chosen)} / {len(combos)} 조합 선택"),
                    ("🎯 적중 시 환급", pay_txt, "맞은 조합에 따라 달라요" if len(set(pays)) > 1 else "고른 조합이 맞으면"),
                    ("🎲 하나라도 맞을 확률", f"{hit_any * 100:.1f}%", "조합끼리는 동시에 맞지 않아요"),
                    ("📊 기대 손익", f"<span style='color:{pnl_color}'>{int(exp_pnl):+,}원</span>",
                     f"{pnl_pct:+.1f}% · 아주 많이 반복했을 때 한 번 평균"),
                ])
                if total_stake > 100000:
                    st.error("⚠️ 총 투표금액이 베트맨 1인 1회차 구매 한도(100,000원)를 넘어요.")
                if chosen:
                    n_m = len(active_selections)
                    # 경기별 환급률 = 1 ÷ (배당 역수의 합). 베트맨은 대부분 1 ÷ 1.15 ≈ 87%, 홀짝은 1 ÷ 1.11 ≈ 90%
                    pay_ratio = 1.0
                    for seq_ in active_selections:
                        o_ = [float(x) for x in pred_map[seq_].get("betman_odds", {}).values() if float(x or 0) > 1.0]
                        pay_ratio *= 1.0 / sum(1.0 / x for x in o_) if o_ else 1.0
                    why = (f" 베트맨은 경기마다 배당에 약 15%(홀짝은 11%)의 마진을 붙여서, 한 경기당 평균 약 87%만 돌려줘요. "
                           f"{n_m}경기를 묶으면 이게 곱해져 평균 환급률이 약 {pay_ratio * 100:.0f}%예요 (조합 배당 올림 덤은 빼고)."
                           if exp_pnl < 0 else " AI 확률 기준이며 아직 검증 중이에요.")
                    # 바깥 padding: Streamlit 글 블록의 -1rem 아래 여백 때문에 아래 캡션과 붙지 않게
                    st.markdown(
                        "<div style='padding-bottom:10px'><div style='font-size:14px;line-height:1.7'>"
                        f"💬 고른 {len(chosen)}개 조합 중 하나가 맞을 확률은 <b>{hit_any * 100:.1f}%</b>이고, 맞으면 <b>{pay_txt}</b>을 돌려받아요. "
                        f"같은 방식을 아주 많이 반복하면 한 번에 평균 <b style='color:{pnl_color}'>{int(exp_pnl):+,}원 ({pnl_pct:+.1f}%)</b>이에요."
                        f"{why}</div></div>", unsafe_allow_html=True)
                else:
                    st.info("표에서 조합을 하나 이상 선택해 주세요.")
                st.caption("확률: AI 적용 유형(② 탭 맨 위 참고)은 AI 확률, 그 외 유형은 배당(마진 제거) 확률이에요. "
                           "표의 선택·투표금액 ✏️ 칸은 직접 바꿀 수 있어요.")

            # 4-1) 이 방식을 과거에 반복했다면? (실제 결과 기준 백테스트)
            with st.container(border=True):
                bt_c1, bt_c2 = st.columns([3, 1.3], vertical_alignment="center")
                rule = extract_rule(active_selections, pred_map)
                with bt_c1:
                    st.markdown("##### 🔁 이 방식을 과거에 반복했다면?")
                    if rule:
                        st.caption(f"규칙: {describe_rule(rule)} — 팀과 상관없이, 과거 회차마다 배당이 가장 비슷한(±20%) 경기로 "
                                   "똑같은 모양으로 걸고 **실제 결과**로 계산해요.")
                    else:
                        st.caption("축구 일반 승무패 경기로만 구성된 조합에서 사용할 수 있습니다 (과거 데이터가 축구 승무패만 있음).")
                with bt_c2:
                    run_bt = st.button("과거 결과로 확인", width="stretch", disabled=not rule, key="run_backtest")
                if run_bt and rule:
                    st.session_state["bt_result"] = (describe_rule(rule), backtest(rule, load_archive_for_backtest()))
                if st.session_state.get("bt_result") and rule and st.session_state["bt_result"][0] == describe_rule(rule):
                    r = st.session_state["bt_result"][1]
                    if not r.get("rounds"):
                        st.warning("조건에 맞는 과거 회차가 없습니다. 배당 범위가 너무 특이한 조합일 수 있습니다.")
                    else:
                        ci_r = r.get("roi_ci")
                        roi_color = "#f87171" if r["roi_pct"] < 0 else "#34d399"
                        render_tiles([
                            ("🗓️ 적용된 과거 회차", f"{r['rounds']}회차", f"회차당 {r['combos_per_round']}조합 · 조합마다 1단위"),
                            ("🎯 적중한 회차", f"{r['hit_rate_pct']:.0f}%", f"{r['hit_rounds']}회 적중 · 최고 {r['best_round_multiple']:.1f}배"),
                            ("📊 실제 수익률", f"<span style='color:{roi_color}'>{r['roi_pct']:+.1f}%</span>",
                             (f"95% 범위 {ci_r[0]:+.1f}% ~ {ci_r[1]:+.1f}%" if ci_r else "")
                             + (f" · 축 경기만 {r['bankers_only_roi_pct']:+.1f}%" if "bankers_only_roi_pct" in r else "")),
                            ("📉 최대 연속 미적중", f"{r['max_losing_streak']}회차", "이만큼 연달아 못 맞힐 수 있어요"),
                        ])
                        if ci_r:
                            if ci_r[1] < 0:
                                verdict_bt = "📉 범위 전체가 마이너스 → 이 모양은 <b>오래 하면 확실히 손해</b>예요."
                            elif ci_r[0] > 0:
                                verdict_bt = ("📈 범위 전체가 플러스 → 과거에는 이익이었어요. 다만 같은 데이터를 보고 고른 규칙은 "
                                              "우연히 좋아 보일 수 있어요.")
                            else:
                                verdict_bt = "⚖️ 범위에 0이 들어 있어 <b>아직 이익인지 손해인지 판단하기 어려워요</b>."
                            st.markdown(f"<div style='padding-bottom:10px'><div style='font-size:14px;line-height:1.7'>{verdict_bt}<br>"
                                        "<span style='opacity:.75;font-size:13px'>배당 모양만 본 결과예요. AI나 팀 정보로 경기를 고른 효과는 "
                                        "③ 성과 검증 탭의 가상 베팅 리그에서 확인하세요.</span></div></div>", unsafe_allow_html=True)
                        st.line_chart(pd.DataFrame({"누적 손익 (회차당 투자 1 기준)": r["cum_curve"]}), height=180)
                        notes = []
                        if "bankers_only_roi_pct" in r and r["bankers_only_roi_pct"] > r["roi_pct"]:
                            notes.append("커버(2~3개 선택)를 붙이면 그 경기의 마진을 추가로 내게 되어 **축 경기만 걸 때보다 손실이 커집니다.**")
                        if r["hit_rate_pct"] >= 50 and r["roi_pct"] < 0:
                            notes.append("자주 맞지만 장기적으로는 잃는 방식입니다. 적중률보다 **수익률**을 보세요.")
                        if r["roi_pct"] >= 0:
                            notes.append("과거에는 플러스였지만, 같은 데이터로 규칙을 고르면 우연히 좋아 보일 수 있습니다. 회차 수가 충분한지 함께 확인하세요.")
                        for n in notes:
                            st.caption("💡 " + n)

            # 5) 베팅 슬립 (복사용)
            with st.expander(f"📋 베팅 슬립 복사 ({len(chosen)}개 조합)", expanded=False):
                lines = ["📋 [베트맨 프로토 승부식 맞춤조건구매 슬립]",
                         f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')} · 선택 {len(chosen)} / 전체 {len(combos)} 조합",
                         "────────────────────────────────────────"]
                for c, stake, pay in chosen:
                    parts = [f"#{p['match_seq']} {p['home_team']} [{p['selection_name']} {p['odds']}]" for p in c["picks"]]
                    lines.append(f"🎫 조합 #{c['combo_index']}: " + " × ".join(parts))
                    lines.append(f"   ↳ 배당 {c['combined_odds']}배 · {stake:,}원 → 적중 시 {pay:,}원 "
                                 f"(확률 {c['prob_pct']}, EV {c['ev_pct']})")
                lines += ["────────────────────────────────────────",
                          f"💰 총 투표금액 {total_stake:,}원 · 기대 손익 {int(exp_pnl):+,}원"]
                st.code("\n".join(lines), language="text")

            # 6) 켈리 자금 배분 (고급)
            with st.expander("📐 켈리 기준 자금 배분 (고급)", expanded=False):
                st.caption("기대 손익이 플러스일 때만 베팅을 권장합니다. 변동성을 줄이려면 1/4 켈리 이하를 권장합니다.")
                kc1, kc2, kc3 = st.columns([1.5, 1.5, 1.2])
                with kc1:
                    bankroll = st.number_input("보유 자금 (원)", min_value=10000, max_value=100000000,
                                               value=st.session_state.get("user_bankroll", 200000), step=10000,
                                               key="custom_bankroll_input")
                    st.session_state["user_bankroll"] = bankroll
                with kc2:
                    kelly_mode = st.selectbox("켈리 비율", ["1/4 켈리 (권장)", "1/2 켈리", "1/8 켈리"], key="kelly_mode")
                    k_mult = {"1/4 켈리 (권장)": 0.25, "1/2 켈리": 0.5, "1/8 켈리": 0.125}[kelly_mode]
                raw_k = 0.0
                if chosen and total_stake and exp_pnl > 0 and hit_any > 0:
                    # 적중 시 평균 환급 배수(투표금 대비)로 단일 베팅처럼 근사
                    avg_mult = exp_return / hit_any / total_stake
                    b_val = max(1e-9, avg_mult - 1.0)
                    raw_k = max(0.0, (hit_any * b_val - (1 - hit_any)) / b_val)
                app_k = min(0.05, raw_k * k_mult)
                k_stake = min(100000, max(0, (int(bankroll * app_k) // unit) * unit))
                with kc3:
                    st.metric("권장 총 베팅액", f"{k_stake:,}원")
                    st.caption(f"자금의 {app_k * 100:.2f}%")
                if k_stake == 0:
                    st.info("기대 손익이 플러스가 아니어서 켈리 기준 권장 베팅액은 0원입니다.")
                elif chosen:
                    per = max(unit, (k_stake // len(chosen) // unit) * unit)
                    if st.button(f"👉 선택 조합에 균등 적용 ({per:,}원씩)"):
                        for c, _, _ in chosen:
                            st.session_state[f"stake_{ckey_of[c['combo_index']]}"] = per
                        _bump_editor()
                        st.rerun()


# ==============================================================================
# TAB 2: 🔍 ② AI 경기 분석 (조회 전용)
# ==============================================================================
with tab2:
    if not predictions:
        st.warning("⚠️ 아직 분석된 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # A. AI 적용 상태 한 줄 + 검증 수치는 펼쳐서 (REVIEW R3: 시장 대비 성적을 항상 투명하게)
        status = load_model_status()
        applied = (["축구 승무패"] if status else []) + [t for t in markets_applied_types() if t != "축구 승무패"]
        if applied:
            # 바깥 padding: Streamlit 글 블록의 -1rem 아래 여백 때문에 아래 펼침 상자와 붙지 않게
            st.markdown(
                "<div style='padding-bottom:12px'><div style='padding:10px 14px;border-radius:8px;"
                "background:rgba(59,130,246,.12);font-size:14px;line-height:1.6'>"
                f"🧠 <b>AI 적용 유형:</b> {' · '.join(applied)} "
                "<span style='opacity:.75'>│ 나머지 유형은 배당 기준 · 모두 <b>검증 중</b>이라 참고용이에요</span></div></div>",
                unsafe_allow_html=True)
        else:
            st.warning("⚠️ 검증을 통과한 AI가 없어 모든 확률이 배당 기준이에요.")
        if status:
            with st.expander("검증 수치 자세히 (과거 시험 · Log loss)"):
                for label, s in status["evals"]:
                    ci = s.get("ll_diff_ci", [None, None])
                    st.markdown(f"- **{status['name']} · {label}** ({s['samples']:,}경기): AI `{s['model_ll']:.4f}` vs 배당 "
                                f"`{s['market_ll']:.4f}` (차이 `{s['ll_diff']:+.4f}`, 95% 범위 `[{ci[0]:+.4f}, {ci[1]:+.4f}]`)")
                st.caption("Log loss는 낮을수록 정확하고, 차이가 음수면 AI가 배당보다 정확했다는 뜻이에요. "
                           "유형별 시험 결과와 실전 성적은 ③ 성과 검증 탭에서 확인하세요.")

        # B. 경기마다 배당 확률 vs AI 확률 비교
        DIFF_PP = 3.0   # 'AI가 다르게 본 경기': 1순위가 다르거나 어떤 선택지든 3%p 이상 차이 (AI 적용 경기의 약 1/4)
        t2_now = datetime.now().timestamp() * 1000

        def _analyze(m):
            opts = available_picks(m)
            odds = m.get("betman_odds", {})
            mk = m.get("market_implied_probs") or {}
            ai = m.get("blended_probabilities") if m.get("prob_source") in AI_SOURCES else None
            mk_k, mk_p = top_pick(mk, opts)
            ai_k, ai_p = top_pick(ai, opts) if ai else (None, 0.0)
            d = {k: (prob_of(ai, k) - prob_of(mk, k)) * 100 for k in opts} if ai else {}
            gap = max((abs(v) for v in d.values()), default=0.0)
            ev = {k: prob_of(ai or mk, k) * float(odds[k]) - 1.0 for k in opts}
            return {"m": m, "opts": opts, "labels": pick_labels(m), "mk": mk, "ai": ai,
                    "mk_k": mk_k, "mk_p": mk_p, "ai_k": ai_k, "ai_p": ai_p, "d": d, "gap": gap,
                    "up_k": max(d, key=d.get) if d else None,
                    "differs": bool(ai) and (ai_k != mk_k or gap >= DIFF_PP),
                    "ev": ev, "gems": {k: v for k, v in ev.items() if GEM_ON and ai and v > 0},
                    "best_ev": max(ev.values()) if ev else -1.0}

        rows_all = [a for a in (_analyze(m) for m in predictions) if a["opts"]]

        # C. 필터: ① 탭과 같은 배지
        T2_QUICK = {"ai": "🧠 AI 적용만", "diff": "⚠️ 배당과 다름", "gem": "💎 플러스", "single": "⚡ 한경기만", "open": "⏳ 마감 전만"}
        if not GEM_ON:   # [6.5] 추천 잠금 중에는 💎 필터 없음
            T2_QUICK.pop("gem")
        T2_SORTS = ["AI 차이 큰 순", "경기 시간순", "기대가치 높은 순"]
        T2_DEFAULTS = {"tab2_search": "", "tab2_sports": [], "tab2_types": [], "tab2_league": "전체 리그",
                       "tab2_quick": [], "tab2_sort_by": T2_SORTS[0]}

        def _t2_reset():
            for k, v in T2_DEFAULTS.items():
                st.session_state[k] = v

        with st.container(border=True):
            b1, b2 = st.columns([1.5, 2.6])
            with b1:
                sp_opts = [s for s in ("SC", "BK", "BS", "VL") if any(a["m"].get("sport") == s for a in rows_all)]
                keep_valid_multi("tab2_sports", sp_opts)
                t_sports = st.pills("종목", sp_opts, selection_mode="multi", key="tab2_sports",
                                    format_func=lambda s: f"{SPORT_ICONS.get(s, '')} {SPORT_NAMES.get(s, s)}") or []
            in_sp = [a for a in rows_all if not t_sports or a["m"].get("sport") in t_sports]
            with b2:
                ty_opts = [t for t in TYPE_BADGES if any(type_badge(a["m"]) == t for a in in_sp)]
                keep_valid_multi("tab2_types", ty_opts)
                t_types = st.pills("게임 유형", ty_opts, selection_mode="multi", key="tab2_types") or []
            f1, f2, f3 = st.columns([1.5, 1.6, 1.0])
            with f1:
                t_search = st.text_input("🔎 팀 이름 검색", placeholder="예: 아스널, 뮌헨, 인천", key="tab2_search")
            with f2:
                lg_opts = ["전체 리그"] + sorted({a["m"].get("league", "기타") for a in in_sp})
                keep_valid("tab2_league", lg_opts)
                t_league = st.selectbox("리그", lg_opts, key="tab2_league")
            with f3:
                t_sort = st.selectbox("정렬", T2_SORTS, key="tab2_sort_by")
            q1, q2 = st.columns([5, 1], vertical_alignment="center")
            with q1:
                keep_valid_multi("tab2_quick", list(T2_QUICK))
                t_quick = st.pills("빠른 필터", list(T2_QUICK), format_func=T2_QUICK.get, selection_mode="multi",
                                   key="tab2_quick", label_visibility="collapsed") or []
            with q2:
                st.button("↺ 필터 초기화", key="tab2_reset", on_click=_t2_reset, width="stretch")

        t_rows = []
        for a in in_sp:
            m = a["m"]
            if t_types and type_badge(m) not in t_types:
                continue
            if t_league != "전체 리그" and m.get("league") != t_league:
                continue
            if t_search and t_search.lower() not in (m.get("home_team", "") + " " + m.get("away_team", "")).lower():
                continue
            if ("ai" in t_quick and not a["ai"]) or ("diff" in t_quick and not a["differs"]) \
                    or ("gem" in t_quick and not a["gems"]) or ("single" in t_quick and not m.get("is_single")) \
                    or ("open" in t_quick and is_sale_closed(m, t2_now)):
                continue
            t_rows.append(a)

        def _seq_n(a):
            s = str(a["m"].get("match_seq", ""))
            return int(s) if s.isdigit() else 0

        if t_sort == "AI 차이 큰 순":   # AI 적용 경기 중 차이 큰 것 먼저, 나머지는 시간순
            t_rows.sort(key=lambda a: (0 if a["ai"] else 1, -a["gap"], time_key(a["m"]), _seq_n(a)))
        elif t_sort == "기대가치 높은 순":
            t_rows.sort(key=lambda a: -a["best_ev"])
        else:
            t_rows.sort(key=lambda a: (time_key(a["m"]), _seq_n(a)))

        # D. 요약 칸
        render_tiles([
            ("📋 경기", f"{len(t_rows):,}경기", f"이번 회차 전체 {len(rows_all):,}경기"),
            ("🧠 AI 적용", f"{sum(1 for a in t_rows if a['ai']):,}경기", "나머지는 배당 기준"),
            ("⚠️ AI가 다르게 본 경기", f"{sum(1 for a in t_rows if a['differs']):,}경기",
             f"1순위가 다르거나 {DIFF_PP:.0f}%p 이상 차이"),
            ("💎 기대가치 플러스", f"{sum(1 for a in t_rows if a['gems']):,}경기", "AI 확률 기준 · 검증 통과") if GEM_ON else
            ("🔒 💎 추천", "잠금", LOCK_NOTE),
        ])

        # E. 경기 목록 (줄을 누르면 아래에 자세히)
        def _ai_top_text(a):
            if not a["ai"] or not a["ai_k"]:
                return "-"
            lab = a["labels"]
            if a["ai_k"] != a["mk_k"]:
                return f"⚠️ {lab[a['ai_k']]} {a['ai_p'] * 100:.0f}% (1순위 다름)"
            dd = a["d"][a["ai_k"]]
            arrow = f" ▲{dd:.0f}" if dd >= 0.5 else (f" ▼{-dd:.0f}" if dd <= -0.5 else "")
            return f"{lab[a['ai_k']]} {a['ai_p'] * 100:.0f}%{arrow}"

        def _ai_up_text(a):
            """AI가 배당보다 가장 높게 본 선택지 (0.5%p 미만이면 '거의 같음')"""
            if not a["ai"] or not a["up_k"]:
                return "-"
            k = a["up_k"]
            if a["d"][k] < 0.5:
                return "거의 같음"
            return f"{'💎 ' if k in a['gems'] else ''}{a['labels'][k]} +{a['d'][k]:.1f}%p"

        def _type_text(m):
            line = ""
            try:
                lv = float(m.get("line") or 0.0)
            except (TypeError, ValueError):
                lv = 0.0
            if lv and "핸디캡" in str(m.get("game_type", "")):
                line = f" H {lv:+.1f}"
            elif lv and "언더오버" in str(m.get("game_type", "")):
                line = f" U/O {lv:g}"
            return f"{type_name(m)}{line}{' ⚡' if m.get('is_single') else ''}"

        st.markdown("#### 📋 경기 목록")
        if not t_rows:
            st.info("조건에 맞는 경기가 없어요. 필터를 초기화해 보세요.")
        else:
            df_t2 = pd.DataFrame([{
                "번호": str(a["m"].get("match_seq", "")),
                "시간": short_when(a["m"]),
                "경기": f"{a['m'].get('home_team', '')} vs {a['m'].get('away_team', '')}",
                "유형": _type_text(a["m"]),
                "배당 1순위": f"{a['labels'][a['mk_k']]} {a['mk_p'] * 100:.0f}%" if a["mk_k"] else "-",
                "AI 1순위": _ai_top_text(a),
                "AI가 더 높게 본 쪽": _ai_up_text(a),
            } for a in t_rows])
            t2_sig = (tuple(t_sports), tuple(t_types), t_league, t_search, tuple(t_quick), t_sort)
            event = st.dataframe(
                df_t2, hide_index=True, width="stretch", height=min(38 + 35 * len(df_t2), 422),
                on_select="rerun", selection_mode="single-row", key=f"tab2_table_{t2_sig}",
                column_config={"번호": st.column_config.TextColumn(width="small"),
                               "시간": st.column_config.TextColumn(width="small"),
                               "경기": st.column_config.TextColumn(width="large"),
                               "AI가 더 높게 본 쪽": st.column_config.TextColumn(
                                   help="AI 확률 − 배당 확률이 가장 큰 선택지 (💎 = AI 기준 기대가치 플러스)")})
            picked_rows = list(event.selection.rows) if event is not None else []
            st.caption("👆 줄을 누르면 아래 분석이 그 경기로 바뀌어요." if not picked_rows
                       else "다른 줄을 누르면 그 경기로 바뀌어요.")

            # F. 경기 분석 (선택한 줄, 없으면 목록 첫 경기)
            a = t_rows[picked_rows[0] if picked_rows and picked_rows[0] < len(t_rows) else 0]
            m = a["m"]
            seq, lab, odds = str(m.get("match_seq", "")), a["labels"], m.get("betman_odds", {})
            closed = is_sale_closed(m, t2_now)

            def _t2_toggle(seq_, k_):
                """선택지 하나를 담거나 빼기 (같은 경기의 다른 선택지는 그대로)"""
                cur = dict(st.session_state["selections"].get(seq_, {}))
                cur[SEL_KEY[k_]] = not cur.get(SEL_KEY[k_], False)
                if any(cur.get(x) for x in ("home", "draw", "away")):
                    st.session_state["selections"][seq_] = cur
                else:
                    st.session_state["selections"].pop(seq_, None)
                st.session_state["editor_version"] += 1   # ① 탭 배당 버튼도 같은 상태로 다시 그림

            def _bar(name, p, color, note=""):
                return (f"<div style='display:flex;align-items:center;gap:8px;font-size:12px;margin:2px 0'>"
                        f"<span style='width:30px;opacity:.7'>{name}</span>"
                        f"<div style='flex:1;height:8px;border-radius:4px;background:rgba(148,163,184,.18)'>"
                        f"<div style='width:{max(0.0, min(100.0, p)):.1f}%;height:100%;border-radius:4px;background:{color}'></div></div>"
                        f"<span style='width:78px;text-align:right;font-size:13px'>{p:.1f}%{note}</span></div>")

            st.markdown("#### 🔎 경기 분석")
            with st.container(border=True):
                status_txt = ("🔒 발매 마감" if closed else
                              f"🟢 {datetime.fromtimestamp(float(m['close_time']) / 1000).strftime('%m.%d %H:%M')} 마감"
                              if m.get("close_time") else "")
                st.markdown(
                    "<div style='padding-bottom:12px'>"
                    "<div style='display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;padding-bottom:6px;"
                    "border-bottom:1px solid rgba(148,163,184,.25)'>"
                    f"<span style='font-size:12px;opacity:.75'>{SPORT_ICONS.get(m.get('sport'), '')} {short_when(m)}</span>"
                    + league_chip(m.get("league", "")) +
                    f"<span style='font-size:17px'><b>{m.get('home_team', '')}</b> <span style='opacity:.6'>vs</span> "
                    f"<b>{m.get('away_team', '')}</b></span>"
                    f"<span style='font-size:14px'><b>{seq}</b> {type_name(m)} {line_chip(m)} "
                    f"{ONE_GAME_TAG if m.get('is_single') else ''}</span>"
                    f"<span style='margin-left:auto;font-size:12px;opacity:.85'>{status_txt}</span></div></div>",
                    unsafe_allow_html=True)

                # 💬 쉬운 말 해석
                margin = (sum(1.0 / float(odds[k]) for k in a["opts"]) - 1.0) * 100
                says = []
                if not a["ai"]:
                    says.append("이 유형은 AI가 적용되지 않아 배당 기준 확률만 보여요 "
                                "(과거 시험에서 배당보다 낫다는 근거가 없었던 유형이에요).")
                elif a["ai_k"] != a["mk_k"]:
                    says.append(f"AI와 배당의 1순위가 달라요: 배당은 <b>{lab[a['mk_k']]}</b> {a['mk_p'] * 100:.0f}%, "
                                f"AI는 <b>{lab[a['ai_k']]}</b> {a['ai_p'] * 100:.0f}%예요.")
                elif a["gap"] < 1.0:
                    says.append(f"AI는 배당과 거의 같게 봐요 (가장 큰 차이 {a['gap']:.1f}%p).")
                else:
                    k = a["up_k"]
                    says.append(f"AI는 배당보다 <b>{lab[k]}</b> 쪽을 {a['d'][k]:.1f}%p 높게 봐요 "
                                f"(AI {prob_of(a['ai'], k) * 100:.1f}% vs 배당 {prob_of(a['mk'], k) * 100:.1f}%).")
                bk = max(a["ev"], key=a["ev"].get)
                bev = a["ev"][bk] * 100
                if bev > 0 and a["ai"] and not GEM_ON:
                    says.append(f"AI 계산으로는 <b>{lab[bk]}</b> 기대가치 {bev:+.1f}%지만, 검증 3단계를 통과하기 전이라 "
                                "추천하지 않아요 (🔒 추천 잠금).")
                elif bev > 0 and a["ai"]:
                    says.append(f"💎 <b>{lab[bk]}</b> 기대가치 {bev:+.1f}%: AI 기준 플러스예요 (검증 3단계 통과).")
                else:
                    says.append(f"가장 나은 선택지(<b>{lab[bk]}</b>)도 기대가치 {bev:+.1f}%예요. 베트맨 마진(약 {margin:.0f}%)이 "
                                + ("AI가 보는 차이보다 커서" if a["ai"] else "있어서") + " 오래 걸면 손해예요.")
                st.markdown("<div style='padding-bottom:12px'><div style='font-size:14px;line-height:1.7'>"
                            + "<br>".join(f"💬 {s}" if i == 0 else s for i, s in enumerate(says)) + "</div></div>",
                            unsafe_allow_html=True)

                # 선택지별: 배당 확률 vs AI 확률 막대 · 기대가치 · 담기
                cart_sel = st.session_state["selections"].get(seq, {})
                for k in a["opts"]:
                    c1, c2, c3, c4 = st.columns([1.1, 3.6, 0.9, 1.1], vertical_alignment="center")
                    c1.markdown(f"<div style='font-size:15px'><b>{lab[k]}</b> <span style='opacity:.8'>{float(odds[k]):.2f}</span>"
                                f"{' 💎' if k in a['gems'] else ''}</div>", unsafe_allow_html=True)
                    bars = _bar("배당", prob_of(a["mk"], k) * 100, "#94a3b8")
                    if a["ai"]:
                        dd = a["d"][k]
                        bars += _bar("AI", prob_of(a["ai"], k) * 100, "#3b82f6",
                                     f" <span style='font-size:11px;opacity:.7'>{dd:+.1f}</span>")
                    c2.markdown(bars, unsafe_allow_html=True)
                    ev_k = a["ev"][k] * 100
                    c3.markdown(f"<div style='font-size:11px;color:#94a3b8'>기대가치</div>"
                                f"<div style='font-size:14px;font-weight:700;color:{'#34d399' if ev_k > 0 else 'inherit'}'>"
                                f"{ev_k:+.1f}%</div>", unsafe_allow_html=True)
                    in_cart = bool(cart_sel.get(SEL_KEY[k]))
                    # help(말풍선)는 넣지 않음: 누르면 글자가 바뀌며 다시 그려져 말풍선이 화면에 남는다
                    c4.button("🔒 마감" if closed else ("✓ 담김" if in_cart else "🛒 담기"), key=f"t2_cart_{seq}_{k}",
                              type="primary" if in_cart and not closed else "secondary", disabled=closed,
                              width="stretch", on_click=_t2_toggle, args=(seq, k))
                if not closed:
                    st.caption("🛒 담기를 누르면 내 선택에 담기고, ✓ 담김을 다시 누르면 빠져요.")

                if m.get("top_scores"):
                    st.caption("🎯 예상 스코어: " + " · ".join(f"{s['score']} ({s['prob']}%)" for s in m["top_scores"]))

                # 최근 경기: 팀마다 카드 (전적 · 승무패 칸 · 경기별 한 줄) / 맞대결: 따로 한 카드 (홈팀 기준 결과)
                hf, af, h2h = m.get("home_form") or {}, m.get("away_form") or {}, m.get("h2h") or {}
                if hf.get("has_data") or af.get("has_data") or h2h.get("has_data"):
                    RES_STYLE = {"W": ("승", "rgba(34,197,94,.2)", "#4ade80"), "D": ("무", "rgba(234,179,8,.2)", "#facc15"),
                                 "L": ("패", "rgba(239,68,68,.2)", "#f87171")}

                    def _pill(r):
                        txt, bg, fg = RES_STYLE.get(r, ("-", "rgba(148,163,184,.2)", "inherit"))
                        return (f"<span style='display:inline-block;width:22px;text-align:center;border-radius:4px;font-size:11px;"
                                f"font-weight:700;background:{bg};color:{fg}'>{txt}</span>")

                    def _ko(name):
                        """예전에 저장된 기록의 영어 팀명(유럽 기록)도 한글로"""
                        name = str(name or "")
                        return name if re.search(r"[가-힣]", name) else KOREAN_NAME.get(name, name)

                    def _short_d(rec, with_year=False):
                        """새 기록은 date_short, 예전 기록은 '26.10.19 (월)' / '18/10/2026' 에서 날짜만"""
                        if rec.get("date_short"):
                            return rec["date_short"]
                        d = str(rec.get("date", ""))
                        mt = re.match(r"(\d{2})\.(\d{2})\.(\d{2})", d)
                        if mt:
                            return (f"{mt.group(1)}." if with_year else "") + f"{mt.group(2)}.{mt.group(3)}"
                        mt = re.match(r"(\d{2})/(\d{2})/\d{2}(\d{2})", d)
                        if mt:
                            return (f"{mt.group(3)}." if with_year else "") + f"{mt.group(2)}.{mt.group(1)}"
                        return d[:10]

                    row_css = ("display:grid;align-items:center;gap:8px;padding:4px 0;font-size:13px;"
                               "border-top:1px solid rgba(148,163,184,.12)")

                    def _standing_line(sd):
                        """현재 리그 순위 · 승점 · 시즌 경기당 득실 (유럽 5대 리그 1부 팀만 있음)"""
                        if not sd:
                            return ""
                        return (f"<div style='font-size:13px;margin:4px 0 2px;display:flex;gap:6px;align-items:baseline;flex-wrap:wrap'>"
                                f"{league_chip(sd['league'])}<b style='font-size:15px'>{sd['pos']}위</b>"
                                f"<span style='opacity:.7'>/ {sd['teams']}팀 · 승점 {sd['pts']} ({sd['played']}경기 {sd['w']}승 "
                                f"{sd['d']}무 {sd['l']}패) · 시즌 경기당 득 <b>{sd['gf_pg']:.1f}</b> · 실 <b>{sd['ga_pg']:.1f}</b> "
                                f"<span style='font-size:11px'>({sd['data_until']}까지)</span></span></div>")

                    def _team_card(icon, team, f):
                        head = (f"<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px'>"
                                f"<span style='font-size:15px'>{icon} <b>{team}</b></span>"
                                + (f"<span style='font-size:14px'><span style='font-size:11px;opacity:.65'>최근 5경기</span> "
                                   f"<b>{f.get('record_str', '')}</b></span>" if f.get("has_data") else "") + "</div>"
                                + _standing_line(f.get("standing")))
                        if not f.get("has_data"):
                            body = "<div style='opacity:.6;font-size:13px;padding-top:4px'>최근 경기 기록 없음</div>"
                        else:
                            body = (f"<div style='margin:4px 0 6px;display:flex;gap:3px;align-items:center;flex-wrap:wrap'>"
                                    + "".join(_pill(r) for r in f.get("form_sequence", []))
                                    + f"<span style='font-size:12px;opacity:.75;margin-left:6px'>최근 경기당 득 {f.get('avg_scored', 0)} · "
                                      f"실 {f.get('avg_conceded', 0)}</span></div>"
                                    + "".join(
                                        f"<div style='{row_css};grid-template-columns:24px 40px 30px 1fr auto'>"
                                        f"{_pill(lm.get('result'))}<span style='opacity:.7'>{_short_d(lm)}</span>"
                                        f"<span style='opacity:.7'>{lm.get('venue', '')}</span>"
                                        f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{_ko(lm.get('opponent'))}</span>"
                                        f"<b>{lm.get('score', '')}</b></div>"
                                        for lm in f.get("last_matches", [])))
                        return ("<div style='border:1px solid rgba(148,163,184,.25);border-radius:8px;padding:10px 12px;"
                                f"margin-bottom:14px'>{head}{body}</div>")

                    st.markdown("<div style='font-size:14px;font-weight:700;margin-top:6px'>📈 순위 · 최근 5경기 "
                                "<span style='font-size:12px;font-weight:400;opacity:.7'>순위는 유럽 5대 리그 1부만 · 위쪽이 가장 최근 · "
                                "이 경기 전 기록 · 점수는 그 팀 점수가 먼저</span></div>", unsafe_allow_html=True)
                    fc1, fc2 = st.columns(2)
                    fc1.markdown(_team_card("🏠", m.get("home_team", ""), hf), unsafe_allow_html=True)
                    fc2.markdown(_team_card("✈️", m.get("away_team", ""), af), unsafe_allow_html=True)

                    if h2h.get("has_data"):
                        home_t = m.get("home_team", "")

                        def _h2h_res(hm):
                            if hm.get("result"):
                                return hm["result"]
                            w = hm.get("winner")
                            return "D" if w == "무승부" else ("W" if _ko(w) == home_t else "L")

                        def _side(name, pos=None):
                            nm = _ko(name)
                            pos_txt = f" <span style='font-size:11px;opacity:.65'>{pos}위</span>" if pos else ""
                            return (f"<b>{nm}</b>" if nm == home_t else nm) + pos_txt

                        h_rows = "".join(
                            f"<div style='{row_css};grid-template-columns:24px 66px 1fr'>"
                            f"{_pill(_h2h_res(hm))}<span style='opacity:.7'>{_short_d(hm, with_year=True)}</span>"
                            f"<span>{_side(hm.get('home'), hm.get('home_pos'))} <b style='margin:0 6px'>{hm.get('score', '')}</b> "
                            f"{_side(hm.get('away'), hm.get('away_pos'))}</span></div>"
                            for hm in h2h.get("matches", []))
                        has_pos = any(hm.get("home_pos") or hm.get("away_pos") for hm in h2h.get("matches", []))
                        st.markdown(
                            "<div style='border:1px solid rgba(148,163,184,.25);border-radius:8px;padding:10px 12px;margin-bottom:14px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap'>"
                            f"<span style='font-size:15px'>🤝 <b>맞대결</b> <span style='font-size:12px;opacity:.7'>최근 "
                            f"{h2h.get('total_matches', 0)}경기 · 경기 순서대로 홈팀 먼저"
                            f"{' · 이름 옆 = 그 경기 당시 순위' if has_pos else ''}</span></span>"
                            f"<b style='font-size:14px'>{home_t} 기준 {h2h.get('home_wins', 0)}승 {h2h.get('draws', 0)}무 "
                            f"{h2h.get('away_wins', 0)}패</b></div>{h_rows}</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 3: 📊 ③ 성과 검증 (실제 경기 대조 & 통계)
# ==============================================================================
with tab3:
    card = load_scorecard()
    clv = load_clv() or {}
    pt = load_paper_trading()
    bf = load_paper_backfill()
    SCORE_MIN, CLV_MIN, PAPER_MIN = 300, 50, 300   # 판정에 필요한 표본 (scorecard · clv · paper_trading 과 같은 기준)
    scan = load_market_scan()

    # [6.6] 🧭 지금 결론 — 그래서 어떻게 판단하나 (쉬운 말)
    render_conclusion(card, scan, load_stale_odds())
    render_glossary()

    # ------------------------------------------------------------------
    # 0. 한눈에 보기 — 세 가지 질문에 쉬운 말로 답하고, 판정까지 얼마나 남았는지 막대로
    # ------------------------------------------------------------------
    CHIP = {"wait": ("rgba(234,179,8,.18)", "#facc15"), "good": ("rgba(34,197,94,.18)", "#4ade80"),
            "bad": ("rgba(239,68,68,.18)", "#f87171"), "neutral": ("rgba(148,163,184,.2)", "#cbd5e1")}

    def _summary_card(question, kind, chip_txt, sentence, n, target, unit):
        bg, fg = CHIP[kind]
        pct = max(0.0, min(1.0, n / target)) if target else 1.0
        return (
            "<div style='flex:1 1 280px;border:1px solid rgba(148,163,184,.3);border-radius:10px;padding:12px 14px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"
            f"<b style='font-size:15px'>{question}</b>"
            f"<span style='white-space:nowrap;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;"
            f"background:{bg};color:{fg}'>{chip_txt}</span></div>"
            f"<div style='font-size:14px;line-height:1.6;margin:8px 0'>{sentence}</div>"
            f"<div style='height:6px;border-radius:3px;background:rgba(148,163,184,.2)'>"
            f"<div style='width:{pct * 100:.0f}%;height:100%;border-radius:3px;background:{fg}'></div></div>"
            f"<div style='font-size:12px;opacity:.7;margin-top:4px'>판정까지: {unit} {n:,} / {target:,}</div></div>")

    summary = []
    # (1) AI가 배당보다 정확한가? — 공식 성적표
    n_g = card.get("graded_matches", 0) if card else 0
    if card and card.get("model_ll") is not None:
        diff, ci_s = card["ll_diff"], card.get("ll_diff_ci") or [0, 0]
        if n_g < SCORE_MIN:
            kind, chip_txt = "wait", "검증 중"
        elif ci_s[1] < 0:
            kind, chip_txt = "good", "AI가 더 정확"
        elif ci_s[0] > 0:
            kind, chip_txt = "bad", "배당이 더 정확"
        else:
            kind, chip_txt = "neutral", "차이 확실치 않음"
        lean = "AI가 배당보다 조금 더 정확" if diff < 0 else ("배당이 AI보다 조금 더 정확" if diff > 0 else "AI와 배당이 같음")
        sent = f"지금까지 {n_g:,}경기에서 <b>{lean}</b>해요." + (" 아직은 운일 수 있어요." if n_g < SCORE_MIN else "")
    else:
        kind, chip_txt, sent = "wait", "검증 중", "아직 채점한 경기가 없어요. 경기 전 AI 기록이 끝난 경기와 만나면 자동으로 채점돼요."
    summary.append(_summary_card("🧠 AI가 배당보다 정확한가?", kind, chip_txt, sent, n_g, SCORE_MIN, "채점한 경기"))

    # (2) 다른 배터들이 나중에 AI 쪽으로 움직였나? — 마감 배당 비교
    moved = int(clv.get("moved") or 0)
    if clv.get("matches"):
        v_txt = str(clv.get("verdict", ""))
        if moved < CLV_MIN:
            kind, chip_txt = "wait", "검증 중"
        elif "좋은" in v_txt:
            kind, chip_txt = "good", "AI 쪽으로 움직임"
        elif "나쁜" in v_txt:
            kind, chip_txt = "bad", "AI 반대쪽으로"
        else:
            kind, chip_txt = "neutral", "뚜렷한 방향 없음"
        sent = (f"배당이 바뀐 {moved}경기 중 <b>{clv.get('toward_ai', 0)}경기가 AI 쪽</b>으로 움직였어요"
                + (f" ({clv['toward_ai_pct']:.0f}% · 비교 기준 {clv['fav_toward_pct']:.0f}%)."
                   if clv.get("toward_ai_pct") is not None and clv.get("fav_toward_pct") is not None else "."))
    else:
        kind, chip_txt, sent = "wait", "검증 중", "아직 비교할 경기가 없어요. AI 기록이 있는 경기가 마감되면 쌓여요."
    summary.append(_summary_card("📈 시장이 나중에 AI 쪽으로 움직였나?", kind, chip_txt, sent, moved, CLV_MIN, "배당이 바뀐 경기"))

    # (3) 돈을 버는 전략이 있나? — 가상 베팅 리그
    live = (pt or {}).get("strategies", {})
    past = (bf or {}).get("strategies", {})
    live_done = max((v["bets"] for v in live.values()), default=0)
    live_wait = sum(v["pending"] for v in live.values())
    proven = [v["label"] for v in list(live.values()) + list(past.values()) if v.get("verdict") == "수익 입증"]
    if proven:
        kind, chip_txt = "good", "수익 입증"
    elif live_done < PAPER_MIN:
        kind, chip_txt = "wait", "검증 중"
    else:
        kind, chip_txt = "bad", "수익 미입증"
    past_txt = " · ".join(f"{v['label'].split(' (')[0]} <b>{v['roi_pct']:+.0f}%</b>" for v in past.values() if v["bets"])
    sent = (f"실시간 가상 베팅: 끝난 {sum(v['bets'] for v in live.values()):,}건 · 결과 대기 {live_wait:,}건."
            + (f"<br>과거 재현: {past_txt}" + (" → 배당만 보는 전략은 모두 손실이에요." if all(v["roi_pct"] < 0 for v in past.values() if v["bets"]) else "")
               if past_txt else ""))
    summary.append(_summary_card("🧪 돈을 버는 전략이 있나?", kind, chip_txt, sent, live_done, PAPER_MIN, "전략별 끝난 베팅(최다)"))

    st.markdown("<div style='padding-bottom:12px'><div style='font-size:20px;font-weight:700;margin-bottom:8px'>📋 한눈에 보기</div>"
                "<div style='display:flex;flex-wrap:wrap;gap:10px'>" + "".join(summary) + "</div></div>", unsafe_allow_html=True)
    # ------------------------------------------------------------------
    # 🔒 추천 잠금 (섀도 모드) — 검증 3단계 · 사전 등록 가설 (Phase 6.5, 6.6 쉬운 말)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🔒 AI 시험 진행 상황")
    st.caption("AI 추천(💎 · 추천 픽 · 조합 찾기)은 아래 시험 3개를 **모두** 통과해야 켜져요. 그동안에도 AI는 뒤에서 계속 예측하고 "
               "채점받아요(섀도 모드). 시험 기준은 2026-10-27에 미리 정했고, 결과를 보고 바꾸지 않아요.")
    for s in REC_STATUS["stages"]:
        st.markdown(f"- {'✅' if s['ok'] else '⏳'} **{s['name']}** — {s['plain']}")
    for f in REC_STATUS["features"].values():
        p = f["profit"]
        st.markdown(f"- {'🔓 켜짐' if f['on'] else '🔒 잠김'} **{f['short']}** · {p['name']} — {p['plain']}")
    with st.expander("숫자로 보기 (자세히)"):
        st.markdown("\n".join([f"- {s['name']}: {s['detail']}" for s in REC_STATUS["stages"]]
                              + [f"- {f['label']} · {f['profit']['name']}: {f['profit']['detail']}"
                                 for f in REC_STATUS["features"].values()]))
        st.caption("① 과거 시험: 워크포워드 합친 결과의 95% 범위 상한 < 0 · ② 실전 확률 시험: 경기 전 고정 예측 300경기 이상, "
                   "95% 범위 상한 < 0 · ③ 실제 돈 시험: 300건 이상, 수익률 95% 범위 하한 > 0. 범위는 모두 회차 단위예요. "
                   "300건은 최소치라, 우위가 작으면(+2~3%) 확인까지 수천 건이 필요해요.")
    hyps = (card or {}).get("hypotheses") or []
    if hyps:
        st.markdown("**📌 미리 적어 둔 예상 (사전 가설)** — 지난 경기 시험에서 AI가 특히 잘 맞힌 두 그룹이 앞으로도 그런지 봐요. "
                    "2026-10-28 이후 경기만 세고, 300경기부터 판정, 600경기에서도 확인 안 되면 접어요.")
        st.dataframe(pd.DataFrame([{
            "예상": f"{h['id']} {h['name']}", "모인 경기": f"{h['n']} / 300",
            "AI − 배당 (음수 = AI가 더 잘 맞힘)": f"{h['ll_diff']:+.4f}" if h.get("ll_diff") is not None else "-",
            "판정": h["verdict"]} for h in hyps]), hide_index=True, width="stretch")

    # [6.6] 💰 베트맨이 떼어가는 몫 — 유형별 · 배당대별 (덜 손해 보는 쪽 판단용)
    st.divider()
    render_margin_tables(scan)

    # ------------------------------------------------------------------
    # A. 공식 AI 성적표 — 경기 시작 전에 고정 저장한 예측만 실제 결과로 채점 (사후 정보 없음)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🏅 공식 AI 성적표")
    st.caption("경기 시작 전에 저장해 둔 AI 확률(data/history/prediction_log)만 실제 결과로 채점해요. "
               "학습에 쓰지 않은 진짜 실전 성적이라 AI 실력의 공식 지표예요.")
    if card and card.get("model_ll") is not None:
        ci = card.get("ll_diff_ci") or [0, 0]
        d_color = "#34d399" if card["ll_diff"] < 0 else ("#f87171" if card["ll_diff"] > 0 else "inherit")
        render_tiles([
            ("📋 채점한 경기", f"{n_g:,}경기",
             (f"판정까지 {SCORE_MIN - n_g:,}경기 남음 · " if n_g < SCORE_MIN else "") + f"회차 {len(card.get('rounds', []))}개"),
            ("🎯 정확도 점수 (낮을수록 좋음)", f"AI {card['model_ll']:.3f}", f"배당 {card['market_ll']:.3f}"),
            ("⚖️ AI − 배당", f"<span style='color:{d_color}'>{card['ll_diff']:+.4f}</span>",
             f"음수면 AI가 더 정확 · 95% 범위 [{ci[0]:+.3f}, {ci[1]:+.3f}]"),
            ("✅ AI 1순위 적중률", f"{card.get('hit_rate_top_pick', 0):.1f}%", "참고용 (배당 1순위도 비슷하게 맞아요)"),
        ])
        if n_g < SCORE_MIN:
            says = (f"💬 지금까지는 AI가 배당보다 {'조금 더' if card['ll_diff'] < 0 else '덜'} 정확해요. "
                    f"하지만 {n_g:,}경기는 너무 적어서 운일 수 있어요. {SCORE_MIN}경기가 모이면 판정해요.")
        else:
            says = f"💬 판정: {card.get('verdict', '-')}"
        gb = card.get("gem_bets")
        if gb:
            says += (f"<br>💎 AI 기준 기대가치 플러스 선택지에 1,000원씩 걸었다면: {gb:,}건 · 적중 {card.get('gem_hits', 0):,}건 · "
                     f"수익률 <b>{card.get('gem_roi_pct', 0):+.1f}%</b> (95% 범위 [{card.get('gem_roi_ci_low', 0):+.1f}%, "
                     f"{card.get('gem_roi_ci_high', 0):+.1f}%])" + (" — 아직 판단하기엔 적어요." if gb < SCORE_MIN else ""))
        elif gb is not None:
            says += "<br>💎 AI 추천(기대가치 플러스) 선택지는 아직 0건이에요. AI가 고치는 폭이 베트맨 마진보다 작아서 드물게 나와요."
        st.markdown(f"<div style='padding-bottom:10px'><div style='font-size:14px;line-height:1.7'>{says}</div></div>",
                    unsafe_allow_html=True)
        hist = [h for h in card.get("history", []) if h.get("ll_diff") is not None]
        if len(hist) >= 2:
            h_df = pd.DataFrame(hist).set_index("date")[["model_ll", "market_ll"]].rename(
                columns={"model_ll": "AI", "market_ll": "배당"})
            st.line_chart(h_df, height=200)
            st.caption("날짜별 누적 정확도 점수 (낮을수록 좋음) — AI 선이 배당 선보다 아래에 있으면 AI가 더 정확해요.")
        render_calibration(card)   # [6.6] 보정 곡선 — AI가 말한 확률이 실제로 맞았나
        if len(card.get("by_game_type", {})) > 1:
            with st.expander("유형별 성적 (어느 유형에서 AI가 앞서는지)"):
                st.dataframe(pd.DataFrame([
                    {"유형": gt, "채점한 경기": v["n"], "AI − 배당 (음수 = AI 우세)": f"{v['ll_diff']:+.4f}",
                     "95% 범위": f"[{v['ll_diff_ci'][0]:+.4f}, {v['ll_diff_ci'][1]:+.4f}]"}
                    for gt, v in sorted(card["by_game_type"].items(), key=lambda x: -x[1]["n"])]),
                    hide_index=True, width="stretch")
    else:
        st.info(f"⏳ 아직 채점할 경기가 없어요 (현재 {n_g}경기). 2026-10-25부터 경기 전 예측을 기록하기 시작했고, "
                f"기록된 경기가 끝날 때마다 자동으로 채점돼요. {SCORE_MIN}경기가 쌓이면 판정해요.")

    status = load_model_status()
    mm = load_markets_model()
    if status or (mm and mm.get("segments")):
        with st.expander("📚 과거 시험 성적 — 실전에 쓰기 전에, 학습에 안 쓴 기간으로 미리 본 시험"):
            if status:
                st.markdown(f"**{status['name']}** (축구 승무패 · 실전 확률과 같은 모델)")
                rows_bt = []
                for label, s in status["evals"]:
                    ci_b = s.get("ll_diff_ci", [None, None])
                    rows_bt.append({"시험": label, "경기 수": s["samples"], "AI 점수": s["model_ll"], "배당 점수": s["market_ll"],
                                    "AI − 배당": f"{s['ll_diff']:+.4f}", "95% 범위": f"[{ci_b[0]:+.4f}, {ci_b[1]:+.4f}]"})
                st.dataframe(pd.DataFrame(rows_bt), width="stretch", hide_index=True)
            if mm and mm.get("segments"):
                st.markdown("**전 종목 보정 AI** — 유형별 시험 (핸디캡·언더오버·홀짝·농구·야구·배구 …)")
                st.caption("유형마다 '베트맨이 일관되게 잘못 매기는 쪽'을 2023~2025년 상반기로 학습하고, 2026년 시험·최근 실전에서 "
                           "둘 다 배당보다 나은 유형만 AI 확률을 계산·기록해요(섀도). 추천은 위 🔒 검증 3단계를 통과해야 켜져요. "
                           f"학습: {mm.get('trained_at', '-')[:10]}")
                seg_rows = []
                for key, v in mm["segments"].items():
                    gt, k = key.split("|")
                    t, h = v.get("test"), v.get("holdout")
                    seg_rows.append({
                        "유형": gt + (" (무 있음)" if k == "3" and "승무패" not in gt else ""),
                        "AI 계산": "✅ 계산(섀도)" if v.get("passes_gate") else ("⏳ 표본 부족" if v.get("reason") else "❌ 배당 기준"),
                        "2026 시험 경기": t["n"] if t else v["sizes"]["test"],
                        "2026 시험 AI − 배당": f"{t['ll_diff']:+.4f}" if t else "-",
                        "95% 범위": f"[{t['ll_diff_ci'][0]:+.4f}, {t['ll_diff_ci'][1]:+.4f}]" if t else "-",
                        "최근 실전 경기": h["n"] if h else 0,
                        "최근 실전 AI − 배당": f"{h['ll_diff']:+.4f}" if h else "-",
                        "강팀 저평가 정도 (β)": round(v["eval_params"]["beta"], 3) if v.get("eval_params") else None,
                    })
                seg_rows.sort(key=lambda r: (not r["AI 계산"].startswith("✅"), -r["2026 시험 경기"]))
                st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width="stretch")
                st.caption("AI − 배당이 음수면 AI가 더 정확해요. β가 0보다 크면 베트맨이 강한 쪽(배당 1순위)을 낮게 잡는다는 뜻이에요.")

    # ------------------------------------------------------------------
    # B. 마감 배당 비교 (CLV) — 시장이 나중에 AI 쪽으로 움직였는가
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📈 마감 배당 비교")
    st.caption("AI가 '배당보다 높다(▲)'고 본 쪽으로, 마감 때까지 다른 배터들의 돈이 몰렸는지 봐요. 몰렸다면 AI가 시장보다 먼저 맞게 본 거예요. "
               "배당 변화는 1시간마다 기록해요.")
    if clv.get("matches"):
        mv = clv.get("mean_move_pp") or {}
        render_tiles([
            ("📋 비교한 경기", f"{clv['matches']:,}경기", f"그중 배당이 바뀐 경기 {moved}"),
            ("🧭 AI 쪽으로 움직인 비율", f"{clv['toward_ai_pct']:.0f}%" if clv.get("toward_ai_pct") is not None else "-",
             f"AI 쪽 {clv.get('toward_ai', 0)}건 · 반대 {clv.get('away_from_ai', 0)}건"),
            ("📏 비교 기준: 배당 1순위 쪽", f"{clv['fav_toward_pct']:.0f}%" if clv.get("fav_toward_pct") is not None else "-",
             "강한 쪽 배당이 원래 잘 내려가는지 (AI가 이보다 높아야 의미)"),
            ("📊 AI ▲ 쪽 평균 확률 변화", f"{mv['mean']:+.2f}%p" if mv.get("mean") is not None else "-",
             f"95% 범위 [{mv['ci'][0]:+.2f}, {mv['ci'][1]:+.2f}]" if mv.get("ci") else ""),
        ])
        says = (f"💬 배당이 바뀐 {moved}경기 중 {clv.get('toward_ai', 0)}경기가 AI 쪽으로 움직였어요. "
                + (f"{CLV_MIN}경기가 모이면 판정해요." if moved < CLV_MIN else f"판정: {clv.get('verdict', '-')}"))
        g = clv.get("gem_clv_pct") or {}
        if g.get("n"):
            says += f"<br>💎 선택지 {g['n']}건은 처음 배당이 마감 배당보다 평균 {g['mean']:+.2f}% 좋았어요 (플러스 = 마감보다 좋은 배당에 산 셈)."
        st.markdown(f"<div style='padding-bottom:10px'><div style='font-size:14px;line-height:1.7'>{says}</div></div>",
                    unsafe_allow_html=True)
    else:
        st.info("⏳ 아직 비교할 경기가 없어요. AI 예측을 기록한 경기가 마감되면 자동으로 쌓여요.")

    # ------------------------------------------------------------------
    # C. 가상 베팅 리그 — 전략별 자동 가상 베팅 (실제 구매 없음)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🧪 가상 베팅 리그")
    st.caption("전략마다 경기 시작 전에 1,000원씩 가상으로 걸어 두고, 결과가 나오면 자동으로 채점해요 (실제 구매 없음). "
               f"AI 전략은 기준선(E. 배당 1순위)보다 나아야 의미가 있고, {PAPER_MIN}건 이상에서 수익률 95% 범위 아래 끝이 0보다 커야 '수익 입증'이에요.")

    clv_by_strategy = clv.get("paper_strategies", {})
    VERDICT_MARK = {"수익 입증": "✅ 입증", "수익 미입증": "❌ 미입증", "판단 보류 (표본 부족)": "⏳ 표본 부족"}

    def _league_rows(strategies: dict, source: str) -> list:
        rows_ = []
        for key, v in strategies.items():
            cv = clv_by_strategy.get(key, {}) if source == "실시간" else {}
            rows_.append({
                "구분": source, "전략": v["label"], "끝난 베팅": v["bets"], "결과 대기": v["pending"],
                "적중률": f"{v['hits'] / v['bets'] * 100:.0f}%" if v["bets"] else "-",
                "수익률": f"{v['roi_pct']:+.1f}%" if v["bets"] else "-",
                "수익률 95% 범위": f"[{v['roi_ci_low']:+.1f}%, {v['roi_ci_high']:+.1f}%]" if v["bets"] else "-",
                "누적 손익(원)": v["profit"], "판정": VERDICT_MARK.get(v["verdict"], v["verdict"]) if v["bets"] else "-",
                "마감 대비 배당": f"{cv['mean']:+.2f}% ({cv['n']})" if cv.get("n") else "-",
            })
        return rows_

    league_rows = _league_rows(live, "실시간") if live else []
    if past:
        league_rows += _league_rows(past, f"과거 재현 ({bf['rounds'][0]}~{bf['rounds'][1]})")
    if not league_rows:
        st.info("⏳ 아직 가상 베팅 기록이 없어요. 다음 파이프라인 실행부터 쌓여요.")
    else:
        if past_txt:
            st.markdown(
                "<div style='padding-bottom:10px'><div style='font-size:14px;line-height:1.7'>"
                f"💬 과거 회차로 재현해 보면 {past_txt}예요. 배당만 보고 고르는 방식은 베트맨 마진 때문에 오래 하면 손해예요. "
                f"AI 전략(A·B·C)은 실시간으로만 채점하고 있어요 (끝난 베팅 {sum(v['bets'] for v in live.values()):,}건 · "
                f"결과 대기 {live_wait:,}건).</div></div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(league_rows), hide_index=True, width="stretch",
                     column_config={"누적 손익(원)": st.column_config.NumberColumn(format="localized"),
                                    "판정": st.column_config.TextColumn(width="small", help="수익 입증 = 300건 이상 + 수익률 95% 범위 아래 끝 > 0"),
                                    "마감 대비 배당": st.column_config.TextColumn(
                                        help="산 배당 ÷ 마감 배당 − 1 의 경기 평균 (괄호는 경기 수). 플러스면 마감보다 좋은 배당에 산 것")})
        st.caption("실시간: 경기 시작 전에 고정 저장한 베팅 (2026-10-26, 260127회차부터). "
                   "과거 재현: 배당만 쓰는 전략(D·E)을 베트맨 과거 회차의 경기 전 배당과 실제 결과로 재현 — "
                   "AI 전략(A·B·C)은 지금 AI가 그 결과까지 학습했기 때문에 재현하지 않아요.")
        curves = {}
        for data_ in (pt, bf):
            for v in (data_ or {}).get("strategies", {}).values():
                if len(v["cum_profit"]) >= 2:
                    curves[v["label"]] = pd.Series(v["cum_profit"])
        if curves:
            st.line_chart(pd.DataFrame(curves), height=220)
            st.caption("가로: 끝난 베팅 순서 · 세로: 누적 가상 손익(원) — 오른쪽 아래로 내려가면 계속 잃고 있다는 뜻이에요.")

        # 개별 베팅 내역
        with st.expander("📜 가상 베팅 내역 (회차·전략·결과별)", expanded=False):
            det = [{**d, "구분": "실시간", "전략명": STRATEGY_LABELS.get(d["strategy"], d["strategy"])}
                   for d in (pt or {}).get("details", [])]
            det += [{**d, "구분": "과거 재현", "전략명": BACKFILL_LABEL_MAP.get(d["strategy"], d["strategy"])}
                    for d in (bf or {}).get("details", [])]
            if not det:
                st.caption("아직 내역이 없어요.")
            else:
                ddf = pd.DataFrame(det)
                h1, h2, h3, h4 = st.columns(4)
                src_sel = h1.selectbox("구분", ["전체"] + sorted(ddf["구분"].unique()), key="pt_src")
                if src_sel != "전체":
                    ddf = ddf[ddf["구분"] == src_sel]
                rnd_sel = h2.selectbox("회차", ["전체"] + sorted(ddf["gm_ts"].unique(), reverse=True), key="pt_round")
                if rnd_sel != "전체":
                    ddf = ddf[ddf["gm_ts"] == rnd_sel]
                stg_sel = h3.selectbox("전략", ["전체"] + sorted(ddf["전략명"].unique()), key="pt_strategy")
                if stg_sel != "전체":
                    ddf = ddf[ddf["전략명"] == stg_sel]
                res_sel = h4.selectbox("결과", ["전체", "적중", "미적중", "대기"], key="pt_status")
                if res_sel != "전체":
                    ddf = ddf[ddf["status"] == res_sel]
                settled = ddf[ddf["status"] != "대기"]
                if len(settled):
                    st.caption(f"표시 {len(ddf):,}건 · 끝난 베팅 {len(settled):,}건 · 적중 {int((settled['status'] == '적중').sum()):,}건 · "
                               f"손익 {int(settled['profit'].sum()):+,}원 (수익률 {settled['profit'].sum() / settled['stake'].sum() * 100:+.1f}%)")
                mark = {"적중": "✅ 적중", "미적중": "❌ 미적중", "대기": "⏳ 대기"}
                done_mask = ddf["status"] != "대기"
                # 환급 = 돌려받은 돈(적중 시 배당 × 금액, 미적중 0), 손익 = 환급 − 건 돈 (적중 1.62배 × 1,000원 → 환급 1,620 · 손익 +620)
                view = pd.DataFrame({
                    "구분": ddf["구분"], "회차": ddf["gm_ts"], "전략": ddf["전략명"], "베팅 내용": ddf["legs"],
                    "배당": ddf["odds"], "건 돈(원)": ddf["stake"], "결과": ddf["status"].map(mark),
                    "환급(원)": ddf["payout"].where(done_mask), "손익(원)": ddf["profit"].where(done_mask),
                })
                st.dataframe(view.head(500), hide_index=True, width="stretch",
                             column_config={"배당": st.column_config.NumberColumn(format="%.2f배"),
                                            "건 돈(원)": st.column_config.NumberColumn(format="localized"),
                                            "환급(원)": st.column_config.NumberColumn(format="localized",
                                                                                   help="돌려받은 돈 (적중 시 배당 × 건 돈, 미적중 0)"),
                                            "손익(원)": st.column_config.NumberColumn(format="%+d", help="환급 − 건 돈"),
                                            "베팅 내용": st.column_config.TextColumn(width="large")})
                st.caption("환급 = 돌려받은 돈, 손익 = 환급 − 건 돈이에요. 예) 1.62배에 1,000원 → 적중 시 환급 1,620원 · 손익 +620원, "
                           "미적중 시 환급 0원 · 손익 −1,000원")
                if len(view) > 500:
                    st.caption(f"최근 500건만 보여요 (전체 {len(view):,}건). 회차·전략으로 걸러 보세요.")


# ==============================================================================
# 사이드바: 🛒 내 선택 — 베트맨 오른쪽 장바구니처럼 어느 탭·스크롤 위치에서도 담은 경기·조합 수·배당이 보임
# (목록에서 고른 내용이 같은 실행에서 반영되도록 스크립트 맨 끝에서 그린다)
# ==============================================================================
def _cart_remove(seq):
    st.session_state["selections"].pop(seq, None)
    st.session_state["editor_version"] += 1   # 목록의 배당 버튼도 선택 해제된 상태로 다시 그림


def _cart_clear():
    st.session_state["selections"] = {}
    st.session_state["editor_version"] += 1


def _save_slip(games_, combos_, gm_ts_):
    """💾 지금 선택(경기 · 고른 선택지 · 조합별 금액·체크)을 기록으로 남김 — 결과가 나오면 자동 채점"""
    kind = "paper" if st.session_state.get("slip_kind") == KIND_LABEL["paper"] else "real"
    add_slip(build_slip(gm_ts_, games_, combos_, st.session_state.get("slip_memo", ""), kind=kind), MY_SLIPS_PATH)
    st.session_state["slip_memo"] = ""
    st.session_state["slip_saved_toast"] = True


def _load_slip(slip):
    """저장한 선택을 다시 담기 (이번 회차 경기만) — 조합별 금액·체크도 함께"""
    st.session_state["selections"] = {
        g["seq"]: {SEL_KEY[p["key"]]: True for p in g["picks"]} for g in slip["games"] if g["seq"] in pred_map}
    for c in slip.get("combos", []):
        st.session_state[f"stake_{c['key']}"] = int(c["stake"])
        st.session_state[f"combo_chk_{c['key']}"] = bool(c.get("picked", True))
    st.session_state["editor_version"] += 1


def _delete_slip(slip_id):
    delete_slip(slip_id, MY_SLIPS_PATH)


def _slip_rows(slip, g, results_idx_):
    """저장한 경기의 베트맨 결과 줄들 (같은 경기번호에 기준값마다 한 줄씩 있을 수 있음)"""
    return results_idx_.get((str(slip.get("gm_ts")), str(g["seq"])), [])


def _slip_result_of(slip, results_idx_):
    """저장한 경기의 실제 결과: home_win/draw/away_win · 'void'(1.0배) · None(아직)
    무효·적중특례, 야구/농구 승패 무승부, 핸디캡·언더오버의 '산 시점 기준값' 채점은 outcome_from_rows 가 처리"""
    return lambda g: outcome_from_rows(g, _slip_rows(slip, g, results_idx_))


# ==============================================================================
# TAB 4: 📒 ④ 내 베팅 기록 — 저장한 선택(실제 구매 · 가상 구매)의 경기별 결과 · 조합별 환급 · 손익
# (저장·불러오기 함수를 쓰므로 그 정의 뒤, 사이드바 앞에서 그린다)
# ==============================================================================
SLIP_STATE = {"hit": ("✅ 적중", "rgba(34,197,94,.15)", "#4ade80"), "miss": ("❌ 미적중", "rgba(239,68,68,.15)", "#f87171"),
              "pending": ("⏳ 결과 대기", "rgba(234,179,8,.15)", "#facc15")}
KIND_STYLE = {"real": ("rgba(59,130,246,.18)", "#60a5fa"), "paper": ("rgba(148,163,184,.2)", "#cbd5e1")}


def _chip(text, bg, fg):
    return (f"<span style='display:inline-block;padding:1px 8px;border-radius:6px;font-size:12px;font-weight:700;"
            f"background:{bg};color:{fg}'>{text}</span>")


def _toggle_kind(slip_id, kind):
    update_slip(slip_id, MY_SLIPS_PATH, kind="paper" if kind == "real" else "real")


def _save_memo(slip_id):
    update_slip(slip_id, MY_SLIPS_PATH, memo=str(st.session_state.get(f"memo_edit_{slip_id}", "")).strip())


with tab4:
    st.subheader("📒 내 베팅 기록")
    st.caption("사이드바 '💾 저장하기'로 남긴 기록이에요. 경기 결과가 나오면 적중·환급금이 자동으로 채워져요 (결과는 1시간마다 갱신). "
               "이 컴퓨터의 user_data 폴더에만 저장되고 GitHub에는 올라가지 않아요.")
    my_all = load_slips(MY_SLIPS_PATH)
    if not my_all:
        st.info("아직 저장한 기록이 없어요. ① 탭에서 경기를 고른 뒤 사이드바의 '💾 저장하기'를 누르세요. "
                "실제로 산 조합은 '실제 구매', 연습은 '가상 구매'로 나눠 저장할 수 있어요.")
    else:
        res_idx4 = load_results_index()
        cur_round4 = str(detect_current_gm_ts(predictions))
        recs = [{"slip": s, "kind": slip_kind(s), "grade": grade_slip(s, _slip_result_of(s, res_idx4))} for s in my_all]

        # 필터: 구분 · 상태 · 회차 · 검색
        f1, f2, f3, f4 = st.columns([1.3, 1.9, 1.0, 1.4], vertical_alignment="bottom")
        with f1:
            kinds_sel = st.pills("구분", list(KIND_LABEL), format_func=KIND_LABEL.get, selection_mode="multi", key="my_kind") or []
        with f2:
            states_sel = st.pills("상태", list(SLIP_STATE), format_func=lambda k: SLIP_STATE[k][0], selection_mode="multi",
                                  key="my_state") or []
        with f3:
            round_opts = ["전체 회차"] + sorted({str(r["slip"].get("gm_ts")) for r in recs}, reverse=True)
            keep_valid("my_round", round_opts)
            round_sel = st.selectbox("회차", round_opts, key="my_round")
        with f4:
            q4 = st.text_input("🔎 메모·팀 검색", key="my_search", placeholder="예: 실제, 토트넘").strip().lower()

        def _hay(s):
            return (s.get("memo", "") + " " + " ".join(f"{g['home']} {g['away']}" for g in s.get("games", []))).lower()

        view = [r for r in recs
                if (not kinds_sel or r["kind"] in kinds_sel) and (not states_sel or r["grade"]["state"] in states_sel)
                and (round_sel == "전체 회차" or str(r["slip"].get("gm_ts")) == round_sel) and (not q4 or q4 in _hay(r["slip"]))]

        # 요약 (지금 필터 기준 · 손익은 결과 나온 기록만)
        done = [r for r in view if r["grade"]["state"] != "pending"]
        inv = sum(r["grade"]["stake"] for r in done)
        ret = sum(r["grade"]["payout"] for r in done)
        pnl = ret - inv
        render_tiles([
            ("📒 기록", f"{len(view):,}건", f"결과 나옴 {len(done)} · 결과 대기 {len(view) - len(done)}"),
            ("💰 총 투자", f"{inv:,}원", "결과 나온 기록만"),
            ("🎯 총 환급", f"{ret:,}원", f"적중 {sum(1 for r in done if r['grade']['state'] == 'hit')} / {len(done)}건"),
            ("📊 손익", f"<span style='color:{'#f87171' if pnl < 0 else '#34d399'}'>{pnl:+,}원</span>",
             f"수익률 {pnl / inv * 100:+.1f}%" if inv else "결과가 나오면 표시"),
        ])
        kind_lines = []
        for k, lab in KIND_LABEL.items():
            kd = [r for r in done if r["kind"] == k]
            if kd:
                ki, kr = sum(r["grade"]["stake"] for r in kd), sum(r["grade"]["payout"] for r in kd)
                kind_lines.append(f"{lab} {len(kd)}건: 투자 {ki:,}원 · 손익 <b>{kr - ki:+,}원</b> ({(kr - ki) / ki * 100:+.1f}%)" if ki
                                  else f"{lab} {len(kd)}건")
        if len(kind_lines) > 1:
            st.markdown("<div style='padding-bottom:10px'><div style='font-size:14px'>💬 " + " · ".join(kind_lines) + "</div></div>",
                        unsafe_allow_html=True)

        # 누적 손익 (결과 나온 기록 · 저장한 순서 · 구분별 선) — 점이 너무 적으면 그래프가 의미 없어 3건부터
        if len(done) >= 3:
            acc, pts = {k: 0 for k in KIND_LABEL}, []
            kinds_in = [k for k in KIND_LABEL if any(x["kind"] == k for x in done)]
            for i, r in enumerate(sorted(done, key=lambda r: r["slip"].get("saved_at", "")), start=1):
                acc[r["kind"]] += r["grade"]["payout"] - r["grade"]["stake"]
                pts.append({"결과 나온 순서": i, **{KIND_LABEL[k]: acc[k] for k in kinds_in}})
            st.line_chart(pd.DataFrame(pts).set_index("결과 나온 순서"), height=200)
            st.caption("가로: 결과 나온 기록 순서(저장한 순) · 세로: 누적 손익(원) — 위로 가면 벌고, 아래로 가면 잃고 있다는 뜻이에요.")

        # 엑셀(CSV)로 받기 — 기록 한 건 = 한 줄
        csv_rows = [{
            "저장 시각": r["slip"].get("saved_at", ""), "회차": r["slip"].get("gm_ts", ""), "구분": KIND_LABEL[r["kind"]],
            "메모": r["slip"].get("memo", ""),
            "경기": " / ".join(f"{g['seq']} {g['home']} vs {g['away']} [{'·'.join(p['label'] for p in g['picks'])}]"
                             for g in r["slip"].get("games", [])),
            "조합 수": sum(1 for c in r["slip"].get("combos", []) if c.get("picked", True)),
            "투자(원)": r["grade"]["stake"], "상태": SLIP_STATE[r["grade"]["state"]][0], "환급(원)": r["grade"]["payout"],
            "손익(원)": (r["grade"]["payout"] - r["grade"]["stake"]) if r["grade"]["state"] != "pending" else "",
        } for r in sorted(view, key=lambda r: r["slip"].get("saved_at", ""), reverse=True)]
        st.download_button("⬇️ 엑셀(CSV)로 받기", data=pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"my_bets_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="my_csv")

        if not view:
            st.info("조건에 맞는 기록이 없어요. 필터를 바꿔 보세요.")
        shown_v = sorted(view, key=lambda r: r["slip"].get("saved_at", ""), reverse=True)[:50]
        for r in shown_v:
            s, gr, kind = r["slip"], r["grade"], r["kind"]
            sid = s["id"]
            st_txt, st_bg, st_fg = SLIP_STATE[gr["state"]]
            if gr["state"] == "hit":
                st_txt += f" · 환급 {gr['payout']:,}원 ({gr['payout'] - gr['stake']:+,}원)"
            elif gr["state"] == "miss":
                st_txt += f" ({-gr['stake']:+,}원)"
            else:
                st_txt += f" ({gr['done']}/{gr['games']}경기 끝남)"
            with st.container(border=True):
                n_comb = sum(1 for c in s.get("combos", []) if c.get("picked", True))
                head = (f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap'>"
                        f"<span style='font-size:14px'>{_chip(KIND_LABEL[kind], *KIND_STYLE[kind])} "
                        f"<b>{s.get('saved_at', '')}</b> <span style='opacity:.7'>· {s.get('gm_ts', '')}회 · "
                        f"{len(s.get('games', []))}경기 {n_comb}조합 · {gr['stake']:,}원</span>"
                        + (f" · <b>{s['memo']}</b>" if s.get("memo") else "") + "</span>"
                        f"{_chip(st_txt, st_bg, st_fg)}</div>")
                # 경기별: 번호 · 경기 · 고른 것 · 실제 결과
                g_rows = []
                for g in s.get("games", []):
                    rows_g = _slip_rows(s, g, res_idx4)
                    out_g = outcome_from_rows(g, rows_g)
                    score_g = display_score(g, rows_g)
                    labels_g = pick_labels({"game_type": g.get("game_type", ""), "category": ""})
                    picks_txt = " / ".join(f"{p['label']} {p['odds']:.2f}" for p in g["picks"])
                    # 핸디캡·언더오버 기준값이 산 뒤에 바뀌어 같은 기준값 결과 줄이 없으면 실제 점수로 다시 채점한 것
                    regraded = bool(rows_g) and ("핸디캡" in str(g.get("game_type")) or "언더오버" in str(g.get("game_type"))) \
                        and line_row(g, rows_g) is None
                    if not rows_g:
                        res_txt = "<span style='opacity:.6'>진행 전 · 결과 대기</span>"
                    elif out_g == "void":
                        why_void = "무승부" if any(str(r.get("game_result")) == "1" for r in rows_g) else "무효·적중특례"
                        res_txt = f"↩️ {why_void} → 배당 1.00 <span style='opacity:.7'>{score_g}</span>"
                    elif out_g is None:
                        res_txt = "<span style='opacity:.6'>결과 확인 중</span>"
                    else:
                        ok = out_g in {p["key"] for p in g["picks"]}
                        res_txt = (f"{'✅' if ok else '❌'} <b>{labels_g.get(out_g, '?')}</b> "
                                   f"<span style='opacity:.7'>{score_g}</span>"
                                   + (" <span style='font-size:11px;opacity:.6'>(산 기준값으로 채점)</span>" if regraded else ""))
                    g_rows.append(
                        "<div style='display:grid;grid-template-columns:44px 1fr 150px 130px;gap:8px;align-items:center;"
                        "padding:4px 0;font-size:13px;border-top:1px solid rgba(148,163,184,.12)'>"
                        f"<b>{g['seq']}</b><span style='word-break:keep-all'>{g['home']} <span style='opacity:.6'>vs</span> {g['away']}"
                        f" <span style='opacity:.6;font-size:12px'>{type_name({'game_type': g.get('game_type', '')})}</span></span>"
                        f"<span>{picks_txt}</span><span>{res_txt}</span></div>")
                st.markdown("<div style='padding-bottom:12px'>" + head
                            + "<div style='display:grid;grid-template-columns:44px 1fr 150px 130px;gap:8px;font-size:11px;"
                              "opacity:.6;margin-top:8px'><span>번호</span><span>경기</span><span>고른 것</span><span>결과</span></div>"
                            + "".join(g_rows) + "</div>", unsafe_allow_html=True)

                with st.expander(f"조합 {n_comb}개 보기 (조합별 배당 · 금액 · 결과 · 환급)"):
                    lab_of = {(g["seq"], p["key"]): p["label"] for g in s.get("games", []) for p in g["picks"]}
                    st.dataframe(pd.DataFrame([{
                        "조합": " × ".join(f"{seq} {lab_of.get((seq, k), k)}" for seq, k, _ in c["legs"]),
                        "배당": calculate_combined_odds([o for _, _, o in c["legs"]]),
                        "금액(원)": c["stake"],
                        "결과": {"hit": "✅ 적중", "miss": "❌", "pending": "⏳ 대기"}[c["state"]],
                        "환급(원)": c["payout"] if c["state"] == "hit" else 0,
                    } for c in combo_results(s, _slip_result_of(s, res_idx4))]), hide_index=True, width="stretch",
                        column_config={"배당": st.column_config.NumberColumn(format="%.2f배"),
                                       "금액(원)": st.column_config.NumberColumn(format="localized"),
                                       "환급(원)": st.column_config.NumberColumn(format="localized")})

                a1, a2, a3, a4, _ = st.columns([1, 1.3, 1, 1, 1.7])
                a1.button("불러오기", key=f"my_load_{sid}", width="stretch", on_click=_load_slip, args=(s,),
                          disabled=str(s.get("gm_ts")) != cur_round4)
                a2.button(f"{KIND_LABEL['paper' if kind == 'real' else 'real']}로 바꾸기", key=f"my_kind_{sid}", width="stretch",
                          on_click=_toggle_kind, args=(sid, kind))
                with a3.popover("✏️ 메모", width="stretch"):
                    st.text_input("메모", value=s.get("memo", ""), key=f"memo_edit_{sid}")
                    st.button("저장", key=f"my_memo_save_{sid}", on_click=_save_memo, args=(sid,))
                with a4.popover("🗑️ 삭제", width="stretch"):
                    st.caption("이 기록을 지울까요? 되돌릴 수 없어요.")
                    st.button("삭제", key=f"my_del_{sid}", type="primary", on_click=_delete_slip, args=(sid,))
        if len(view) > len(shown_v):
            st.caption(f"최근 {len(shown_v)}건만 보여요 (전체 {len(view):,}건). 필터로 걸러 보거나 CSV로 받아 보세요.")


with st.sidebar:
    cart = selection_status()
    combo_state = {}   # 조합 내용 키 → {금액, 체크} (자동 저장용)
    st.markdown(f"### 🛒 내 선택 <span style='font-size:15px;opacity:.7'>{cart['n']}경기</span>", unsafe_allow_html=True)
    if st.session_state.pop("slip_saved_toast", False):
        st.toast("💾 저장했어요. 아래 '저장한 선택'에서 볼 수 있어요.")
    if not cart["active"]:
        st.caption("① 맞춤조건구매 탭에서 배당 버튼을 누르면 여기에 담겨요. 고른 경기는 자동 저장돼서 새로고침해도 남아요.")
    else:
        cart_now = datetime.now().timestamp() * 1000
        chip_css = "display:inline-block;background:#1e3a5f;border-radius:12px;padding:1px 9px;margin:2px 4px 0 0;font-size:13px"
        n_closed = 0
        for seq, v in sorted(cart["active"].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            m = v["match"]
            closed = is_sale_closed(m, cart_now)
            n_closed += closed
            tags = line_chip(m) + (" <span style='font-size:11px;color:#f87171'>🔒 마감</span>" if closed else "")
            picks = "".join(f"<span style='{chip_css}'>{p['selection_name']} {p['odds']:.2f}</span>" for p in v["picks"])
            with st.container(border=True):
                a, b = st.columns([7, 1], vertical_alignment="top")
                # keep-all: 팀 이름이 글자 중간이 아니라 띄어쓰기에서 줄바꿈
                # 마지막 padding: Streamlit 글 블록의 -1rem 아래 여백 때문에 배당 칩이 카드 아래 테두리에 붙지 않게
                a.markdown(
                    f"<div style='font-size:12px;opacity:.8'>{SPORT_ICONS.get(m.get('sport'), '')} <b>{seq}</b> {type_name(m)} {tags}</div>"
                    f"<div style='font-size:14px;line-height:1.35;word-break:keep-all'><b>{m.get('home_team', '')}</b> "
                    f"<span style='opacity:.6'>vs</span> <b>{m.get('away_team', '')}</b></div>"
                    f"<div style='padding-bottom:14px'>{picks}</div>",
                    unsafe_allow_html=True)
                # help(말풍선) 없음: 누르면 버튼이 사라져 말풍선이 화면에 남을 수 있음
                b.button("✕", key=f"cart_rm_{seq}", type="tertiary", on_click=_cart_remove, args=(seq,))

        # 베트맨 장바구니처럼: 선택 경기수 · 조합 수 · 예상 적중 배당 · 투표금액 · 예상 적중금액
        # 조합별 금액·선택 체크는 STEP 02 표와 같은 값(조합 내용 키)을 읽어 본문 합계와 항상 같게 한다
        unit = 1000 if cart["n"] == 1 else 100
        stake = max(unit, (int(st.session_state.get("batch_stake", 1000)) // unit) * unit)
        slip_combos = []   # 저장용: 조합마다 [경기, 선택지, 배당] · 금액 · 체크
        if cart["combos"] <= MAX_SAVED_COMBOS:
            picked = []   # (조합 배당, 금액) — 체크된 조합만
            for combo in itertools.product(*[[(seq, p) for p in v["picks"]] for seq, v in sorted(cart["active"].items())]):
                ck = combo_key((seq, p["selection"]) for seq, p in combo)
                chk = bool(st.session_state.get(f"combo_chk_{ck}", True))
                s_c = max(unit, (int(st.session_state.get(f"stake_{ck}", stake)) // unit) * unit)
                o_c = calculate_combined_odds([p["odds"] for _, p in combo])
                combo_state[ck] = {"stake": s_c, "picked": chk}
                slip_combos.append({"key": ck, "legs": [[seq, p["selection"], float(p["odds"])] for seq, p in combo],
                                    "stake": s_c, "picked": chk})
                if chk:
                    picked.append((o_c, s_c))
        else:   # 조합이 아주 많으면 모두 같은 금액·전부 선택으로 어림
            picked = [(cart["odds_lo"], stake), (cart["odds_hi"], stake)]
        n_pick = len(picked) if cart["combos"] <= MAX_SAVED_COMBOS else cart["combos"]
        total = sum(s for _, s in picked) if cart["combos"] <= MAX_SAVED_COMBOS else stake * cart["combos"]
        pays = [calculate_payout(o, s) for o, s in picked]
        lo, hi = (min(o for o, _ in picked), max(o for o, _ in picked)) if picked else (0.0, 0.0)
        odds_txt = "-" if not picked else (f"{lo:,.2f}배" if lo == hi else f"{lo:,.2f} ~ {hi:,.2f}배")
        pay_txt = "-" if not pays else (f"{min(pays):,}원" if min(pays) == max(pays) else f"{min(pays):,} ~ {max(pays):,}원")
        stakes = {s for _, s in picked}
        rows = [("선택 경기수", f"{cart['n']} 경기"),
                ("조합 수", f"{cart['combos']:,} 개" if n_pick == cart["combos"] else f"{n_pick:,} / {cart['combos']:,} 개 선택"),
                ("예상 적중 배당", odds_txt),
                ("조합당 금액", "-" if not stakes else (f"{next(iter(stakes)):,}원" if len(stakes) == 1 else "조합마다 다름")),
                ("총 투표금액", f"{total:,}원"), ("적중 시 환급", pay_txt)]
        st.markdown(
            "<div style='font-size:13px;line-height:1.9;margin:4px 0 6px'>" + "".join(
                f"<div style='display:flex;justify-content:space-between'><span style='opacity:.75'>{k}</span><b>{val}</b></div>"
                for k, val in rows) + "</div>", unsafe_allow_html=True)
        st.caption("조합당 금액은 STEP 02에서 바꿀 수 있어요. 여러 개를 고른 경기는 조합마다 배당이 달라 범위로 보여요.")
        if total > 100_000:
            st.error("총 투표금액이 베트맨 1회차 구매 한도(100,000원)를 넘어요.")
        if n_closed:
            st.warning(f"발매 마감된 경기 {n_closed}개는 살 수 없어요. ✕로 빼 주세요.")
        bg, fg = ("rgba(16,185,129,.15)", "#34d399") if cart["valid"] else ("rgba(245,158,11,.15)", "#fbbf24")
        # 바깥 padding: Streamlit 글 블록의 -1rem 아래 여백 때문에 아래 버튼과 붙지 않게
        st.markdown(f"<div style='padding-bottom:12px'><div style='font-size:13px;line-height:1.5;padding:8px 10px;"
                    f"border-radius:8px;background:{bg};color:{fg}'>{cart['msg']}</div></div>", unsafe_allow_html=True)
        st.button("➡️ 조합 결과 보기 (STEP 02)", type="primary", disabled=not cart["valid"], width="stretch",
                  on_click=_go_step, args=(2,), key="cart_go_step2")   # help 없음: 누르면 화면이 바뀌어 말풍선이 남음
        st.button("🗑️ 모두 빼기", width="stretch", on_click=_cart_clear, key="cart_clear")

        # 💾 저장: 실제로 산 조합 등을 기록으로 남김 (결과가 나오면 아래 목록에서 자동 채점)
        games_payload = [{
            "seq": seq, "home": v["match"].get("home_team", ""), "away": v["match"].get("away_team", ""),
            "game_type": v["match"].get("game_type", ""), "line": float(v["match"].get("line") or 0.0),
            "league": v["match"].get("league", ""), "match_time": v["match"].get("match_time", ""),
            "picks": [{"key": p["selection"], "label": p["selection_name"], "odds": float(p["odds"])} for p in v["picks"]],
        } for seq, v in sorted(cart["active"].items())]
        slip_round = next((v["match"].get("gm_ts") for v in cart["active"].values() if v["match"].get("gm_ts")),
                          detect_current_gm_ts(predictions))
        st.markdown("<div style='font-size:13px;font-weight:700;margin-top:10px'>💾 이 선택 저장</div>", unsafe_allow_html=True)
        st.radio("구분", [KIND_LABEL["real"], KIND_LABEL["paper"]], key="slip_kind", horizontal=True, label_visibility="collapsed")
        st.text_input("메모", key="slip_memo", placeholder="메모 (선택) 예: 소액 테스트", label_visibility="collapsed")
        st.button("💾 저장하기", width="stretch", key="slip_save", on_click=_save_slip,
                  args=(games_payload, slip_combos, slip_round), disabled=not slip_combos)
        if not slip_combos:
            st.caption(f"조합이 {MAX_SAVED_COMBOS:,}개를 넘으면 저장할 수 없어요.")

    # 📂 저장한 기록: 사이드바는 최근 3건만 짧게 — 전체 · 경기별 결과 · 불러오기 · 삭제는 📒 ④ 탭
    my_slips = load_slips(MY_SLIPS_PATH)
    if my_slips:
        res_idx_s = load_results_index()
        recent_html = []
        for s in reversed(my_slips[-3:]):
            gr = grade_slip(s, _slip_result_of(s, res_idx_s))
            _, bg, fg = SLIP_STATE[gr["state"]]
            txt = {"hit": f"✅ {gr['payout']:,}원", "miss": "❌ 미적중", "pending": "⏳ 대기"}[gr["state"]]   # 좁은 칸이라 짧게
            recent_html.append(
                "<div style='display:flex;justify-content:space-between;align-items:center;gap:6px;padding:5px 0;"
                "border-top:1px solid rgba(148,163,184,.15);font-size:12px'>"
                f"<span>{_chip(KIND_LABEL[slip_kind(s)][:2], *KIND_STYLE[slip_kind(s)])} {s.get('saved_at', '')[5:]} "
                f"· {gr['stake']:,}원</span>{_chip(txt, bg, fg)}</div>")
        st.markdown("<div style='padding:14px 0 12px'><div style='font-size:13px;font-weight:700;margin-bottom:4px'>"
                    f"📂 저장한 기록 {len(my_slips)}건 <span style='font-weight:400;opacity:.7'>(최근 3건)</span></div>"
                    + "".join(recent_html) + "<div style='font-size:12px;opacity:.7;margin-top:4px'>전체 기록 · 경기별 결과 · "
                    "불러오기 · 삭제는 📒 ④ 내 베팅 기록 탭에서</div></div>", unsafe_allow_html=True)

    # 지금 고르는 중인 선택 자동 저장 (바뀌었을 때만 파일에 씀) → 새로고침·재시작 후 같은 회차면 복원
    cur_payload = {"gm_ts": str(detect_current_gm_ts(predictions)), "selections": st.session_state["selections"],
                   "batch_stake": st.session_state.get("batch_stake"), "combo_state": combo_state}
    cur_sig = json.dumps(cur_payload, sort_keys=True, ensure_ascii=False)
    if st.session_state.get("_cur_saved_sig") != cur_sig:
        try:
            save_json(MY_CURRENT_PATH, cur_payload)
            st.session_state["_cur_saved_sig"] = cur_sig
        except Exception:
            pass
