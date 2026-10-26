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

# Streamlit 세션 유지 중 코드 변경 시 서브모듈 핫 리로딩 보장
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("models") or mod_name.startswith("tracker"):
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


def _metric_note(label, value, note=None):
    """지표 + 화살표 없는 보조 설명 (st.metric의 delta는 ↑/↓ 화살표가 붙어 증감처럼 보이므로 설명에는 쓰지 않음)"""
    st.metric(label, value)
    if note:
        st.caption(note)


# 3-1. 경기 결과 · 경기 전 AI 예측 기록 (맞춤조건구매 목록의 적중 여부 표시용)
PICK_KEYS = ("home_win", "draw", "away_win")
SEL_KEY = {"home_win": "home", "draw": "draw", "away_win": "away"}
RESULT_CODE_TO_KEY = {"0": "home_win", "1": "draw", "2": "away_win"}   # 2-way(언더/오버, 홀/짝)도 0=앞, 2=뒤
AI_SOURCES = ("betman_corrector", "market_corrector", "betman_markets")


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
    cands =results_idx.get((str(gm_ts), str(m.get("match_seq"))), [])
    if len(cands) > 1:
        line = float(m.get("line") or 0.0)
        for c in cands:
            if any(abs(float(c.get(k) or 0.0) - line) < 1e-9 for k in ("handi_win", "handi_val", "handi_lose")):
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
    else:
        valid, msg = True, f"✅ 베트맨 프로토 규정 통과! ({n}경기 선택 / 총 {combos:,}개 조합)"
    return {"active": active, "n": n, "combos": combos, "odds_lo": odds_lo, "odds_hi": odds_hi,
            "valid": valid, "msg": msg}


# 4. 세션 상태 초기화 (단일 상태: selections)
if "selections" not in st.session_state:
    st.session_state["selections"] = {}  # { match_seq: {"home": bool, "draw": bool, "away": bool} }

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
tab1, tab2, tab3 = st.tabs([
    "🛒 ① 맞춤조건구매 (시뮬레이터)",
    "🔍 ② AI 경기 분석 (조회 전용)",
    "📊 ③ 성과 검증 (실제 경기 대조 & 통계)"
])

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
        """버튼 콜백: 단계 값과 단계 버튼 상태를 함께 바꾼다 (버튼 상태가 남아 단계가 되돌아가는 문제 방지)"""
        st.session_state["custom_step"] = n
        st.session_state["step_nav"] = STEP_LABELS[n - 1]

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
        _chip = "display:inline-block;margin:0 14px 6px 0;font-size:13px;color:#cbd5e1"
        st.markdown(
            "<div style='margin:-4px 0 8px 0'>"
            f"<span style='{_chip}'>⭐ AI 확률 1순위 <span style='color:#94a3b8'>(AI 없으면 배당 1순위)</span></span>"
            f"<span style='{_chip}'>💎 AI 기준 기대가치 플러스 <span style='color:#f59e0b'>(검증 중)</span></span>"
            f"<span style='{_chip}'>⚠️ AI 예상이 배당과 다름</span>"
            f"<span style='{_chip}'>✅ ❌ 끝난 경기의 적중 여부</span>"
            "</div>",
            unsafe_allow_html=True)

        # 🧩 조합 찾기 — 데이터로 확인한 원칙(저배당 정배 축, AI ▲, 💎 우선, 커버 없음)으로 2~3폴더 후보
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
            """💎 = AI 확률 기준 기대가치(EV) 플러스 선택지 (1순위가 아니어도 표시)"""
            out = {}
            if ai_probs:
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
                    if res is None and ev.get("is_recommended"):
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

            # 2) 금액 · 정렬 · 빠른 선택 — 금액은 바꾸는 즉시 모든 조합에 적용 (표에서 조합별로 따로 바꿀 수도 있음)
            def _apply_stake(v):
                v = max(unit, min(100000, (int(v) // unit) * unit))
                st.session_state["batch_stake"] = v
                st.session_state["batch_stake_input"] = v
                for c in combos:
                    st.session_state[f"stake_{c['combo_index']}"] = v
                _bump_editor()

            def _quick_select(mode):
                for c in combos:
                    st.session_state[f"combo_chk_{c['combo_index']}"] = (
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
                idx = c["combo_index"]
                st.session_state.setdefault(f"combo_chk_{idx}", True)
                st.session_state.setdefault(f"stake_{idx}", default_stake)
                row = {"선택": bool(st.session_state[f"combo_chk_{idx}"]), "#": idx}
                for p in c["picks"]:
                    row[match_cols[str(p["match_seq"])]] = f"{p['selection_name']} {float(p['odds']):.2f}"
                stake = max(unit, (int(st.session_state[f"stake_{idx}"]) // unit) * unit)
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
            row_ids = [c["combo_index"] for c in combos]

            def _sync_editor(key=editor_key, ids=row_ids, unit_=unit):
                """편집 직후(재실행 전) 세션에 반영하고 표를 새로 그린다.
                이렇게 하지 않으면 '적중 시 환급' 같은 계산 열이 이전 금액 기준으로 한 번 늦게 표시된다."""
                for pos, ch in st.session_state.get(key, {}).get("edited_rows", {}).items():
                    idx = ids[int(pos)]
                    if "선택" in ch:
                        st.session_state[f"combo_chk_{idx}"] = bool(ch["선택"])
                    if ch.get("투표금액") is not None:
                        v = int(ch["투표금액"])
                        st.session_state[f"stake_{idx}"] = max(unit_, min(100000, (v // unit_) * unit_))
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
                st.session_state[f"combo_chk_{idx}"] = bool(r["선택"])
                st.session_state[f"stake_{idx}"] = stake
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
                    why = (f" 베트맨 마진(경기당 약 15%)이 {n_m}경기만큼 겹쳐 평균 환급률이 약 {0.85 ** n_m * 100:.0f}%이기 때문이에요."
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
                        st.caption(f"규칙: {describe_rule(rule)} — 과거 회차마다 배당이 가장 비슷한(±20%) 경기로 똑같이 걸고 **실제 결과**로 계산합니다.")
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
                        r1, r2, r3, r4 = st.columns(4)
                        with r1:
                            st.metric("적용된 과거 회차", f"{r['rounds']}회차")
                            st.caption(f"회차당 {r['combos_per_round']}조합")
                        with r2:
                            st.metric("적중한 회차", f"{r['hit_rate_pct']:.0f}%")
                            st.caption(f"{r['hit_rounds']}회 적중")
                        with r3:
                            st.metric("실제 수익률", f"{r['roi_pct']:+.1f}%")
                            if "bankers_only_roi_pct" in r:
                                st.caption(f"축 경기만 걸었다면 {r['bankers_only_roi_pct']:+.1f}%")
                        with r4:
                            st.metric("최대 연속 미적중", f"{r['max_losing_streak']}회차")
                            st.caption(f"최고 적중 {r['best_round_multiple']:.1f}배")
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
                            st.session_state[f"stake_{c['combo_index']}"] = per
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
                    "ev": ev, "gems": {k: v for k, v in ev.items() if ai and v > 0},
                    "best_ev": max(ev.values()) if ev else -1.0}

        rows_all = [a for a in (_analyze(m) for m in predictions) if a["opts"]]

        # C. 필터: ① 탭과 같은 배지
        T2_QUICK = {"ai": "🧠 AI 적용만", "diff": "⚠️ 배당과 다름", "gem": "💎 플러스", "single": "⚡ 한경기만", "open": "⏳ 마감 전만"}
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
            ("💎 기대가치 플러스", f"{sum(1 for a in t_rows if a['gems']):,}경기", "AI 확률 기준 · 검증 중"),
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
                if bev > 0 and a["ai"]:
                    says.append(f"💎 <b>{lab[bk]}</b> 기대가치 {bev:+.1f}%: AI 기준으로는 플러스지만 아직 검증 중이에요.")
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
                    c4.button("🔒 마감" if closed else ("✓ 담김" if in_cart else "🛒 담기"), key=f"t2_cart_{seq}_{k}",
                              type="primary" if in_cart and not closed else "secondary", disabled=closed,
                              width="stretch", on_click=_t2_toggle, args=(seq, k),
                              help=None if closed else ("한 번 더 누르면 빠져요" if in_cart else "🛒 내 선택에 담기"))

                if m.get("top_scores"):
                    st.caption("🎯 예상 스코어: " + " · ".join(f"{s['score']} ({s['prob']}%)" for s in m["top_scores"]))

                # 최근 경기 · 맞대결: 색 점 한 줄, 경기별 목록은 펼쳐서
                hf, af, h2h = m.get("home_form") or {}, m.get("away_form") or {}, m.get("h2h") or {}
                if hf.get("has_data") or af.get("has_data"):
                    dots = {"W": "🟢", "D": "🟡", "L": "🔴"}

                    def _form_line(icon, team, f):
                        if not f.get("has_data"):
                            return f"{icon} <b>{team}</b> <span style='opacity:.6'>기록 없음</span>"
                        return (f"{icon} <b>{team}</b> {''.join(dots.get(r, '⚪') for r in f.get('form_sequence', []))} "
                                f"<span style='opacity:.75'>{f.get('record_str', '')} · 평균 득 {f.get('avg_scored', 0)} / "
                                f"실 {f.get('avg_conceded', 0)}</span>")

                    lines = [_form_line("🏠", m.get("home_team", ""), hf), _form_line("✈️", m.get("away_team", ""), af)]
                    if h2h.get("has_data"):
                        lines.append(f"🤝 맞대결 최근 {h2h.get('total_matches', 0)}경기: {m.get('home_team', '')} {h2h.get('home_wins', 0)}승 · "
                                     f"무 {h2h.get('draws', 0)} · {m.get('away_team', '')} {h2h.get('away_wins', 0)}승")
                    st.markdown("<div style='padding:6px 0 12px'><div style='font-size:12px;opacity:.7'>"
                                "최근 경기 (왼쪽이 가장 최근 · 🟢승 🟡무 🔴패)</div>"
                                "<div style='font-size:14px;line-height:1.8'>" + "<br>".join(lines) + "</div></div>",
                                unsafe_allow_html=True)
                    with st.expander("경기별 보기"):
                        fc1, fc2 = st.columns(2)
                        for col, team, f in ((fc1, m.get("home_team", ""), hf), (fc2, m.get("away_team", ""), af)):
                            with col:
                                st.markdown(f"**{team}**")
                                for lm in f.get("last_matches", []):
                                    st.caption(f"{dots.get(lm.get('result'), '⚪')} {lm.get('date')} vs {lm.get('opponent')} · {lm.get('score')}")
                        for hm in h2h.get("matches", []) if h2h.get("has_data") else []:
                            st.caption(f"🤝 {hm.get('date')} {hm.get('home')} vs {hm.get('away')} · {hm.get('score')}")


# ==============================================================================
# TAB 3: 📊 ③ 성과 검증 (실제 경기 대조 & 통계)
# ==============================================================================
with tab3:
    # ------------------------------------------------------------------
    # A. 공식 AI 성적표 — 경기 시작 전에 고정 저장한 예측만 실제 결과로 채점 (사후 정보 없음)
    # ------------------------------------------------------------------
    st.subheader("🏅 공식 AI 성적표 (경기 전 예측만 채점)")
    st.caption("파이프라인이 경기 시작 전에 고정 저장한 AI 확률(data/history/prediction_log)만 실제 결과로 채점합니다. "
               "학습·튜닝에 쓰이지 않은 진짜 앞으로의 성적이며, AI 실력의 공식 지표입니다. Log loss는 낮을수록 정확합니다.")
    card = load_scorecard()
    if card and card.get("model_ll") is not None:
        k1, k2, k3, k4 = st.columns(4)
        ci = card.get("ll_diff_ci", [0, 0])
        for col, label, val, sub in [
            (k1, "채점 경기", f"{card['graded_matches']:,}경기", f"회차 {len(card.get('rounds', []))}개"),
            (k2, "Log loss (AI)", f"{card['model_ll']:.4f}", f"시장 {card['market_ll']:.4f}"),
            (k3, "차이 (음수 = AI 우세)", f"{card['ll_diff']:+.4f}", f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"),
            (k4, "AI 1순위 적중률", f"{card.get('hit_rate_top_pick', 0):.1f}%",
             f"EV>2% {card.get('value_bets', 0)}건 ROI {card.get('roi_pct', 0):+.1f}%"),
        ]:
            with col:
                st.metric(label, val)
                st.caption(sub)
        st.markdown(f"**판정:** {card.get('verdict', '-')}")
        if len(card.get("by_game_type", {})) > 1:
            with st.expander("유형별 성적 (어느 유형에서 AI가 앞서는지)", expanded=False):
                st.dataframe(pd.DataFrame([
                    {"유형": gt, "채점 경기": v["n"], "차이 (음수 = AI 우세)": f"{v['ll_diff']:+.4f}",
                     "95% CI": f"[{v['ll_diff_ci'][0]:+.4f}, {v['ll_diff_ci'][1]:+.4f}]"}
                    for gt, v in sorted(card["by_game_type"].items(), key=lambda x: -x[1]["n"])]),
                    hide_index=True, width="stretch")
        if card.get("gem_bets") is not None:
            gb = card["gem_bets"]
            st.markdown(
                f"**💎 경기 성적** (AI 기준 기대가치 플러스 선택지에 1단위씩 걸었다면): **{gb}건**, 적중 {card.get('gem_hits', 0)}건, "
                f"수익률 **{card.get('gem_roi_pct', 0):+.1f}%** (95% CI [{card.get('gem_roi_ci_low', 0):+.1f}%, {card.get('gem_roi_ci_high', 0):+.1f}%])"
                + (" — 표본이 적어 아직 판단 불가 (300건 이상 필요)" if gb < 300 else ""))
        hist = [h for h in card.get("history", []) if h.get("ll_diff") is not None]
        if len(hist) >= 2:
            h_df = pd.DataFrame(hist).set_index("date")[["model_ll", "market_ll"]].rename(
                columns={"model_ll": "AI", "market_ll": "시장"})
            st.line_chart(h_df, height=220)
    else:
        n = card.get("graded_matches", 0) if card else 0
        st.info(f"⏳ 아직 채점할 경기가 없습니다 (현재 {n}경기). 2026-10-25부터 경기 전 예측을 기록하기 시작했으며, "
                "기록된 경기가 끝날 때마다 자동으로 채점됩니다. 300경기가 쌓이면 판정합니다.")

    status = load_model_status()
    if status:
        with st.container(border=True):
            st.markdown(f"**📚 과거 시험 성적 — {status['name']}** (학습에 쓰지 않은 기간, 실전 확률과 같은 모델)")
            rows_bt = []
            for label, s in status["evals"]:
                ci = s.get("ll_diff_ci", [None, None])
                rows_bt.append({"시험": label, "경기 수": s["samples"], "AI LL": s["model_ll"], "시장 LL": s["market_ll"],
                                "차이": f"{s['ll_diff']:+.4f}", "95% CI": f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                                "EV>2% 베팅": s.get("bets", 0)})
            st.dataframe(pd.DataFrame(rows_bt), width="stretch", hide_index=True)

    mm = load_markets_model()
    if mm and mm.get("segments"):
        with st.expander("📚 전 종목 보정 AI — 유형별 과거 시험 (핸디캡·언더오버·홀짝·농구·야구·배구 …)", expanded=False):
            st.caption("유형마다 '베트맨이 일관되게 잘못 매기는 쪽'(강한 쪽 저평가 β, 특정 선택지 편향 c)을 2023~2025년 상반기로 학습하고 "
                       "2026년(TEST)·최근(HOLDOUT)에서 시험했습니다. 둘 다 시장보다 나은 유형만 실전에 씁니다. "
                       f"학습: {mm.get('trained_at', '-')[:10]}")
            seg_rows = []
            for key, v in mm["segments"].items():
                gt, k = key.split("|")
                t, h = v.get("test"), v.get("holdout")
                seg_rows.append({
                    "유형": gt + (" (무 있음)" if k == "3" and "승무패" not in gt else ""),
                    "실전 적용": "✅" if v.get("passes_gate") else ("표본 부족" if v.get("reason") else "❌"),
                    "TEST 경기": t["n"] if t else v["sizes"]["test"],
                    "TEST 차이": f"{t['ll_diff']:+.4f}" if t else "-",
                    "TEST 95% CI": f"[{t['ll_diff_ci'][0]:+.4f}, {t['ll_diff_ci'][1]:+.4f}]" if t else "-",
                    "HOLDOUT 경기": h["n"] if h else 0,
                    "HOLDOUT 차이": f"{h['ll_diff']:+.4f}" if h else "-",
                    "β (강한 쪽 저평가)": round(v["eval_params"]["beta"], 3) if v.get("eval_params") else None,
                })
            st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width="stretch")
            st.caption("차이가 음수면 AI가 시장보다 정확. β > 0 이면 베트맨이 배당 1순위(강한 쪽)를 낮게 잡는다는 뜻입니다.")

    # ------------------------------------------------------------------
    # A-1. 마감 배당 비교 (CLV) — 시장이 나중에 AI 쪽으로 움직였는가
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📈 마감 배당 비교 (CLV)")
    st.caption("AI가 '배당보다 높다(▲)'고 본 선택지의 배당이 마감 때까지 어떻게 움직였는지 봅니다. 다른 배터들의 돈이 나중에 AI 쪽으로 "
               "몰려 배당이 내려가면 AI가 시장보다 먼저 맞게 본 것입니다. 승패 결과보다 운이 덜 섞여 **성적표보다 적은 경기로 방향**을 볼 수 있습니다. "
               "배당 변화는 1시간마다 기록합니다.")
    clv = load_clv()
    if clv and clv.get("matches"):
        mv = clv.get("mean_move_pp") or {}
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric_note("비교 경기", f"{clv['matches']:,}경기", f"배당이 바뀐 경기 {clv['moved']}")
        with c2:
            _metric_note("AI 쪽으로 움직임", f"{clv['toward_ai_pct']:.0f}%" if clv.get("toward_ai_pct") is not None else "-",
                         f"{clv['toward_ai']}건 / 반대 {clv['away_from_ai']}건")
        with c3:
            _metric_note("기준선: 배당 1순위 쪽", f"{clv['fav_toward_pct']:.0f}%" if clv.get("fav_toward_pct") is not None else "-",
                         "강한 쪽 배당이 원래 잘 내려가는지 비교용")
        with c4:
            _metric_note("AI ▲ 평균 확률 변화", f"{mv['mean']:+.2f}%p" if mv.get("mean") is not None else "-",
                         f"95% CI [{mv['ci'][0]:+.2f}, {mv['ci'][1]:+.2f}]" if mv.get("ci") else None)
        st.markdown(f"**판정:** {clv.get('verdict', '-')}")
        g = clv.get("gem_clv_pct") or {}
        if g.get("n"):
            st.caption(f"💎 선택지 {g['n']}건: 처음 배당이 마감 배당보다 평균 {g['mean']:+.2f}% 높음 (플러스 = 마감보다 좋은 배당에 산 셈)")
    else:
        st.info("⏳ 아직 비교할 경기가 없습니다. AI 예측을 기록한 경기가 마감되면 자동으로 쌓입니다.")

    # ------------------------------------------------------------------
    # A-2. 가상 베팅 리그 — 전략별 자동 가상 베팅 (실제 구매 없음)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🧪 가상 베팅 리그 (자동 · 실제 구매 없음)")
    st.caption("새 회차마다 파이프라인이 전략별로 경기 시작 전에 가상 베팅(1,000원 단위)을 고정 저장하고, 결과가 나오면 자동 채점합니다. "
               "AI 전략은 기준선(E. 배당 1순위)보다 나아야 의미가 있으며, 300건 이상 + 수익률 신뢰구간 하한 > 0 일 때만 '수익 입증'입니다.")
    pt = load_paper_trading()
    bf = load_paper_backfill()

    clv_by_strategy = (load_clv() or {}).get("paper_strategies", {})

    def _league_rows(strategies: dict, source: str) -> list:
        rows_ = []
        for key, v in strategies.items():
            cv = clv_by_strategy.get(key, {}) if source == "실시간" else {}
            rows_.append({
                "구분": source, "전략": v["label"], "정산": v["bets"], "대기": v["pending"],
                "적중률": f"{v['hits'] / v['bets'] * 100:.0f}%" if v["bets"] else "-",
                "가상 수익률": f"{v['roi_pct']:+.1f}%" if v["bets"] else "-",
                "95% CI": f"[{v['roi_ci_low']:+.1f}%, {v['roi_ci_high']:+.1f}%]" if v["bets"] else "-",
                "누적 손익(원)": v["profit"], "판정": v["verdict"] if v["bets"] else "-",
                "마감 대비 배당": f"{cv['mean']:+.2f}% ({cv['n']})" if cv.get("n") else "-",
            })
        return rows_

    league_rows = (_league_rows(pt["strategies"], "실시간") if pt and pt.get("strategies") else [])
    if bf and bf.get("strategies"):
        league_rows += _league_rows(bf["strategies"], f"과거 재현 ({bf['rounds'][0]}~{bf['rounds'][1]})")
    if not league_rows:
        st.info("⏳ 아직 가상 베팅 기록이 없습니다. 다음 파이프라인 실행부터 쌓입니다.")
    else:
        st.dataframe(pd.DataFrame(league_rows), hide_index=True, width="stretch",
                     column_config={"누적 손익(원)": st.column_config.NumberColumn(format="localized")})
        st.caption("마감 대비 배당: 산 배당 ÷ 마감 배당 − 1의 경기(다리) 평균 (괄호는 경기 수). 플러스면 마감보다 좋은 배당에 산 것입니다. "
                   "실시간: 경기 시작 전에 고정 저장된 베팅 (2026-10-26, 260127회차부터). "
                   "과거 재현: 배당만 쓰는 전략(D·E)을 베트맨 과거 회차의 경기 전 배당·실제 결과로 재현 — "
                   "AI 전략(A·B·C)은 현재 AI가 그 결과까지 학습했으므로 재현하지 않습니다.")
        curves = {}
        for src, data_ in (("실시간", pt), ("과거 재현", bf)):
            for v in (data_ or {}).get("strategies", {}).values():
                if len(v["cum_profit"]) >= 2:
                    curves[f"{v['label']}" if src == "실시간" else v["label"]] = pd.Series(v["cum_profit"])
        if curves:
            st.line_chart(pd.DataFrame(curves), height=220)
            st.caption("가로축: 정산된 베팅 순서 · 세로축: 누적 가상 손익(원)")

        # 개별 베팅 내역
        with st.expander("📜 가상 베팅 내역 (회차·전략·결과별)", expanded=False):
            det = [{**d, "구분": "실시간", "전략명": STRATEGY_LABELS.get(d["strategy"], d["strategy"])}
                   for d in (pt or {}).get("details", [])]
            det += [{**d, "구분": "과거 재현", "전략명": BACKFILL_LABEL_MAP.get(d["strategy"], d["strategy"])}
                    for d in (bf or {}).get("details", [])]
            if not det:
                st.caption("아직 내역이 없습니다.")
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
                    st.caption(f"표시 {len(ddf):,}건 · 정산 {len(settled):,}건 · 적중 {int((settled['status'] == '적중').sum()):,}건 · "
                               f"손익 {int(settled['profit'].sum()):+,}원 (수익률 {settled['profit'].sum() / settled['stake'].sum() * 100:+.1f}%)")
                mark = {"적중": "✅ 적중", "미적중": "❌ 미적중", "대기": "⏳ 대기"}
                view = pd.DataFrame({
                    "구분": ddf["구분"], "회차": ddf["gm_ts"], "전략": ddf["전략명"], "베팅 내용": ddf["legs"],
                    "배당": ddf["odds"], "결과": ddf["status"].map(mark), "손익(원)": ddf["profit"],
                })
                st.dataframe(view.head(500), hide_index=True, width="stretch",
                             column_config={"배당": st.column_config.NumberColumn(format="%.2f배"),
                                            "손익(원)": st.column_config.NumberColumn(format="localized"),
                                            "베팅 내용": st.column_config.TextColumn(width="large")})
                if len(view) > 500:
                    st.caption(f"최근 500건만 표시합니다 (전체 {len(view):,}건). 회차·전략으로 걸러 보세요.")


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


with st.sidebar:
    cart = selection_status()
    st.markdown(f"### 🛒 내 선택 <span style='font-size:15px;opacity:.7'>{cart['n']}경기</span>", unsafe_allow_html=True)
    if not cart["active"]:
        st.caption("① 맞춤조건구매 탭에서 배당 버튼을 누르면 여기에 담겨요.")
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
                b.button("✕", key=f"cart_rm_{seq}", type="tertiary", on_click=_cart_remove, args=(seq,), help="이 경기 빼기")

        # 베트맨 장바구니처럼: 선택 경기수 · 조합 수 · 예상 적중 배당 · 투표금액 · 예상 적중금액
        unit = 1000 if cart["n"] == 1 else 100
        stake = max(unit, (int(st.session_state.get("batch_stake", 1000)) // unit) * unit)
        lo, hi = cart["odds_lo"], cart["odds_hi"]
        odds_txt = f"{lo:,.2f}배" if lo == hi else f"{lo:,.2f} ~ {hi:,.2f}배"
        pay_lo, pay_hi = calculate_payout(lo, stake), calculate_payout(hi, stake)
        pay_txt = f"{pay_lo:,}원" if pay_lo == pay_hi else f"{pay_lo:,} ~ {pay_hi:,}원"
        total = stake * cart["combos"]
        rows = [("선택 경기수", f"{cart['n']} 경기"), ("조합 수", f"{cart['combos']:,} 개"),
                ("예상 적중 배당", odds_txt), ("조합당 금액", f"{stake:,}원"),
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
                  on_click=_go_step, args=(2,), key="cart_go_step2", help="① 맞춤조건구매 탭에서 열려요")
        st.button("🗑️ 모두 빼기", width="stretch", on_click=_cart_clear, key="cart_clear")
