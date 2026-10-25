"""
dashboard/app.py
================
Streamlit 기반 스포츠 승부 예측 & 가치 픽(Value Bet) 실시간 웹 대시보드
[Tab 1: 🛒 맞춤조건구매 (STEP 01 맞춤조건설정 & STEP 02 조합결과보기 시뮬레이터)]
[Tab 2: 🔍 AI 경기 분석 (조회 전용 - 실시간 배당률, Dixon-Coles 모델 & 블렌딩 비교)]
[Tab 3: 📊 성과 검증 (실제 경기 결과 1:1 대조, 누적 ROI, 95% 신뢰구간 & CLV 통계)]
"""

import sys
import os
from pathlib import Path
import json
import re
from datetime import datetime
import streamlit as st
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

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="스포츠 승부예측 & 맞춤조건구매 대시보드",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. 커스텀 CSS 스타일링 (번역 방지 및 반응형 배지 UI, 라이트/다크 테마 호환)
st.markdown("""
<meta name="google" content="notranslate">
<style>
    .notranslate { translate: no !important; }
    .badge-s { background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-a { background-color: #10b981; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-b { background-color: #3b82f6; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; }
    .badge-pass { background-color: #64748b; color: white; padding: 3px 8px; border-radius: 6px; font-size: 12px; }
    .badge-single { background-color: #f59e0b; color: #1e293b; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; margin-left: 6px; }
    .badge-cat { background-color: #1e293b; color: #94a3b8; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 11px; margin-left: 6px; border: 1px solid #334155; }
    .match-meta-bar {
        background-color: rgba(30, 41, 59, 0.6);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 8px;
        display: inline-block;
    }
    .step-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .metric-card-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# 3. 데이터 로딩 헬퍼
@st.cache_data(ttl=60)
def load_data():
    pred_path = project_root / "data" / "predictions.json"
    stats_path = project_root / "data" / "performance_stats.json"
    
    preds = []
    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
            
    stats = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            
    return preds, stats


predictions, stats = load_data()


@st.cache_data(ttl=60)
def load_baseline_summary():
    """evaluation/BASELINE.md에서 최신 모델 vs 시장 기준선 검증 지표를 로드합니다."""
    baseline_path = project_root / "evaluation" / "BASELINE.md"
    if not baseline_path.exists():
        return None
    try:
        content = baseline_path.read_text(encoding="utf-8")
        m_bm = re.search(r"실전 베트맨.*?Log Loss:\s*모델\s*`([0-9.]+)`\s*vs\s*시장\s*`([0-9.]+)`", content, re.DOTALL)
        m_csv = re.search(r"과거 유럽 5대 리그.*?Log Loss:\s*모델\s*`([0-9.]+)`\s*vs\s*시장\s*`([0-9.]+)`", content, re.DOTALL)
        return {
            "betman_model_ll": float(m_bm.group(1)) if m_bm else 0.9622,
            "betman_market_ll": float(m_bm.group(2)) if m_bm else 0.9467,
            "test_model_ll": float(m_csv.group(1)) if m_csv else 1.0010,
            "test_market_ll": float(m_csv.group(2)) if m_csv else 0.9790,
        }
    except Exception:
        return None


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


@st.cache_data(ttl=60)
def load_scorecard():
    path = project_root / "data" / "model_scorecard.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_grade_badge_html(grade_raw: str) -> str:
    """등급 문자열을 표준 HTML 배지로 변환합니다."""
    if "S (" in grade_raw:
        return "<span class='badge-s'>GRADE S (참고용 - 고승률)</span>"
    elif "A (" in grade_raw:
        return "<span class='badge-a'>GRADE A (참고용 - 밸런스)</span>"
    elif "B (" in grade_raw:
        return "<span class='badge-b'>GRADE B (참고용 - 고배당)</span>"
    return "<span class='badge-pass'>PASS (참고용)</span>"


def get_result_badge_html(is_hit: bool) -> str:
    """적중 여부 HTML 배지"""
    if is_hit:
        return "<span style='background:#10b981; color:white; padding:3px 8px; border-radius:6px; font-weight:800; font-size:12px;'>✅ 적중!</span>"
    return "<span style='background:#ef4444; color:white; padding:3px 8px; border-radius:6px; font-weight:800; font-size:12px;'>❌ 미적중</span>"


def generate_bet_slip_text(picks, combined_odds, combined_prob_pct=None, combined_ev_pct=None, stake=None):
    """
    베트맨 프로토 마킹용 텍스트 슬립 생성 (카카오톡/메모장 복사용)
    """
    lines = [
        "📋 [베트맨 프로토 승부식 베팅 슬립]",
        "────────────────────────────────────────"
    ]
    for idx, p in enumerate(picks, 1):
        m_seq = p.get("match_seq", "")
        league = p.get("league", "")
        home = p.get("home_team", "")
        away = p.get("away_team", "")
        cat = p.get("category_label", p.get("game_type", "승무패"))
        pick_name = p.get("selection_name", "")
        odds = p.get("odds", 0.0)
        prob = p.get("prob", 0.0)
        lines.append(f"[{idx}] 경기 #{m_seq} ({league})")
        lines.append(f"    {home} vs {away}")
        lines.append(f"    👉 픽: [{cat}] {pick_name} | 배당: {odds}배 (AI 승률 {prob*100:.1f}%)")
        lines.append("")
        
    lines.append("────────────────────────────────────────")
    summary = f"📊 총 {len(picks)}폴더 | 결합 배당률: {combined_odds}배"
    if combined_prob_pct:
        summary += f" | AI 승률: {combined_prob_pct}"
    if combined_ev_pct:
        summary += f" | 기대가치: {combined_ev_pct}"
    lines.append(summary)
    
    if stake and stake > 0:
        payout = calculate_payout(combined_odds, stake)
        net_profit = payout - stake
        lines.append(f"💰 베팅금: {stake:,}원 ➔ 예상 환급금: {payout:,}원 (순수익: {net_profit:+,}원)")
        
    lines.append("────────────────────────────────────────")
    return "\n".join(lines)


# 3-1. 경기 결과 · 경기 전 AI 예측 기록 (맞춤조건구매 목록의 적중 여부 표시용)
PICK_KEYS = ("home_win", "draw", "away_win")
SEL_KEY = {"home_win": "home", "draw": "draw", "away_win": "away"}
RESULT_CODE_TO_KEY = {"0": "home_win", "1": "draw", "2": "away_win"}   # 2-way(언더/오버, 홀/짝)도 0=앞, 2=뒤
AI_SOURCES = ("betman_corrector", "market_corrector")


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
    snap_dir = project_root / "data" / "history" / "odds_snapshots"
    names = sorted(p.stem for p in snap_dir.glob("*.json")) if snap_dir.exists() else []
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
st.caption("베트맨 프로토 승부식 공식 계산 규칙 준수 · 복수 선택 직교 전개 시뮬레이터 · Dixon-Coles & 블렌딩 통계 모델")

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
        <b>💡 [베트맨 프로토 승부식 맞춤조건구매 시뮬레이터]</b><br>
        경기별로 [승 / 무 / 패]를 복수 선택하면 가능한 모든 단일 조합을 직교 곱(Cartesian Product)으로 전개하여 배당률, 환급금, 비과세 여부를 자동 계산합니다.<br>
        <span style='font-size:12px; color:#64748b;'>※ 본 기능은 계산 및 시뮬레이션 전용이며 실제 베트맨 사이트 구매 자동화는 지원하지 않습니다.</span>
    </div>
    """, unsafe_allow_html=True)

    # 상단 단계 전환 라디오
    step_col1, step_col2 = st.columns([2.5, 1.5])
    with step_col1:
        step_choice = st.radio(
            "진행 단계",
            ["STEP 01: 맞춤 조건 설정 (경기 및 승무패 선택)", "STEP 02: 조합 결과 및 시뮬레이션"],
            index=0 if st.session_state["custom_step"] == 1 else 1,
            horizontal=True,
            key="step_radio_selector"
        )
        st.session_state["custom_step"] = 1 if "STEP 01" in step_choice else 2

    with step_col2:
        # 프리셋 및 빠른 초기화 버튼
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            if st.button("🛡️ Banker+Wheel 프리셋", help="AI 고승률 축 2경기 + 변수 1경기(승무패 3장 전수) 자동 체크"):
                # Banker 2경기 + Wheel 1경기 추출
                rec_candidates = [m for m in predictions if m.get("sport") == "SC" and m.get("betman_odds", {}).get("home_win", 0) > 1.0]
                rec_candidates.sort(key=lambda x: max(x.get("blended_probabilities", {}).values()), reverse=True)
                if len(rec_candidates) >= 3:
                    b1 = rec_candidates[0]
                    b2 = rec_candidates[1]
                    wm = rec_candidates[2]
                    
                    b1_sel = b1["evaluation"]["best_selection"]
                    b2_sel = b2["evaluation"]["best_selection"]
                    
                    st.session_state["selections"] = {
                        str(b1["match_seq"]): {
                            "home": (b1_sel == "home_win"),
                            "draw": (b1_sel == "draw"),
                            "away": (b1_sel == "away_win")
                        },
                        str(b2["match_seq"]): {
                            "home": (b2_sel == "home_win"),
                            "draw": (b2_sel == "draw"),
                            "away": (b2_sel == "away_win")
                        },
                        str(wm["match_seq"]): {
                            "home": True,
                            "draw": True if wm["betman_odds"].get("draw", 0) > 0 else False,
                            "away": True
                        }
                    }
                    st.session_state["editor_version"] += 1
                    st.rerun()

        with p_c2:
            if st.button("🗑️ 선택 전체 초기화"):
                st.session_state["selections"] = {}
                st.session_state["editor_version"] += 1
                st.rerun()

    st.divider()

    # -------------------------------------------------------------
    # STEP 01: 맞춤 조건 설정 화면
    # -------------------------------------------------------------
    if st.session_state["custom_step"] == 1:
        st.markdown("### 📋 STEP 01. 대상 경기 검색 및 승/무/패 다중 선택")
        st.caption("배당 버튼을 눌러 선택하세요. 한 경기에서 여러 개를 고르면 가능한 모든 조합이 자동으로 계산됩니다.")
        _chip = "display:inline-block;margin:0 14px 6px 0;font-size:13px;color:#cbd5e1"
        st.markdown(
            "<div style='margin:-4px 0 8px 0'>"
            f"<span style='{_chip}'>⭐ AI 확률 1순위 <span style='color:#94a3b8'>(AI 없으면 배당 1순위)</span></span>"
            f"<span style='{_chip}'>💎 AI 기준 기대가치 플러스 <span style='color:#f59e0b'>(검증 중)</span></span>"
            f"<span style='{_chip}'>⚠️ AI 예상이 배당과 다름</span>"
            f"<span style='{_chip}'>✅ ❌ 끝난 경기의 적중 여부</span>"
            "</div>",
            unsafe_allow_html=True)

        # 필터 컨트롤
        current_gm_ts = detect_current_gm_ts(predictions)
        with st.container(border=True):
            # 회차 선택: 현재 판매 회차 + 결과가 있는 과거 회차 (과거는 조회 전용)
            past_rounds = sorted({ts for ts, _ in load_results_index().keys() if ts and ts != str(current_gm_ts)}, reverse=True)
            round_labels = {"ALL": f"전체 회차 (현재 + 지난 {len(past_rounds)}개)"}
            if current_gm_ts:
                round_labels[str(current_gm_ts)] = f"{current_gm_ts}회차 (현재 판매)"
            round_labels.update({ts: f"{ts}회차 (종료·조회 전용)" for ts in past_rounds})
            r1, _ = st.columns([1.5, 3.6])
            with r1:
                sel_round = st.selectbox("🗓️ 회차", list(round_labels), index=1 if current_gm_ts else 0,
                                         format_func=lambda x: round_labels[x], key="step1_round")
            is_all = sel_round == "ALL"
            is_past = (not is_all) and sel_round != str(current_gm_ts)
            if is_all:
                round_matches = list(predictions) + [m for ts in past_rounds for m in load_past_round_matches(ts)]
            else:
                round_matches = load_past_round_matches(sel_round) if is_past else predictions

            f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.2, 1.2])
            with f1:
                search_kw = st.text_input("🔎 팀 대진 검색", placeholder="예: 아스널, 뮌헨, 인천", key="step1_search")
            with f2:
                avail_leagues = sorted(list(set(m.get("league", "기타") for m in round_matches)))
                sel_league = st.selectbox("리그 필터", ["전체 리그"] + avail_leagues, key="step1_league")
            with f3:
                avail_cats = sorted(list(set(category_group(m) for m in round_matches)))
                sel_cat = st.selectbox("게임 유형 필터", ["전체 유형"] + avail_cats, key="step1_cat")
            with f4:
                avail_sports = sorted(list(set(m.get("sport", "기타") for m in round_matches)))
                sel_sport = st.selectbox("종목 필터", ["전체 종목"] + avail_sports, key="step1_sport")

            sub_c1, sub_c2, sub_c3 = st.columns(3)
            with sub_c1:
                only_open = st.checkbox("⏳ 마감 전 경기만 보기", value=False, key="step1_only_open")
            with sub_c2:
                only_single_opt = st.checkbox("⚡ 한경기(단통) 가능만 보기", value=False, key="step1_only_single")
            with sub_c3:
                only_rec_opt = st.checkbox("💎 AI 기대가치 플러스만 보기", value=False, key="step1_only_rec")

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
            if sel_cat != "전체 유형" and category_group(m) != sel_cat:
                continue
            if sel_sport != "전체 종목" and m.get("sport") != sel_sport:
                continue
            if only_open and m.get("close_time"):
                if float(m["close_time"]) <= now_ts:
                    continue
            if only_single_opt and not m.get("is_single", False):
                continue
            filtered_matches.append(m)

        # 경기 카드 목록: 배당 버튼을 눌러 바로 선택 (가로 스크롤 없이 휴대폰에서도 사용)
        results_idx = load_results_index()
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
            """AI 예상 확률: 축구 승무패만. 끝난 경기는 경기 전 고정 기록으로만 (경기 후 재계산 값은 결과가 섞일 수 있음)"""
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

        lc1, lc2, lc3 = st.columns([1.4, 1.2, 1.4])
        with lc1:
            day_opts = ["전체 날짜"] + sorted({_match_day(m) for m in filtered_matches})
            # 기본값: 아직 결과가 안 나온 경기가 있는 첫 날짜 (바로 고를 수 있는 경기부터 보이도록)
            open_days = sorted({_match_day(m) for m in filtered_matches if _res(m) is None})
            day_idx = day_opts.index(open_days[0]) if (open_days and not is_all) else 0
            sel_day = st.selectbox("📅 경기 날짜", day_opts, index=day_idx, key=f"step1_day_{sel_round}")
        with lc2:
            only_picked = st.checkbox("✔️ 선택한 경기만", value=False, key="step1_only_picked")
        if sel_day != "전체 날짜":
            filtered_matches = [m for m in filtered_matches if _match_day(m) == sel_day]
        if only_rec_opt:
            filtered_matches = [m for m in filtered_matches if _gems(
                m, _ai_probs(m, _res(m), _log(m)),
                available_picks(m))]
        if only_picked:
            filtered_matches = [m for m in filtered_matches if str(m.get("match_seq")) in st.session_state["selections"]]

        # 아직 결과가 없는(선택 가능한) 경기를 먼저, 끝난 경기는 뒤로 (각각 원래 시간 순서 유지)
        def _time_key(m):
            t = re.search(r"(\d{2})\.(\d{2})\.(\d{2}).*?(\d{1,2}):(\d{2})", str(m.get("match_time", "")))
            return "".join(f"{int(x):02d}" for x in t.groups()) if t else ""

        # 선택 가능한(결과 없는) 경기는 시간 순으로 먼저, 끝난 경기는 최근 경기부터
        open_ms = sorted([m for m in filtered_matches if _res(m) is None], key=_time_key)
        done_ms = sorted([m for m in filtered_matches if _res(m) is not None], key=_time_key, reverse=True)
        filtered_matches = open_ms + done_ms

        PAGE_SIZE = 25
        n_pages = max(1, (len(filtered_matches) + PAGE_SIZE - 1) // PAGE_SIZE)
        with lc3:
            page = st.number_input(f"페이지 (총 {n_pages}쪽, {len(filtered_matches)}경기)", min_value=1, max_value=n_pages,
                                   value=1, step=1, key=f"step1_page_{sel_day}_{n_pages}")
        page_matches = filtered_matches[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

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
        if mk_hits:
            g1, g2, g3 = st.columns(3)
            g1.metric("결과 나온 경기 (현재 필터)", f"{len(mk_hits)}경기")
            g2.metric("배당 1순위 적중", f"{sum(mk_hits) / len(mk_hits) * 100:.1f}%")
            if ai_pairs:
                g3.metric(f"AI 1순위 적중 ({len(ai_pairs)}경기)", f"{sum(a for a, _ in ai_pairs) / len(ai_pairs) * 100:.1f}%",
                          f"같은 경기 배당 1순위 {sum(b for _, b in ai_pairs) / len(ai_pairs) * 100:.1f}%", delta_color="off")
            else:
                g3.metric("AI 1순위 적중", "기록 없음", "경기 전 AI 기록은 260126회차부터", delta_color="off")
            st.caption("적중률은 참고용입니다. 배당 1순위만 찍어도 비슷하게 맞으므로 AI는 '같은 경기'의 배당 1순위와 비교하세요. "
                       "AI 실력의 공식 지표는 '성과 검증' 탭의 성적표(Log loss)입니다.")

        if not page_matches:
            st.info("선택하신 필터 조건에 해당하는 발매 경기가 없습니다.")

        def _prob(probs, k):
            return float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0) if probs else 0.0

        def _mark(pick, actual):
            if not pick or not actual:
                return ""
            ok = pick == actual
            return f"<span style='color:{'#10b981' if ok else '#ef4444'}'>{'✅' if ok else '❌'}</span>"

        cap = "<div style='font-size:11px;color:#94a3b8'>{}</div>"

        for m in page_matches:
            seq = str(m.get("match_seq", ""))
            odds = m.get("betman_odds", {})
            labels = pick_labels(m)
            opts = available_picks(m)
            if not opts:
                continue
            log = _log(m)
            res = _res(m)

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
            single = " · ⚡단통" if m.get("is_single") else ""
            actual = RESULT_CODE_TO_KEY.get(str(res.get("game_result"))) if res is not None else None

            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns([3.2, 2.8, 1.0, 1.4, 0.9, 1.2], vertical_alignment="center")
                with c1:
                    # 정보 줄은 한 줄 고정(넘치면 …), 마우스를 올리면 전체 표시
                    tm = re.search(r"\d{2}\.(\d{2}\.\d{2})\s*\(([^)]*)\)\s*(\d{1,2}:\d{2})", str(m.get("match_time", "")))
                    when = f"{tm.group(1)}({tm.group(2)}) {tm.group(3)}" if tm else m.get("match_time", "")
                    cat_lbl = category_display(m)
                    rnd = f"{_ts(m)}회 · " if is_all else ""
                    meta = f"{rnd}#{seq} · {when} · {m.get('league', '')} · {cat_lbl}{single}"
                    nowrap = "white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
                    st.markdown(
                        f"<div title='{meta}' style='font-size:12px;color:#94a3b8;{nowrap}'>{meta}</div>"
                        f"<div title='{m.get('home_team', '')} vs {m.get('away_team', '')}' style='font-size:15px;{nowrap}'>"
                        f"<b>{m.get('home_team', '')}</b> vs <b>{m.get('away_team', '')}</b></div>",
                        unsafe_allow_html=True)
                with c2:
                    picked = st.pills(
                        f"선택 #{seq}", options=opts, format_func=lambda k, d=disp: d[k], selection_mode="multi",
                        default=default, key=f"pick_{_ts(m)}_{m.get('_key', seq)}_{st.session_state['editor_version']}",
                        label_visibility="collapsed", disabled=(res is not None or _past(m)))
                    ev = m.get("evaluation", {})
                    if res is None and ev.get("is_recommended"):
                        st.markdown(f"<div style='font-size:12px;margin-top:-6px'>🔥 가치 픽 {ev.get('best_ev_pct', '')}</div>",
                                    unsafe_allow_html=True)
                with c3:
                    body = f"<b>{labels[mk_k]}</b> {mk_p * 100:.0f}% {_mark(mk_k, actual)}" if mk_k else "-"
                    st.markdown(cap.format("배당 예상") + f"<div style='font-size:14px'>{body}</div>", unsafe_allow_html=True)
                with c4:
                    if ai_k and ai_k == mk_k:
                        delta = (ai_p - _prob(mk_probs, ai_k)) * 100
                        arrow = "▲" if delta >= 0.5 else ("▼" if delta <= -0.5 else "")
                        body = f"<b>{labels[ai_k]}</b> {ai_p * 100:.0f}% <span style='font-size:11px;color:#94a3b8'>{arrow}{abs(delta):.0f}</span>" if arrow \
                            else f"<b>{labels[ai_k]}</b> {ai_p * 100:.0f}%"
                    elif ai_k:
                        body = f"<span style='color:#f59e0b'>⚠️ <b>{labels[ai_k]}</b> {ai_p * 100:.0f}% 다름</span>"
                    elif res is not None and m.get("game_type") == "축구 승무패":
                        body = "<span style='color:#94a3b8;font-size:12px'>경기 전 기록 없음</span>"
                    else:
                        body = "<span style='color:#94a3b8'>-</span>"
                    if ai_k:
                        body += f" {_mark(ai_k, actual)}"
                    for gk, gev in sorted(gems.items(), key=lambda x: -x[1]):
                        body += (f"<div style='font-size:12px;color:#38bdf8'>💎 <b>{labels[gk]}</b> {_prob(ai_probs, gk) * 100:.0f}% · "
                                 f"EV {gev * 100:+.1f}% {_mark(gk, actual)}</div>")
                    st.markdown(cap.format("AI 예상") + f"<div style='font-size:14px'>{body}</div>", unsafe_allow_html=True)
                with c5:
                    score = res.get("score", "") if res is not None else "-"
                    note = " (핸디)" if res is not None and "핸디캡" in str(m.get("category", "")) else ""
                    st.markdown(cap.format("점수" + note) + f"<div style='font-size:16px;font-weight:700'>{score}</div>",
                                unsafe_allow_html=True)
                with c6:
                    if actual:
                        body = f"<div style='font-size:15px'><b>{labels.get(actual, '?')}</b></div>"
                    else:
                        body = "<div style='font-size:13px;color:#94a3b8'>진행 전</div>"
                    st.markdown(cap.format("결과") + body, unsafe_allow_html=True)

            if res is None and not _past(m):
                picked = picked or []
                if picked:
                    st.session_state["selections"][seq] = {SEL_KEY[k]: (k in picked) for k in PICK_KEYS}
                elif seq in st.session_state["selections"]:
                    del st.session_state["selections"][seq]

        # 조합 구성 검증 및 전개 수 계산
        active_selections = {}
        total_combos_count = 1
        has_any_selection = False

        for seq, sel in list(st.session_state["selections"].items()):
            if seq not in pred_map:
                continue
            m = pred_map[seq]
            odds = m.get("betman_odds", {})
            probs = m.get("blended_probabilities", m.get("model_probabilities", {}))
            sel_labels = m.get("selection_labels", {})
            
            picks = []
            if sel.get("home") and odds.get("home_win", 0) > 0:
                picks.append({
                    "selection": "home_win",
                    "selection_name": sel_labels.get("home_win", f"{m['home_team']} 승"),
                    "odds": odds["home_win"],
                    "prob": probs.get("home_win", 0.33)
                })
            if sel.get("draw") and odds.get("draw", 0) > 0:
                picks.append({
                    "selection": "draw",
                    "selection_name": sel_labels.get("draw", "무승부"),
                    "odds": odds["draw"],
                    "prob": probs.get("draw", 0.33)
                })
            if sel.get("away") and odds.get("away_win", 0) > 0:
                picks.append({
                    "selection": "away_win",
                    "selection_name": sel_labels.get("away_win", f"{m['away_team']} 승"),
                    "odds": odds["away_win"],
                    "prob": probs.get("away_win", 0.33)
                })

            if picks:
                has_any_selection = True
                total_combos_count *= len(picks)
                active_selections[seq] = {
                    "match_seq": seq,
                    "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}",
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "league": m.get("league", ""),
                    "game_type": m.get("category_label", m.get("game_type", "승무패")),
                    "is_single": m.get("is_single", False),
                    "picks": picks
                }

        num_matches = len(active_selections)
        if not has_any_selection:
            total_combos_count = 0

        # 하단 상태 요약 및 다음 단계 이동 바
        st.markdown("#### 📊 현재 선택 상태 요약")
        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns([1.5, 1.5, 2.5, 1.5])
        with sum_c1:
            st.metric("선택 경기 수", f"{num_matches} 경기", help="조합 구매 규정상 2~10경기 (단통 허용 경기는 1경기 가능)")
        with sum_c2:
            st.metric("생성 조합 수", f"{total_combos_count:,} 개 조합", help="각 경기별 선택지 개수의 연속 곱 (Cartesian Product)")
        
        # 규정 위반 검사
        is_step1_valid = False
        val_msg = ""
        if num_matches == 0:
            val_msg = "ℹ️ 대상 경기를 선택해 주세요 (조합은 최소 2경기, 단통은 1경기 가능)."
        elif num_matches == 1:
            first_m = list(active_selections.values())[0]
            if first_m["is_single"]:
                is_step1_valid = True
                val_msg = "✅ 한경기(단통) 구매 지정 경기입니다. 조합 결과 보기가 가능합니다."
            else:
                val_msg = "⚠️ 1경기만 선택되었습니다. 해당 경기는 단통 구매 불가 경기이므로 최소 2경기 이상 선택해야 합니다."
        elif num_matches > 10:
            val_msg = f"⚠️ 베트맨 규정상 조합은 최대 10경기까지만 가능합니다 (현재 {num_matches}경기)."
        else:
            # 동일 경기 중복 검사
            keys = [m["match_key"] for m in active_selections.values()]
            if len(keys) != len(set(keys)):
                val_msg = "⚠️ 동일 대상경기(match_key)의 다른 유형(예: 승무패와 핸디캡)은 한 조합에 묶을 수 없습니다."
            else:
                is_step1_valid = True
                val_msg = f"✅ 베트맨 프로토 규정 통과! ({num_matches}경기 선택 / 총 {total_combos_count:,}개 조합)"

        with sum_c3:
            if is_step1_valid:
                st.success(val_msg)
            else:
                st.warning(val_msg)

        with sum_c4:
            if st.button("➡️ STEP 02 결과보기", type="primary", disabled=not is_step1_valid, use_container_width=True):
                st.session_state["custom_step"] = 2
                st.rerun()

    # -------------------------------------------------------------
    # STEP 02: 조합 결과 및 시뮬레이션 화면
    # -------------------------------------------------------------
    else:
        st.markdown("### 🎯 STEP 02. 생성된 조합 결과 & 베팅 시뮬레이션")
        st.caption("선택한 경기들로 전개된 모든 조합의 공식 배당률, AI 결합 승률, 기대가치(EV)를 확인하고 베팅금을 시뮬레이션합니다.")

        # 선택 경기 데이터 직교 전개
        active_selections = {}
        for seq, sel in st.session_state["selections"].items():
            if seq not in pred_map:
                continue
            m = pred_map[seq]
            odds = m.get("betman_odds", {})
            probs = m.get("blended_probabilities", m.get("model_probabilities", {}))
            sel_labels = m.get("selection_labels", {})
            
            picks = []
            if sel.get("home") and odds.get("home_win", 0) > 0:
                picks.append({
                    "selection": "home_win",
                    "selection_name": sel_labels.get("home_win", f"{m['home_team']} 승"),
                    "odds": odds["home_win"],
                    "prob": probs.get("home_win", 0.33)
                })
            if sel.get("draw") and odds.get("draw", 0) > 0:
                picks.append({
                    "selection": "draw",
                    "selection_name": sel_labels.get("draw", "무승부"),
                    "odds": odds["draw"],
                    "prob": probs.get("draw", 0.33)
                })
            if sel.get("away") and odds.get("away_win", 0) > 0:
                picks.append({
                    "selection": "away_win",
                    "selection_name": sel_labels.get("away_win", f"{m['away_team']} 승"),
                    "odds": odds["away_win"],
                    "prob": probs.get("away_win", 0.33)
                })

            if picks:
                active_selections[seq] = {
                    "match_seq": seq,
                    "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}",
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "league": m.get("league", ""),
                    "game_type": m.get("category_label", m.get("game_type", "승무패")),
                    "is_single": m.get("is_single", False),
                    "picks": picks
                }

        combos = expand_combinations(active_selections)

        if not combos:
            st.warning("선택된 조합이 없습니다. STEP 01로 돌아가 경기를 선택해 주세요.")
            if st.button("⬅️ STEP 01로 돌아가기"):
                st.session_state["custom_step"] = 1
                st.rerun()
        else:
            # 1. 일괄 금액 적용 바
            with st.container(border=True):
                st.markdown("##### ⚙️ 개별 투표금액 일괄 적용 & 정렬 설정")
                bc1, bc2, bc3, bc4 = st.columns([1.5, 1, 1.5, 1])
                
                is_single_only = len(active_selections) == 1
                min_stake_unit = 1000 if is_single_only else 100
                unit_label = "한경기 1,000원 단위" if is_single_only else "조합 100원 단위"

                with bc1:
                    batch_in = st.number_input(
                        f"일괄 적용 투표금액 ({unit_label})",
                        min_value=min_stake_unit,
                        max_value=100000,
                        value=max(min_stake_unit, st.session_state["batch_stake"]),
                        step=min_stake_unit,
                        key="batch_stake_input"
                    )
                with bc2:
                    st.write("")
                    st.write("")
                    if st.button("일괄 적용", use_container_width=True):
                        st.session_state["batch_stake"] = batch_in
                        for c in combos:
                            st.session_state[f"stake_{c['combo_index']}"] = batch_in
                        st.rerun()

                with bc3:
                    sort_option = st.selectbox(
                        "조합 정렬 기준",
                        ["기본 전개순", "예상 배당률 높은순", "예상 배당률 낮은순", "AI 적중확률 높은순", "기대가치(EV) 높은순"]
                    )
                with bc4:
                    st.write("")
                    st.write("")
                    if st.button("⬅️ 조건 재설정", use_container_width=True):
                        st.session_state["custom_step"] = 1
                        st.rerun()

            # 정렬 적용
            if sort_option == "예상 배당률 높은순":
                combos.sort(key=lambda x: x["combined_odds"], reverse=True)
            elif sort_option == "예상 배당률 낮은순":
                combos.sort(key=lambda x: x["combined_odds"])
            elif sort_option == "AI 적중확률 높은순":
                combos.sort(key=lambda x: x["prob_product"], reverse=True)
            elif sort_option == "기대가치(EV) 높은순":
                combos.sort(key=lambda x: x["ev"], reverse=True)

            # 2. 조합 테이블 데이터 구축
            combo_rows = []
            selected_combos_for_slip = []
            total_stake_sum = 0
            total_expected_return = 0.0
            prob_sum = 0.0
            payouts = []

            for c in combos:
                c_idx = c["combo_index"]
                
                # 세션 상태에서 개별 체크 및 베팅금 가져오기
                check_key = f"combo_chk_{c_idx}"
                if check_key not in st.session_state:
                    st.session_state[check_key] = True
                is_selected = st.session_state[check_key]

                stake_key = f"stake_{c_idx}"
                if stake_key not in st.session_state:
                    st.session_state[stake_key] = st.session_state["batch_stake"]
                c_stake = st.session_state[stake_key]

                # 공식 배당률 및 환급금 계산
                c_odds = c["combined_odds"]
                c_payout = calculate_payout(c_odds, c_stake)
                tax_free = is_tax_free(c_odds, c_payout)

                desc_parts = []
                for p in c["picks"]:
                    desc_parts.append(f"#{p['match_seq']} {p['home_team']} [{p['selection_name']}] ({p['odds']}배)")
                combo_desc = " × ".join(desc_parts)

                if is_selected:
                    total_stake_sum += c_stake
                    total_expected_return += (c_payout * c["prob_product"])
                    prob_sum += c["prob_product"]
                    payouts.append(c_payout)
                    selected_combos_for_slip.append({
                        "combo": c,
                        "desc": combo_desc,
                        "stake": c_stake,
                        "payout": c_payout,
                        "odds": c_odds
                    })

                combo_rows.append({
                    "선택": is_selected,
                    "번호": c_idx,
                    "경기 및 예상내역": combo_desc,
                    "배당률": f"{c_odds:.1f}배" if len(c["picks"]) > 1 else f"{c_odds:.2f}배",
                    "투표금액(원)": c_stake,
                    "예상적중금(원)": f"{c_payout:,}원",
                    "AI승률": c["prob_pct"],
                    "기대가치": c["ev_pct"],
                    "비과세": "✅ 비과세" if tax_free else "⚠️ 과세"
                })

            df_combos = pd.DataFrame(combo_rows)

            st.dataframe(
                df_combos,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "선택": st.column_config.CheckboxColumn("선택", default=True),
                    "번호": st.column_config.NumberColumn("번호", width="small"),
                    "경기 및 예상내역": st.column_config.TextColumn("경기 및 예상내역", width="large"),
                    "배당률": st.column_config.TextColumn("배당률", width="small"),
                    "투표금액(원)": st.column_config.NumberColumn("투표금액(원)", format="%d원"),
                    "예상적중금(원)": st.column_config.TextColumn("예상적중금(원)"),
                    "AI승률": st.column_config.TextColumn("AI 승률", width="small"),
                    "기대가치": st.column_config.TextColumn("기대가치(EV)", width="small"),
                    "비과세": st.column_config.TextColumn("비과세", width="small"),
                }
            )

            # 3. 합계 통계 카드
            total_net_pnl = total_expected_return - total_stake_sum
            min_payout = min(payouts) if payouts else 0
            max_payout = max(payouts) if payouts else 0

            st.markdown("#### 💰 선택 조합 종합 지표 & 기대 손익")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("총 투표금액", f"{total_stake_sum:,} 원", f"선택 {len(selected_combos_for_slip)} / 전체 {len(combos)}조합")
            with m2:
                st.metric("최고 적중 환급금", f"{max_payout:,} 원", f"최저: {min_payout:,}원")
            with m3:
                st.metric("적중 확률 합산", f"{prob_sum*100:.1f}%", "단일 적중 배타적 확률 합")
            with m4:
                pnl_color = "normal" if total_net_pnl >= 0 else "inverse"
                st.metric("전체 기대 손익(EV)", f"{int(total_net_pnl):+,} 원", delta=f"{total_net_pnl / total_stake_sum * 100:+.1f}% ROI" if total_stake_sum > 0 else "0.0%")

            if total_stake_sum > 100000:
                st.warning("⚠️ **[구매 한도 초과 경고]** 총 투표금액이 베트맨 1인 1회차 구매 한도인 **100,000원**을 초과했습니다.")

            st.divider()

            # 4. 켈리 기준 자금 관리 계산기 (A.6에 따라 STEP 02로 이동)
            st.markdown("### 📐 켈리 기준(Kelly Criterion) 자금 배분 시뮬레이터")
            st.caption("전체 기대가치(EV)와 승률에 맞춘 수학적 권장 총 베팅액을 계산합니다 (변동성 방지를 위해 1/4~1/2 켈리 권장).")

            kc1, kc2, kc3 = st.columns([1.5, 1.5, 1.2])
            with kc1:
                bankroll = st.number_input(
                    "💼 총 보유 시드머니 (Bankroll, 원)",
                    min_value=10000,
                    max_value=100000000,
                    value=st.session_state.get("user_bankroll", 200000),
                    step=10000,
                    key="custom_bankroll_input"
                )
                st.session_state["user_bankroll"] = bankroll
            with kc2:
                kelly_mode = st.selectbox(
                    "🛡️ 켈리 운용 전략",
                    [
                        "1/4 켈리 (0.25x) - 안정 지향 / MDD 최소화 (권장 ⭐)",
                        "1/2 켈리 (0.50x) - 밸런스 베팅 (허용 상한선)",
                        "1/8 켈리 (0.125x) - 보수적 소액 테스트"
                    ]
                )
                k_mult = 0.25
                if "1/2" in kelly_mode:
                    k_mult = 0.50
                elif "1/8" in kelly_mode:
                    k_mult = 0.125
            with kc3:
                # 가중 평균 배당률 및 가중 EV 계산
                avg_odds = (sum(c["combined_odds"] for c in combos) / len(combos)) if combos else 2.0
                raw_k = 0.0
                if avg_odds > 1.0 and total_net_pnl > 0 and prob_sum > 0:
                    b_val = avg_odds - 1.0
                    q_val = max(0.0, 1.0 - prob_sum)
                    raw_k = max(0.0, (prob_sum * b_val - q_val) / b_val)
                app_k = min(0.05, raw_k * k_mult)
                k_stake = int(bankroll * app_k)
                k_stake_unit = max(min_stake_unit, (k_stake // min_stake_unit) * min_stake_unit)
                k_stake_unit = min(100000, k_stake_unit)

                st.metric("권장 총 베팅액", f"{k_stake_unit:,} 원", f"시드의 {app_k*100:.2f}%")

            if k_stake_unit > 0 and len(selected_combos_for_slip) > 0:
                per_combo_k = max(min_stake_unit, (k_stake_unit // len(selected_combos_for_slip) // min_stake_unit) * min_stake_unit)
                if st.button(f"👉 켈리 권장 금액 조합별 균등 적용 ({per_combo_k:,}원씩)"):
                    st.session_state["batch_stake"] = per_combo_k
                    for c in combos:
                        st.session_state[f"stake_{c['combo_index']}"] = per_combo_k
                    st.rerun()

            st.divider()

            # 5. 베트맨 프로토 마킹 슬립 복사
            st.markdown("### 📋 선택 조합 베팅 슬립 원클릭 복사")
            slip_lines = [
                "📋 [베트맨 프로토 승부식 맞춤조건구매 슬립]",
                f"📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"📊 전체 {len(combos)}개 중 {len(selected_combos_for_slip)}개 조합 선택",
                "────────────────────────────────────────"
            ]
            for item in selected_combos_for_slip:
                c = item["combo"]
                slip_lines.append(f"🎫 [조합 #{c['combo_index']}] {item['desc']}")
                slip_lines.append(f"   ↳ 배당: {item['odds']}배 | 승률: {c['prob_pct']} | EV: {c['ev_pct']}")
                slip_lines.append(f"   ↳ 베팅금: {item['stake']:,}원 ➔ 예상 환급금: {item['payout']:,}원")
                slip_lines.append("")
            slip_lines.append("────────────────────────────────────────")
            slip_lines.append(f"💰 총 투표금액: {total_stake_sum:,}원 (예상 순익: {int(total_net_pnl):+,}원)")
            slip_lines.append("────────────────────────────────────────")

            st.code("\n".join(slip_lines), language="text")


# ==============================================================================
# TAB 2: 🔍 ② AI 경기 분석 (조회 전용)
# ==============================================================================
with tab2:
    if not predictions:
        st.warning("⚠️ 아직 분석된 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # 현재 실전 AI의 검증 상태 (REVIEW R3 원칙: 시장 대비 성적을 항상 투명하게 표시)
        status = load_model_status()
        if status:
            lines = []
            for label, s in status["evals"]:
                ci = s.get("ll_diff_ci", [None, None])
                lines.append(f"- **{label}** ({s['samples']:,}경기): AI `{s['model_ll']:.4f}` vs 시장 `{s['market_ll']:.4f}` "
                             f"(차이 `{s['ll_diff']:+.4f}`, 95% CI `[{ci[0]:+.4f}, {ci[1]:+.4f}]`)")
            st.info(
                f"🧠 **현재 확률 기준: {status['name']}** (축구 일반 승무패에 적용, 나머지 유형은 배당 기준)\n\n"
                + "\n".join(lines) + "\n\n"
                "과거 시험에서는 시장(배당)보다 조금 더 정확했지만 차이가 작아 **검증 중**입니다. "
                "Log loss는 낮을수록 정확하고, 차이가 음수면 AI 우세입니다. 실전 성적은 ③ 성과 검증 탭의 공식 성적표에서 확인하세요. "
                "모든 등급(S/A/B)은 **참고용**입니다."
            )
        else:
            st.warning("⚠️ 검증을 통과한 AI 모델이 없어 **배당(시장) 확률 기준**으로 표시됩니다. 모든 등급은 참고용입니다.")

        # 1. 상단 이번 회차 KPI 카드
        total_matches = len(predictions)
        rec_matches = [m for m in predictions if m.get("evaluation", {}).get("is_recommended", False)]
        rec_count = len(rec_matches)
        single_count = sum(1 for m in predictions if m.get("is_single", False))
        highest_ev_match = max(predictions, key=lambda x: x.get("evaluation", {}).get("best_ev", -1.0))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("총 발매 경기", f"{total_matches} 개", f"한경기(단통) {single_count}개")
        with c2:
            st.metric("추천 가치 픽 (EV > 0)", f"{rec_count} 개", delta=f"{rec_count/total_matches*100:.0f}% 비율")
        with c3:
            st.metric("최고 기대가치", highest_ev_match["evaluation"]["best_ev_pct"], f"{highest_ev_match['home_team']} vs {highest_ev_match['away_team']}")
        with c4:
            st.metric("맞춤구매 담긴 경기", f"{len(st.session_state['selections'])} 개", "Tab 1 시뮬레이터 연동")

        st.divider()

        # 2. 🔍 검색 및 다차원 필터 바
        with st.container(border=True):
            st.markdown("##### 🔍 경기 검색 및 필터")
            q_col1, q_col2, q_col3 = st.columns([1.5, 1.2, 1.2])
            with q_col1:
                t_search = st.text_input("🔎 팀 대진 검색", placeholder="예: 바르셀로나, 뮌헨, 인천", key="tab2_search")
            with q_col2:
                t_leagues = sorted(list(set(m["league"] for m in predictions)))
                t_sel_league = st.selectbox("리그 선택", ["전체 리그"] + t_leagues, key="tab2_league")
            with q_col3:
                t_cats = ["전체 보기", "일반 승무패", "핸디캡", "언더오버", "전반 승무패", "전반 핸디캡", "전반 언더오버"]
                t_sel_cat = st.selectbox("게임 유형 선택", t_cats, key="tab2_cat")

            t_sort = st.radio(
                "정렬 기준",
                ["경기 시작 시간순", "기대가치(EV) 높은순", "홈 승률 높은순", "배당률 높은순"],
                horizontal=True,
                key="tab2_sort"
            )

            c_chk1, c_chk2, c_chk3 = st.columns(3)
            with c_chk1:
                t_only_rec = st.checkbox("🔥 추천 가치 픽만 보기", value=False, key="tab2_only_rec")
            with c_chk2:
                t_only_single = st.checkbox("⚡ 한경기(단통) 구매 가능만 보기", value=False, key="tab2_only_single")
            with c_chk3:
                t_only_has_form = st.checkbox("📊 최근 5경기 전적 있는 경기만 보기", value=False, key="tab2_only_form")

        # 필터링 및 정렬 처리
        t_filtered = predictions
        if t_search:
            t_filtered = [m for m in t_filtered if t_search.lower() in m["home_team"].lower() or t_search.lower() in m["away_team"].lower()]
        if t_sel_league != "전체 리그":
            t_filtered = [m for m in t_filtered if m["league"] == t_sel_league]
        if t_sel_cat != "전체 보기":
            t_filtered = [m for m in t_filtered if t_sel_cat in m.get("category", "")]
        if t_only_single:
            t_filtered = [m for m in t_filtered if m.get("is_single", False)]
        if t_only_rec:
            t_filtered = [m for m in t_filtered if m.get("evaluation", {}).get("is_recommended", False)]
        if t_only_has_form:
            t_filtered = [m for m in t_filtered if m.get("home_form", {}).get("has_data") or m.get("away_form", {}).get("has_data")]

        if "기대가치" in t_sort:
            t_filtered.sort(key=lambda x: x.get("evaluation", {}).get("best_ev", -1.0), reverse=True)
        elif "경기 시작 시간" in t_sort:
            t_filtered.sort(key=lambda x: x.get("match_time", ""))
        elif "홈 승률" in t_sort:
            t_filtered.sort(key=lambda x: x.get("blended_probabilities", {}).get("home_win", x.get("model_probabilities", {}).get("home_win", 0.0)), reverse=True)
        elif "배당률" in t_sort:
            t_filtered.sort(key=lambda x: max(x.get("betman_odds", {}).values()) if x.get("betman_odds") else 0.0, reverse=True)

        st.subheader(f"📋 경기 목록 ({len(t_filtered)}경기)")

        # 3. 고성능 데이터 테이블 뷰 & 상세 인스펙터 (A.6 렌더링 성능 최적화)
        t_rows = []
        for m in t_filtered:
            eval_info = m.get("evaluation", {})
            odds = m.get("betman_odds", {})
            t_rows.append({
                "경기번호": str(m.get("match_seq", "")),
                "일시": m.get("match_time", ""),
                "리그": m.get("league", ""),
                "유형": m.get("category_label", m.get("game_type", "승무패")),
                "홈팀": m.get("home_team", ""),
                "원정팀": m.get("away_team", ""),
                "홈배당": float(odds.get("home_win", 0.0)),
                "무배당": float(odds.get("draw", 0.0)),
                "원정배당": float(odds.get("away_win", 0.0)),
                "확률 기준": "AI" if m.get("prob_source") in AI_SOURCES else "배당",
                "EV 최고 선택": eval_info.get("best_selection_name", "-"),
                "기대가치": eval_info.get("best_ev_pct", "-"),
                "등급": "추천" if eval_info.get("is_recommended") else "패스",
                "단통": "⚡" if m.get("is_single") else ""
            })

        df_matches = pd.DataFrame(t_rows)
        st.dataframe(df_matches, use_container_width=True, hide_index=True)

        st.divider()

        # 4. 개별 경기 상세 분석 인스펙터
        st.markdown("### 🔎 경기 세부 통계 & AI 모델 vs 시장 확률 심층 분석")
        match_options = [f"#{m['match_seq']} [{m['league']}] {m['home_team']} vs {m['away_team']} ({m.get('category_label', m.get('game_type', '승무패'))})" for m in t_filtered[:100]]
        
        if match_options:
            selected_match_label = st.selectbox("분석할 경기를 선택하세요:", match_options, key="tab2_match_selector")
            selected_seq = selected_match_label.split(" ")[0].replace("#", "")
            target_m = pred_map.get(selected_seq)

            if target_m:
                eval_info = target_m.get("evaluation", {})
                odds = target_m.get("betman_odds", {})
                probs = target_m.get("model_probabilities", {})
                mkt_probs = target_m.get("market_implied_probs", {})
                blended_probs = target_m.get("blended_probabilities", probs)
                sel_labels = target_m.get("selection_labels", {})

                with st.container(border=True):
                    # 상단 헤더
                    d_c1, d_c2 = st.columns([3, 1])
                    with d_c1:
                        badge_html = get_grade_badge_html(eval_info.get("grade", ""))
                        single_badge = "<span class='badge-single'>⚡ 단통 1경기 가능</span>" if target_m.get("is_single") else ""
                        cat_badge = f"<span class='badge-cat'>{target_m.get('category_label', target_m.get('game_type', '승무패'))}</span>"
                        st.markdown(
                            f"<div class='notranslate' style='font-size: 1.2rem; font-weight: 700; margin-bottom: 6px;'>"
                            f"[{target_m['league']}] {target_m['home_team']} 🆚 {target_m['away_team']} &nbsp;&nbsp; {badge_html} {single_badge} {cat_badge}"
                            f"</div>"
                            f"<div class='match-meta-bar notranslate'>"
                            f"📅 개최일시: {target_m.get('match_time')} &nbsp;|&nbsp; 🔢 베트맨 경기번호: #{selected_seq}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with d_c2:
                        st.markdown(f"**EV 최고 선택:** <span style='color:#10b981; font-weight:800;'>{eval_info.get('best_selection_name')}</span>", unsafe_allow_html=True)
                        st.markdown(f"기대가치: **`{eval_info.get('best_ev_pct')}`**")
                        if st.button("🛒 맞춤조건구매에 담기", use_container_width=True):
                            b_sel = eval_info.get("best_selection", "home_win")
                            st.session_state["selections"][selected_seq] = {
                                "home": (b_sel == "home_win"),
                                "draw": (b_sel == "draw"),
                                "away": (b_sel == "away_win")
                            }
                            st.session_state["editor_version"] += 1
                            st.success(f"#{selected_seq} 경기가 맞춤조건구매에 담겼습니다!")

                    st.divider()

                    # 확률 비교: 배당(마진 제거) 확률 vs AI 확률, EV는 실전 확률 기준
                    is_ai_src = target_m.get("prob_source") in AI_SOURCES
                    src_name = {"betman_corrector": "베트맨 학습 보정 AI", "market_corrector": "유럽 학습 보정 AI"}.get(
                        target_m.get("prob_source"), "배당 기준 (AI 미적용 유형)")
                    st.markdown(f"##### 📊 배당 확률 vs AI 확률 & 기대가치(EV) — 확률 기준: {src_name}")
                    icons = {"home_win": "🏠", "draw": "🤝", "away_win": "✈️"}
                    defaults = {"home_win": "홈 승", "draw": "무승부", "away_win": "원정 승"}
                    p_cols = st.columns(3)
                    for col, k in zip(p_cols, ("home_win", "draw", "away_win")):
                        with col:
                            o = float(odds.get(k, 0.0) or 0.0)
                            if o <= 1.0:
                                st.caption("해당 선택지 없음 (2지선다 게임)")
                                continue
                            mk = float(mkt_probs.get(k.replace("_win", ""), 0.0) or 0.0)
                            fin = float(blended_probs.get(k, mk) or mk) if is_ai_src else mk
                            ev = (fin * o - 1) * 100
                            st.markdown(f"**{icons[k]} {sel_labels.get(k, defaults[k])} ({o}배)**")
                            if is_ai_src:
                                st.caption(f"• 배당 확률: **{mk*100:.1f}%**\n\n• AI 확률: **{fin*100:.1f}%** ({(fin - mk)*100:+.1f}%p)")
                            else:
                                st.caption(f"• 배당 확률: **{mk*100:.1f}%**")
                            st.markdown(f"• EV: **{ev:+.1f}%**")
                            st.progress(max(0.0, min(1.0, fin)))

                    # 스코어 및 폼 데이터
                    if target_m.get("top_scores"):
                        st.divider()
                        score_text = " | ".join([f"{s['score']} ({s['prob']}%)" for s in target_m.get("top_scores", [])])
                        st.caption(f"🎯 **예상 스코어 TOP 3:** {score_text}")

                    home_form = target_m.get("home_form", {})
                    away_form = target_m.get("away_form", {})
                    h2h = target_m.get("h2h", {})
                    if home_form.get("has_data") or away_form.get("has_data"):
                        st.divider()
                        st.markdown("##### 📈 최근 5경기 폼 & 맞대결(H2H) 전적")
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            st.markdown(f"**🏠 {target_m['home_team']} 최근 5경기:**")
                            st.write(f"• 전적: {home_form.get('record_str', '-')} (평균 득 {home_form.get('avg_scored', 0)} / 실 {home_form.get('avg_conceded', 0)})")
                            if home_form.get("last_matches"):
                                for lm in home_form["last_matches"]:
                                    st.caption(f"• {lm.get('date')} vs {lm.get('opponent')}: {lm.get('score')} ({lm.get('result')})")
                        with f_col2:
                            st.markdown(f"**✈️ {target_m['away_team']} 최근 5경기:**")
                            st.write(f"• 전적: {away_form.get('record_str', '-')} (평균 득 {away_form.get('avg_scored', 0)} / 실 {away_form.get('avg_conceded', 0)})")
                            if away_form.get("last_matches"):
                                for lm in away_form["last_matches"]:
                                    st.caption(f"• {lm.get('date')} vs {lm.get('opponent')}: {lm.get('score')} ({lm.get('result')})")


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
        k1.metric("채점 경기", f"{card['graded_matches']:,}경기", f"회차 {len(card.get('rounds', []))}개")
        k2.metric("Log loss (AI vs 시장)", f"{card['model_ll']:.4f}", f"시장 {card['market_ll']:.4f}", delta_color="off")
        ci = card.get("ll_diff_ci", [0, 0])
        k3.metric("차이 (음수 = AI 우세)", f"{card['ll_diff']:+.4f}", f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]", delta_color="off")
        k4.metric("AI 1순위 적중률", f"{card.get('hit_rate_top_pick', 0):.1f}%",
                  f"EV>2% {card.get('value_bets', 0)}건 ROI {card.get('roi_pct', 0):+.1f}%", delta_color="off")
        st.markdown(f"**판정:** {card.get('verdict', '-')}")
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
            st.dataframe(pd.DataFrame(rows_bt), use_container_width=True, hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # B. 구 모델 기록 (참고) — performance_stats.json
    # ------------------------------------------------------------------
    col_t_title, col_t_btn = st.columns([3, 1.2])
    with col_t_title:
        st.subheader("🗂️ 구 모델(Dixon-Coles) 추천 채점 기록 — 참고용")
        st.warning("이 섹션은 **현재 실전 AI가 아닌 예전 Dixon-Coles 모델**이 과거 회차에 낸 추천을 다시 계산해 채점한 기록입니다. "
                   "모델이 채점 경기 이후의 데이터까지 학습했을 수 있어(미래 정보 혼입 가능) **실력 지표로 쓰면 안 됩니다.** "
                   "실력 판단은 위의 공식 성적표를 보세요.")
    with col_t_btn:
        st.write("")
        if st.button("🔄 구 모델 재채점", key="re_evaluate_btn_tab3", use_container_width=True):
            with st.spinner("베트맨 최근 경기 결과 전수 채점 중..."):
                try:
                    from tracker.result_evaluator import evaluate_betman_results
                    evaluate_betman_results()
                    st.cache_data.clear()
                    st.success("채점 완료! 최신 경기 결과와 적중률이 갱신되었습니다.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"채점 오류: {ex}")

    if not stats:
        st.info("💡 아직 채점된 통계 데이터가 없습니다. 상단의 '실시간 결과 재채점' 버튼을 눌러주세요.")
    else:
        # 1. 상단 통일 KPI 카드 (performance_stats.json 기반)
        sc1, sc2, sc3, sc4 = st.columns(4)
        roi_ci_l = stats.get("roi_ci_low", 0.0)
        roi_ci_h = stats.get("roi_ci_high", 0.0)
        with sc1:
            st.metric("종료 분석 경기", f"{stats.get('sample_matches_analyzed', 0):,} 건", f"{stats.get('total_bets_placed', 0):,}건 추천")
        with sc2:
            st.metric("전체 적중률", f"{stats.get('win_rate_pct', 0.0):.1f}%", f"{stats.get('won_bets', 0)}/{stats.get('total_bets_placed', 0)} 적중")
        with sc3:
            st.metric("누적 수익률 (고정 1만원)", f"{stats.get('roi_pct', 0.0):+.2f}%", f"95% CI: [{roi_ci_l:+.1f}%, {roi_ci_h:+.1f}%]")
        with sc4:
            st.metric("실전 순익 (1만원 고정)", f"{stats.get('net_profit_krw', 0):+,} 원", "과장 없는 투명 공개")

        st.divider()

        # 2. 자금 관리 방식 비교 & 마감 배당 대비 성과 (CLV)
        st.markdown("### ⚖️ 자금 관리 방식 비교 & 마감 배당 대비 성과 (CLV)")
        stk = stats.get("staking_comparison", {})
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.metric("고정 베팅 (1만 원)", f"{stk.get('flat_roi_pct', stats.get('roi_pct', 0)):+.2f}%", f"순익: {stk.get('flat_net_profit_krw', 0):+,}원")
            st.caption("베트맨 1회차 1경기당 고정 10,000원 베팅")
        with c_col2:
            st.metric("1/4 켈리 동적 베팅", f"{stk.get('kelly_roi_pct', 0):+.2f}%", f"순익: {stk.get('kelly_net_profit_krw', 0):+,}원")
            st.caption(f"초기 100만 원 ➔ 최종: {stk.get('kelly_final_bankroll_krw', 0):,}원 (최대 2% 상한)")
        with c_col3:
            clv_cnt = stats.get("clv_sample_count", 0)
            clv_avg = stats.get("avg_clv_pct", 0.0)
            st.metric("CLV (마감 대비 우위)", f"{clv_avg:+.2f}%", f"스냅샷 표본: {clv_cnt}건")
            st.caption("예측 시점 배당 / 마감 배당 비율 (장기 우위 지표)")

        st.divider()

        # 3. 등급별 세부 성과 분석
        st.markdown("### 🏆 등급별(Grade) 적중률 및 수익률 비교")
        gb = stats.get("grade_breakdown", {})
        g_cols = st.columns(3)
        with g_cols[0]:
            s_data = gb.get("S", {})
            st.markdown("#### 👑 GRADE S (참고용 - 고승률)")
            st.write(f"• 베팅: **{s_data.get('bets', 0)}건** 중 **{s_data.get('wins', 0)}건 적중** ({s_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{s_data.get('returned', 0) - s_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{s_data.get('roi_pct', 0):+.1f}%`**")
            st.caption(f"95% CI: [{s_data.get('roi_ci_low', 0.0):+.1f}%, {s_data.get('roi_ci_high', 0.0):+.1f}%]")
            
        with g_cols[1]:
            a_data = gb.get("A", {})
            st.markdown("#### 🟢 GRADE A (참고용 - 밸런스)")
            st.write(f"• 베팅: **{a_data.get('bets', 0)}건** 중 **{a_data.get('wins', 0)}건 적중** ({a_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{a_data.get('returned', 0) - a_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{a_data.get('roi_pct', 0):+.1f}%`**")
            st.caption(f"95% CI: [{a_data.get('roi_ci_low', 0.0):+.1f}%, {a_data.get('roi_ci_high', 0.0):+.1f}%]")

        with g_cols[2]:
            b_data = gb.get("B", {})
            st.markdown("#### 🔵 GRADE B (참고용 - 고배당)")
            st.write(f"• 베팅: **{b_data.get('bets', 0)}건** 중 **{b_data.get('wins', 0)}건 적중** ({b_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{b_data.get('returned', 0) - b_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{b_data.get('roi_pct', 0):+.1f}%`**")
            st.caption(f"95% CI: [{b_data.get('roi_ci_low', 0.0):+.1f}%, {b_data.get('roi_ci_high', 0.0):+.1f}%]")

        st.info(f"💡 구 모델 누적 ROI {stats.get('roi_pct', 0.0):+.2f}% (95% CI [{stats.get('roi_ci_low', 0.0):+.1f}%, {stats.get('roi_ci_high', 0.0):+.1f}%]). "
                "베트맨 환급률이 약 85%라 단일 경기 마진이 약 15%이고, 조합은 마진이 곱으로 누적됩니다(2폴더 약 28%, 3폴더 약 39%). "
                "단일 경기에서 우위가 검증되기 전까지 조합은 시뮬레이션 용도로만 쓰세요.")

        st.divider()

        # 4. 회차별 성과 비교 테이블
        rb = stats.get("round_breakdown", {})
        if rb:
            st.markdown("### 📅 베트맨 최근 회차별 실전 성과")
            r_rows = []
            for r_num, r_data in sorted(rb.items(), key=lambda x: str(x[0]), reverse=True):
                r_rows.append({
                    "회차": f"{r_num}회차",
                    "AI 추천 경기": f"{r_data.get('bets', 0)}건",
                    "적중 건수": f"{r_data.get('wins', 0)}건",
                    "적중률": f"{r_data.get('win_rate_pct', 0)}%",
                    "가상 순익": f"{r_data.get('net_profit', 0):+,}원",
                    "수익률 (ROI)": f"{r_data.get('roi_pct', 0):+.1f}%",
                    "95% CI": f"[{r_data.get('roi_ci_low', 0.0):+.1f}%, {r_data.get('roi_ci_high', 0.0):+.1f}%]"
                })
            r_df = pd.DataFrame(r_rows)
            st.dataframe(r_df, use_container_width=True, hide_index=True)

        st.divider()

        # 5. 지난 경기 결과 1:1 대조 목록
        recent_picks = stats.get("recent_evaluated_picks", [])
        if recent_picks:
            st.markdown("### 📋 채점 경기 개별 대조 (1:1 매칭)")
            
            with st.container(border=True):
                rf1, rf2, rf3, rf4 = st.columns([1.5, 1.5, 1.5, 2])
                with rf1:
                    avail_rounds = sorted(list(set(str(p.get("round", "")) for p in recent_picks)), reverse=True)
                    round_filter = st.selectbox("회차 선택", ["전체 회차"] + [f"{r}회차" for r in avail_rounds], key="eval_round_filter")
                with rf2:
                    hit_filter = st.selectbox("적중 여부", ["전체 보기", "✅ 적중한 경기만", "❌ 미적중 경기만"], key="eval_hit_filter")
                with rf3:
                    grade_filter = st.selectbox("AI 등급", ["전체 등급", "GRADE S (참고용 - 고승률)", "GRADE A (참고용 - 밸런스)", "GRADE B (참고용 - 고배당)"], key="eval_grade_filter")
                with rf4:
                    eval_search = st.text_input("팀 검색", placeholder="예: 브리즈번, 플라멩구", key="eval_team_search")

            f_picks = recent_picks
            if round_filter != "전체 회차":
                t_r = round_filter.replace("회차", "")
                f_picks = [p for p in f_picks if str(p.get("round")) == t_r]
            if hit_filter == "✅ 적중한 경기만":
                f_picks = [p for p in f_picks if p.get("is_hit")]
            elif hit_filter == "❌ 미적중 경기만":
                f_picks = [p for p in f_picks if not p.get("is_hit")]
            if grade_filter != "전체 등급":
                g_tag = "S (" if "S" in grade_filter else ("A (" if "A" in grade_filter else "B (")
                f_picks = [p for p in f_picks if g_tag in p.get("grade", "")]
            if eval_search:
                q = eval_search.lower()
                f_picks = [p for p in f_picks if q in p.get("match", "").lower() or q in p.get("home_team", "").lower() or q in p.get("away_team", "").lower()]

            st.caption(f"총 {len(f_picks)}건 대조")
            
            rec_df = pd.DataFrame(f_picks)
            cols_order = ["round", "date", "league", "match", "game_type", "score", "actual_result", "pick", "grade", "ai_prob", "odds", "is_hit", "net_pnl"]
            existing_cols = [c for c in cols_order if c in rec_df.columns]
            rec_df = rec_df[existing_cols]
            rename_map = {
                "round": "회차",
                "date": "경기일자",
                "league": "리그",
                "match": "경기 대진",
                "game_type": "유형",
                "score": "실제 스코어",
                "actual_result": "실제 결과",
                "pick": "AI 추천 픽",
                "grade": "등급",
                "ai_prob": "AI 승률(%)",
                "odds": "배당률",
                "is_hit": "적중 여부",
                "net_pnl": "가상 손익"
            }
            rec_df = rec_df.rename(columns=rename_map)
            if "적중 여부" in rec_df.columns:
                rec_df["적중 여부"] = rec_df["적중 여부"].apply(lambda x: "✅ 적중" if x else "❌ 미적중")
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
