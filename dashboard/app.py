"""
dashboard/app.py
================
Streamlit 기반 스포츠 승부 예측 & 가치 픽(Value Bet) 실시간 웹 대시보드
[Tab 1: 이번 회차 실시간 경기 분석]
[Tab 2: 🎯 AI 추천 2~3폴더 조합 & 베팅 바구니]
[Tab 3: 📊 AI 누적 적중률 & 모의 베팅 수익률(ROI)]
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가 (dashboard 하위 폴더에서 실행 시 models 패키지 인식)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Streamlit 장시간 세션 유지 시 models/tracker 모듈 메모리 캐시 강제 무효화 (핫 리로딩 보장)
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("models") or mod_name.startswith("tracker"):
        del sys.modules[mod_name]

import json
import re
from datetime import datetime
import streamlit as st
import pandas as pd

from models.combo_builder import calculate_combined_odds, calculate_payout
from models.parlay_builder import (
    build_recommended_parlays,
    build_banker_wheel_combinations,
    calculate_parlay_metrics,
    get_pick_item
)

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="스포츠 승부예측 & AI 가치 조합 대시보드",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 커스텀 CSS 스타일링 (다크 테마 & 번역 방지 & 카드 UI)
st.markdown("""
<meta name="google" content="notranslate">
<style>
    .main { background-color: #0e1117; }
    .notranslate { translate: no !important; }
    .badge-s { background-color: #ff4757; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    .badge-a { background-color: #2ed573; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    .badge-b { background-color: #1e90ff; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    .badge-pass { background-color: #57606f; color: white; padding: 4px 10px; border-radius: 6px; font-size: 13px; }
    .badge-single { background-color: #ffa502; color: #1e272e; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 12px; margin-left: 6px; }
    .badge-cat { background-color: #2f3542; color: #70a1ff; padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; margin-left: 6px; border: 1px solid #70a1ff; }
    .match-meta-bar {
        background-color: #161b26;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 13px;
        color: #8b9bb4;
        margin-bottom: 12px;
        display: inline-block;
    }
    .parlay-card {
        background: linear-gradient(135deg, #1a2234 0%, #111827 100%);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .parlay-metric {
        font-size: 1.4rem;
        font-weight: 800;
        color: #60a5fa;
    }
    .cart-summary-box {
        background: #1e293b;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 18px 24px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


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


# 세션 상태: 사용자의 맞춤 조합 바구니 (경기 번호 -> 선택 마킹 'home_win' | 'draw' | 'away_win' | 'best')
if "cart_picks" not in st.session_state:
    st.session_state["cart_picks"] = {}

# 하위 호환성 유지: cart_seqs
st.session_state["cart_seqs"] = list(st.session_state["cart_picks"].keys())

pred_map = {str(m["match_seq"]): m for m in predictions}
current_cart_picks = [
    get_pick_item(pred_map[seq], selection=st.session_state["cart_picks"][seq])
    for seq in st.session_state["cart_picks"]
    if seq in pred_map
]

# -------------------------------------------------------------
# 🛒 베트맨 실시간 투표용지 사이드바 위젯 (사이드바에는 실시간 투표용지만 배치)
# -------------------------------------------------------------
st.sidebar.markdown(f"### 🛒 실시간 투표용지 ({len(current_cart_picks)}폴더)")
if not current_cart_picks:
    st.sidebar.info("💡 경기 카드의 **[승 / 무 / 패]** 배당 버튼을 누르면 투표용지에 실시간으로 담깁니다.")
else:
    s_metrics = calculate_parlay_metrics(current_cart_picks)
    if not s_metrics["valid"]:
        st.sidebar.error(s_metrics["error"])
    else:
        st.sidebar.markdown(
            f"<div class='cart-summary-box' style='padding:12px; margin-bottom:10px;'>"
            f"• <b>선택 폴더:</b> <span style='color:#60a5fa; font-weight:bold;'>{s_metrics['folders']}경기</span><br>"
            f"• <b>합산 배당률:</b> <span style='color:#2ed573; font-size:1.25rem; font-weight:800;'>{s_metrics['combined_odds']}배</span><br>"
            f"• <b>AI 결합 승률:</b> {s_metrics['combined_prob_pct']}<br>"
            f"• <b>조합 기대가치:</b> {s_metrics['combined_ev_pct']}"
            f"</div>",
            unsafe_allow_html=True
        )
        for cp in current_cart_picks:
            del_col1, del_col2 = st.sidebar.columns([4, 1])
            with del_col1:
                st.markdown(f"<span style='font-size:12px;'>#{cp['match_seq']} <b>{cp['selection_name']}</b> ({cp['odds']}배)</span>", unsafe_allow_html=True)
            with del_col2:
                if st.sidebar.button("✕", key=f"side_del_{cp['match_seq']}"):
                    del st.session_state["cart_picks"][cp["match_seq"]]
                    st.rerun()

        st.sidebar.caption("👉 'Tab 2'에서 켈리 자금관리 및 복사 가능")
        if st.sidebar.button("🗑️ 투표용지 비우기", key="side_clear_cart", use_container_width=True):
            st.session_state["cart_picks"] = {}
            st.rerun()

# 타이틀 헤더
st.title("⚽ 스포츠 승부 예측 & 가치 픽 (Value Bet) 대시보드")
st.caption("베트맨(Betman) 프로토 배당률 및 Dixon-Coles 통계 모델 기반 기대가치(EV) 실시간 분석")

# 상단 탭 4개 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 이번 회차 실시간 베트맨 경기",
    "🎯 AI 추천 2~3폴더 조합 & 베팅 바구니",
    "🏁 지난 경기 결과 vs AI 예측 채점 (실전 검증)",
    "📊 AI 누적 적중률 & 수익률(ROI) 통계"
])

# ==============================================================================
# TAB 1: 이번 회차 실시간 베트맨 경기 분석
# ==============================================================================
with tab1:
    if not predictions:
        st.warning("⚠️ 아직 분석된 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # Phase 1 체크포인트 조치 (REVIEW R3): 모델 Log Loss가 시장 기준선보다 열세일 때 투명하게 경고 배너 표시
        b_sum = load_baseline_summary()
        if b_sum:
            b_m_ll = b_sum["betman_model_ll"]
            b_mkt_ll = b_sum["betman_market_ll"]
            t_m_ll = b_sum["test_model_ll"]
            t_mkt_ll = b_sum["test_market_ll"]
            if b_m_ll > b_mkt_ll or t_m_ll > t_mkt_ll:
                st.warning(
                    f"⚠️ **[공지: 모델 검증 안내 — 모든 추천 픽은 '참고용'으로 제공됩니다]**\n\n"
                    f"현재 순수 AI 통계 모델은 검증 데이터에서 **시장 배당 기준선(Market Baseline)보다 Log Loss가 높아 예측 오차가 더 큽니다**.\n\n"
                    f"- **실전 베트맨 검증셋:** AI 모델 `{b_m_ll:.4f}` vs 시장 기준선 `{b_mkt_ll:.4f}` (시장 우위: `+{b_m_ll - b_mkt_ll:.4f}`)\n"
                    f"- **유럽 5대 리그 OOS 검증셋:** AI 모델 `{t_m_ll:.4f}` vs 시장 기준선 `{t_mkt_ll:.4f}` (시장 우위: `+{t_m_ll - t_mkt_ll:.4f}`)\n\n"
                    f"이에 따라 Phase 1 체크포인트 규칙에 의거, 대시보드의 모든 추천 등급(S/A/B)은 무조건적 신뢰가 아닌 **'참고용(Reference Only)'**으로 표시됩니다. "
                    f"현재 샤프 배당(Betfair Exchange)과의 결합 및 모델·시장 블렌딩(Phase 3)을 통한 고도화 작업이 진행 중입니다."
                )

        # 실전 누적 검증 성과 & 95% 신뢰구간 (Step 4.3 대시보드 정직화)
        if stats:
            s_roi = stats.get("roi_pct", 0.0)
            s_ci_low = stats.get("roi_ci_low", 0.0)
            s_ci_high = stats.get("roi_ci_high", 0.0)
            s_clv = stats.get("avg_clv_pct", 0.0)
            s_clv_cnt = stats.get("clv_sample_count", 0)
            stk_comp = stats.get("staking_comparison", {})
            k_roi = stk_comp.get("kelly_roi_pct", 0.0)
            with st.container(border=True):
                st.markdown("##### 📊 실전 베트맨 누적 검증 성과 (과장 없는 투명 공개)")
                hc1, hc2, hc3, hc4 = st.columns(4)
                with hc1:
                    st.metric("실전 누적 ROI (고정 1만원)", f"{s_roi:+.2f}%", f"95% CI: [{s_ci_low:+.1f}%, {s_ci_high:+.1f}%]")
                with hc2:
                    st.metric("1/4 켈리 동적 자금관리 ROI", f"{k_roi:+.2f}%", "최대 2% 상한선 적용")
                with hc3:
                    st.metric("CLV (마감 배당 대비 우위)", f"{s_clv:+.2f}%", f"스냅샷 표본: {s_clv_cnt}건")
                with hc4:
                    st.metric("전체 적중률", f"{stats.get('win_rate_pct', 0.0):.1f}%", f"{stats.get('won_bets', 0)}/{stats.get('total_bets_placed', 0)} 적중")

        # 1. 상단 이번 회차 KPI 카드
        total_matches = len(predictions)
        rec_matches = [m for m in predictions if m["evaluation"]["is_recommended"]]
        rec_count = len(rec_matches)
        single_count = sum(1 for m in predictions if m.get("is_single", False))
        highest_ev_match = max(predictions, key=lambda x: x["evaluation"]["best_ev"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("총 발매 번호", f"{total_matches} 개", f"한경기(단통) {single_count}개")
        with c2:
            st.metric("추천 가치 픽 (EV > 0)", f"{rec_count} 개", delta=f"{rec_count/total_matches*100:.0f}% 비율")
        with c3:
            st.metric("최고 기대가치", highest_ev_match["evaluation"]["best_ev_pct"], f"{highest_ev_match['home_team']} vs {highest_ev_match['away_team']}")
        with c4:
            st.metric("투표용지 담긴 경기", f"{len(current_cart_picks)} 개", "사이드바 및 Tab 2에서 확인")

        st.divider()

        # -------------------------------------------------------------
        # 1. 일자별 빠른 선택 Pills (Image 2 위치 상단)
        # -------------------------------------------------------------
        date_map = {}
        for m in predictions:
            mt = m.get("match_time", "")
            d_match = re.search(r"(\d{2}\.\d{2}\.\d{2})\s*(\([가-힣a-zA-Z]+\))?", mt)
            if d_match:
                d_key = d_match.group(1)
                d_day = d_match.group(2) if d_match.group(2) else ""
                label = f"{d_key} {d_day}".strip()
                if d_key not in date_map:
                    date_map[d_key] = label

        sorted_dates = sorted(date_map.keys())
        today_str = datetime.now().strftime("%y.%m.%d")

        date_options = ["전체 일자"]
        for d in sorted_dates:
            d_label = date_map[d]
            if d == today_str:
                d_label = f"⚡ 오늘 ({d_label})"
            date_options.append(d_label)

        if hasattr(st, "pills"):
            selected_date_pill = st.pills("📅 일자 빠른 필터", options=date_options, default="전체 일자")
        elif hasattr(st, "segmented_control"):
            selected_date_pill = st.segmented_control("📅 일자 빠른 필터", options=date_options, default="전체 일자")
        else:
            selected_date_pill = st.radio("📅 일자 빠른 필터", options=date_options, horizontal=True)

        # -------------------------------------------------------------
        # 2. 🔍 검색 및 베팅 필터 (1번 이미지 기능 ➔ 2번 이미지 위치로 이동)
        # -------------------------------------------------------------
        with st.container(border=True):
            st.markdown("##### 🔍 검색 및 베팅 필터")

            f_col1, f_col2, f_col3 = st.columns([1.5, 1.2, 1.2])
            with f_col1:
                search_query = st.text_input("🔎 팀 이름 검색", placeholder="예: 바르셀로나, 뮌헨, 인천")
            with f_col2:
                leagues = sorted(list(set(m["league"] for m in predictions)))
                selected_league = st.selectbox("리그 선택", ["전체"] + leagues)
            with f_col3:
                categories = ["전체 보기", "일반 승무패", "핸디캡", "언더오버", "전반 승무패", "전반 핸디캡", "전반 언더오버"]
                selected_category = st.selectbox("게임 유형 선택", categories)

            sort_by = st.radio(
                "정렬 기준",
                ["기대가치(EV) 높은순 (AI 추천순)", "경기 시작 시간순", "홈 승률 높은순", "배당률 높은순"],
                horizontal=True
            )

            chk_col1, chk_col2, chk_col3, chk_col4 = st.columns(4)
            with chk_col1:
                only_recommended = st.checkbox("🔥 추천 가치 픽(EV > 0)만 보기", value=True)
            with chk_col2:
                only_single = st.checkbox("⚡ 한경기(단통) 구매 가능만 보기", value=False)
            with chk_col3:
                only_has_form = st.checkbox("📊 최근 5경기 전적 있는 경기만 보기", value=False)
            with chk_col4:
                show_all_details = st.checkbox("📂 과거 5경기 스코어 상세 펼치기", value=False)

        # -------------------------------------------------------------
        # 3. 필터링 및 정렬 처리
        # -------------------------------------------------------------
        filtered = predictions
        if selected_date_pill != "전체 일자":
            target_search = re.search(r"\d{2}\.\d{2}\.\d{2}", selected_date_pill)
            if target_search:
                target_key = target_search.group(0)
                filtered = [m for m in filtered if target_key in m.get("match_time", "")]

        if search_query:
            filtered = [m for m in filtered if search_query.lower() in m["home_team"].lower() or search_query.lower() in m["away_team"].lower()]
        if selected_league != "전체":
            filtered = [m for m in filtered if m["league"] == selected_league]
        if selected_category != "전체 보기":
            filtered = [m for m in filtered if selected_category in m.get("category", "")]
        if only_single:
            filtered = [m for m in filtered if m.get("is_single", False)]
        if only_recommended:
            filtered = [m for m in filtered if m["evaluation"]["is_recommended"]]
        if only_has_form:
            filtered = [m for m in filtered if m.get("home_form", {}).get("has_data") or m.get("away_form", {}).get("has_data")]

        if "기대가치" in sort_by:
            filtered.sort(key=lambda x: x["evaluation"]["best_ev"], reverse=True)
        elif "경기 시작 시간" in sort_by:
            filtered.sort(key=lambda x: x.get("match_time", ""))
        elif "홈 승률" in sort_by:
            filtered.sort(key=lambda x: x["model_probabilities"]["home_win"], reverse=True)
        elif "배당률" in sort_by:
            filtered.sort(key=lambda x: max(x["betman_odds"].values()), reverse=True)

        st.subheader(f"📋 경기 분석 결과 ({len(filtered)}경기)")

        for m in filtered[:60]:  # 상위 60개 렌더링
            eval_info = m["evaluation"]
            odds = m["betman_odds"]
            probs = m.get("model_probabilities", {})
            mkt_probs = m.get("market_implied_probs", {})
            blended_probs = m.get("blended_probabilities", {})
            sel_labels = m.get("selection_labels", {})
            m_seq = str(m["match_seq"])
            
            # 배지 스타일링
            grade_str = eval_info["grade"]
            if "S (" in grade_str:
                badge_html = "<span class='badge-s'>GRADE S (참고용 - 고승률)</span>"
            elif "A (" in grade_str:
                badge_html = "<span class='badge-a'>GRADE A (참고용 - 밸런스)</span>"
            elif "B (" in grade_str:
                badge_html = "<span class='badge-b'>GRADE B (참고용 - 고배당)</span>"
            else:
                badge_html = "<span class='badge-pass'>PASS (가치 낮음)</span>"

            single_badge = "<span class='badge-single'>⚡ 단통 1경기 가능</span>" if m.get("is_single") else ""
            cat_badge = f"<span class='badge-cat'>{m.get('category_label', m.get('game_type', '승무패'))}</span>"

            with st.container(border=True):
                head_col1, head_col2, head_col3 = st.columns([2.8, 1.2, 0.8])
                with head_col1:
                    st.markdown(
                        f"<div class='notranslate' translate='no' style='font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;'>"
                        f"[{m['league']}] {m['home_team']} 🆚 {m['away_team']} &nbsp;&nbsp; {badge_html} {single_badge} {cat_badge}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='match-meta-bar notranslate' translate='no'>"
                        f"📅 <b>개최일시:</b> {m['match_time']} &nbsp;|&nbsp; "
                        f"🏷️ <b>게임유형:</b> {m.get('game_type', '승무패')} &nbsp;|&nbsp; "
                        f"🔢 <b>베트맨 경기번호:</b> #{m_seq}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with head_col2:
                    st.markdown(f"**🎯 AI 추천:**")
                    st.markdown(f"<div class='notranslate' style='color: #2ed573; font-weight: 800; font-size: 1.02rem;'>{eval_info['best_selection_name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"기대가치: **`{eval_info['best_ev_pct']}`**")
                # 현재 슬립에 담긴 픽 확인
                current_pick_sel = st.session_state["cart_picks"].get(m_seq)

                with head_col3:
                    # AI 추천 픽 토글 담기 버튼
                    is_ai_in_cart = (current_pick_sel in ("best", eval_info["best_selection"]))
                    ai_btn_label = "✅ AI추천 담김" if is_ai_in_cart else "⚡ AI추천 담기"
                    ai_btn_type = "primary" if is_ai_in_cart else "secondary"
                    if st.button(ai_btn_label, key=f"cart_ai_btn_{m_seq}", type=ai_btn_type, use_container_width=True):
                        if is_ai_in_cart:
                            del st.session_state["cart_picks"][m_seq]
                        else:
                            st.session_state["cart_picks"][m_seq] = eval_info["best_selection"]
                        st.rerun()

                # 각 선택지별 라벨 및 배당/확률
                label_h = sel_labels.get("home_win", f"{m['home_team']} 승")
                label_d = sel_labels.get("draw", "무승부")
                label_a = sel_labels.get("away_win", f"{m['away_team']} 승")

                # 베트맨 승 / 무 / 패 실시간 마킹 버튼 영역
                col_h, col_d, col_a = st.columns(3)
                with col_h:
                    is_h = (current_pick_sel == "home_win")
                    btn_h_txt = f"{'✅ ' if is_h else ''}🏠 {label_h}  [{odds['home_win']}배]"
                    h_type = "primary" if is_h else "secondary"
                    if st.button(btn_h_txt, key=f"btn_h_{m_seq}", type=h_type, use_container_width=True):
                        if is_h:
                            del st.session_state["cart_picks"][m_seq]
                        else:
                            st.session_state["cart_picks"][m_seq] = "home_win"
                        st.rerun()
                    m_h = probs.get('home_win', 0.0)
                    mkt_h = mkt_probs.get('home', 0.0)
                    b_h = blended_probs.get('home_win', m_h)
                    diff_h = m_h - mkt_h
                    ev_h = ((b_h * odds['home_win']) - 1) * 100 if odds['home_win'] > 1.0 else 0
                    st.caption(f"모델: **{m_h*100:.1f}%** | 시장: **{mkt_h*100:.1f}%** ({diff_h*100:+.1f}%p)")
                    st.caption(f"블렌딩: **{b_h*100:.1f}%** | EV: **{ev_h:+.1f}%**")
                    st.progress(max(0.0, min(1.0, b_h)))

                with col_d:
                    if odds['draw'] > 0:
                        is_d = (current_pick_sel == "draw")
                        btn_d_txt = f"{'✅ ' if is_d else ''}🤝 {label_d}  [{odds['draw']}배]"
                        d_type = "primary" if is_d else "secondary"
                        if st.button(btn_d_txt, key=f"btn_d_{m_seq}", type=d_type, use_container_width=True):
                            if is_d:
                                del st.session_state["cart_picks"][m_seq]
                            else:
                                st.session_state["cart_picks"][m_seq] = "draw"
                            st.rerun()
                        m_d = probs.get('draw', 0.0)
                        mkt_d = mkt_probs.get('draw', 0.0)
                        b_d = blended_probs.get('draw', m_d)
                        diff_d = m_d - mkt_d
                        ev_d = ((b_d * odds['draw']) - 1) * 100 if odds['draw'] > 1.0 else 0
                        st.caption(f"모델: **{m_d*100:.1f}%** | 시장: **{mkt_d*100:.1f}%** ({diff_d*100:+.1f}%p)")
                        st.caption(f"블렌딩: **{b_d*100:.1f}%** | EV: **{ev_d:+.1f}%**")
                        st.progress(max(0.0, min(1.0, b_d)))
                    else:
                        st.caption("무승부 없음 (2지선다 게임)")

                with col_a:
                    is_a = (current_pick_sel == "away_win")
                    btn_a_txt = f"{'✅ ' if is_a else ''}✈️ {label_a}  [{odds['away_win']}배]"
                    a_type = "primary" if is_a else "secondary"
                    if st.button(btn_a_txt, key=f"btn_a_{m_seq}", type=a_type, use_container_width=True):
                        if is_a:
                            del st.session_state["cart_picks"][m_seq]
                        else:
                            st.session_state["cart_picks"][m_seq] = "away_win"
                        st.rerun()
                    m_a = probs.get('away_win', 0.0)
                    mkt_a = mkt_probs.get('away', 0.0)
                    b_a = blended_probs.get('away_win', m_a)
                    diff_a = m_a - mkt_a
                    ev_a = ((b_a * odds['away_win']) - 1) * 100 if odds['away_win'] > 1.0 else 0
                    st.caption(f"모델: **{m_a*100:.1f}%** | 시장: **{mkt_a*100:.1f}%** ({diff_a*100:+.1f}%p)")
                    st.caption(f"블렌딩: **{b_a*100:.1f}%** | EV: **{ev_a:+.1f}%**")
                    st.progress(max(0.0, min(1.0, b_a)))

                # 스코어 및 부가 지표
                if m.get("top_scores"):
                    score_text = " | ".join([f"{s['score']} ({s['prob']}%)" for s in m.get("top_scores", [])])
                    st.caption(f"🎯 **예상 스코어 TOP 3:** {score_text}")

                # 팀별 최근 5경기 폼 및 맞대결(H2H) 카드 표면 직관 노출 (클릭 불필요!)
                home_form = m.get("home_form", {})
                away_form = m.get("away_form", {})
                h2h = m.get("h2h", {})

                has_form = home_form.get("has_data") or away_form.get("has_data")
                
                # 데이터가 실제로 존재하는 경우에만 카드 표면에 깔끔하게 노출 (없는 경기는 여백 낭비 없이 숨김)
                if has_form:
                    h_badges = ""
                    if home_form.get("has_data"):
                        b_list = []
                        for r in home_form.get("form_sequence", []):
                            bg = "#2ed573" if r == "W" else ("#747d8c" if r == "D" else "#ff4757")
                            b_list.append(f"<span style='background:{bg}; color:white; padding:1px 6px; border-radius:3px; font-weight:800; font-size:11px;'>{r}</span>")
                        h_badges = " ".join(b_list) + f" &nbsp;<span style='color:#a0aec0; font-size:12px;'>({home_form['record_str']}, 득{home_form['avg_scored']}/실{home_form['avg_conceded']})</span>"
                    else:
                        h_badges = "<span style='color:#718096; font-size:12px;'>전적 없음</span>"

                    a_badges = ""
                    if away_form.get("has_data"):
                        b_list = []
                        for r in away_form.get("form_sequence", []):
                            bg = "#2ed573" if r == "W" else ("#747d8c" if r == "D" else "#ff4757")
                            b_list.append(f"<span style='background:{bg}; color:white; padding:1px 6px; border-radius:3px; font-weight:800; font-size:11px;'>{r}</span>")
                        a_badges = " ".join(b_list) + f" &nbsp;<span style='color:#a0aec0; font-size:12px;'>({away_form['record_str']}, 득{away_form['avg_scored']}/실{away_form['avg_conceded']})</span>"
                    else:
                        a_badges = "<span style='color:#718096; font-size:12px;'>전적 없음</span>"

                    h2h_html = ""
                    if h2h.get("has_data"):
                        h2h_html = f"<div style='background:#1e293b; border:1px solid #3b82f6; border-radius:6px; padding:3px 10px; font-size:12px; font-weight:700; color:#60a5fa;'>⚔️ 맞대결: {h2h['record_str']} (평균 {h2h['avg_goals']}골)</div>"

                    st.markdown(
                        f"<div class='notranslate' translate='no' style='background:#141b27; border:1px solid #233147; border-radius:8px; padding:8px 14px; margin-top:10px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
                        f"<div><b>🏠 {m['home_team']}:</b> {h_badges}</div>"
                        f"{h2h_html}"
                        f"<div><b>✈️ {m['away_team']}:</b> {a_badges}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # 사이드바에서 '과거 5경기 스코어 상세 펼치기' 체크 시 상세 스코어 표시
                    if show_all_details:
                        det_col1, det_col2 = st.columns(2)
                        with det_col1:
                            if home_form.get("has_data") and home_form.get("last_matches"):
                                st.caption(f"🏠 **{m['home_team']} 최근 5경기 스코어:**")
                                for lm in home_form["last_matches"]:
                                    res_c = "#2ed573" if lm['result'] == "W" else ("#747d8c" if lm['result'] == "D" else "#ff4757")
                                    st.markdown(f"<span style='font-size:12px; color:#a0aec0;'>• {lm['date']} ({lm['venue']}) vs {lm['opponent']} : <b style='color:{res_c};'>{lm['score']} ({lm['result']})</b></span>", unsafe_allow_html=True)
                        with det_col2:
                            if away_form.get("has_data") and away_form.get("last_matches"):
                                st.caption(f"✈️ **{m['away_team']} 최근 5경기 스코어:**")
                                for lm in away_form["last_matches"]:
                                    res_c = "#2ed573" if lm['result'] == "W" else ("#747d8c" if lm['result'] == "D" else "#ff4757")
                                    st.markdown(f"<span style='font-size:12px; color:#a0aec0;'>• {lm['date']} ({lm['venue']}) vs {lm['opponent']} : <b style='color:{res_c};'>{lm['score']} ({lm['result']})</b></span>", unsafe_allow_html=True)

        if len(filtered) > 60:
            st.info(f"💡 현재 상위 60개 경기를 표시 중입니다. (상단의 게임 유형 및 리그 필터를 활용해 보세요!)")


# ==============================================================================
# TAB 2: 🎯 AI 추천 2~3폴더 조합 & 베팅 바구니 (NEW!)
# ==============================================================================
with tab2:
    st.subheader("🎯 AI 엄선 2~3폴더 조합 & 맞춤 베팅 바구니")
    st.caption("베트맨 프로토 규정(동일 경기 중복 배제) 및 확률 곱셈 법칙 기반 최적의 다폴더 조합을 생성합니다.")

    if not predictions:
        st.warning("⚠️ 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # -------------------------------------------------------------
        # 고정 축(Banker) + 변수 경기 전수 마킹(Wheel) 헤징 전략
        # -------------------------------------------------------------
        st.markdown("### 🛡️ 고정 축(Banker 2경기) + 변수 경기 분할 마킹(Wheel 1경기) 전략")
        st.caption("💡 승률 높은 축 2경기에 변수 경기 [승 / 무 / 패] 3가지를 분할 마킹하는 전략입니다. (주의: 축 2경기 중 하나라도 틀리면 3장 모두 미적중되어 전액 손실이 발생합니다)")

        wheel_data = build_banker_wheel_combinations(predictions)
        if wheel_data.get("valid"):
            b1 = wheel_data["banker_1"]
            b2 = wheel_data["banker_2"]
            bank_odds = wheel_data["bank_combined_odds"]
            bank_prob_pct = wheel_data["bank_combined_prob_pct"]
            wm = wheel_data["wheel_match"]
            min_combo_odds = wheel_data["min_combo_odds"]
            tickets = wheel_data["tickets"]

            with st.container(border=True):
                # 상단 헤더 요약 3열
                wc1, wc2, wc3 = st.columns([1.5, 1.5, 1.2])
                with wc1:
                    st.markdown("##### 📌 고정 축 (Bankers 2경기)")
                    st.markdown(
                        f"<div class='notranslate' style='font-size:0.95rem; line-height:1.6;'>"
                        f"• <b>#{b1['match_seq']} [{b1['league']}]</b> {b1['home_team']} vs {b1['away_team']}<br>"
                        f"&nbsp;&nbsp;&nbsp;↳ 픽: <span style='color:#2ed573; font-weight:bold;'>{b1['pick']}</span> ({b1['odds']}배 | AI 승률 {b1['prob']*100:.1f}%)<br>"
                        f"• <b>#{b2['match_seq']} [{b2['league']}]</b> {b2['home_team']} vs {b2['away_team']}<br>"
                        f"&nbsp;&nbsp;&nbsp;↳ 픽: <span style='color:#2ed573; font-weight:bold;'>{b2['pick']}</span> ({b2['odds']}배 | AI 승률 {b2['prob']*100:.1f}%)"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"🔒 **축 2경기 결합 배당:** `{bank_odds}배` (AI 결합 승률: `{bank_prob_pct}`)")

                with wc2:
                    st.markdown("##### 🎲 변수 타겟 (Wheel 1경기 분할 마킹)")
                    st.markdown(
                        f"<div class='notranslate' style='font-size:0.95rem; line-height:1.6;'>"
                        f"• <b>#{wm['match_seq']} [{wm['league']}]</b> {wm['home_team']} vs {wm['away_team']}<br>"
                        f"&nbsp;&nbsp;&nbsp;↳ 🏠 홈 승리: <b>{wm['odds_home']}배</b><br>"
                        f"&nbsp;&nbsp;&nbsp;↳ 🤝 무승부: <b style='color:#ffa502;'>{wm['odds_draw']}배</b><br>"
                        f"&nbsp;&nbsp;&nbsp;↳ ✈️ 원정 승리: <b>{wm['odds_away']}배</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("⚡ **[홈 / 무 / 원정] 3장 분할 마킹 (축 2경기 통과 시 1장 적중)**")

                with wc3:
                    st.markdown("##### 🎯 전략 기대 성과")
                    st.metric("최소 결합 배당률", f"{min_combo_odds} 배", "3장 중 최저 배당 티켓")
                    st.metric("축 2경기 결합 확률", bank_prob_pct, "축 2경기 모두 적중해야 환급")

                st.divider()

                # 시뮬레이터
                st.markdown("##### 💰 티켓 3장 실전 베팅 & 환급금 시뮬레이터")
                sim_col1, sim_col2 = st.columns([1.5, 2.5])
                with sim_col1:
                    w_stake = st.number_input(
                        "티켓당 베팅 금액 (원, 조합 100원 단위)",
                        min_value=100,
                        max_value=100000,
                        value=10000,
                        step=100,
                        key="wheel_stake_input"
                    )
                    total_w_stake = w_stake * 3
                    st.caption(f"총 투자 금액: **3장 × {w_stake:,}원 = {total_w_stake:,}원** (1인 1회차 한도 10만 원)")
                    if total_w_stake > 100000:
                        st.warning("⚠️ 3장 총 투표금액이 1인 1회차 구매 한도(100,000원)를 초과했습니다.")
                
                with sim_col2:
                    st.info(
                        f"💡 **구조 안내:** 3장 중 최대 1장만 당첨됩니다. "
                        f"축 2경기(승률 {bank_prob_pct})가 모두 적중할 때에만 당첨금이 발생하며, 축 경기 중 하나라도 틀리면 총 베팅금({total_w_stake:,}원) 전체가 손실됩니다."
                    )

                # 3장 티켓별 결과 시뮬레이션 표
                t1_odds = tickets[0]["combined_odds"]
                t2_odds = tickets[1]["combined_odds"]
                t3_odds = tickets[2]["combined_odds"]

                t1_ret = calculate_payout(t1_odds, w_stake)
                t2_ret = calculate_payout(t2_odds, w_stake)
                t3_ret = calculate_payout(t3_odds, w_stake)

                t1_prof = t1_ret - total_w_stake
                t2_prof = t2_ret - total_w_stake
                t3_prof = t3_ret - total_w_stake

                t1_roi = (t1_prof / total_w_stake) * 100
                t2_roi = (t2_prof / total_w_stake) * 100
                t3_roi = (t3_prof / total_w_stake) * 100

                wheel_df = pd.DataFrame([
                    {
                        "티켓": "🎫 티켓 1 (홈승)",
                        "마킹 조합": f"[축 2경기] + [{wm['home_team']} 승 ({wm['odds_home']}배)]",
                        "합산 배당률": f"{t1_odds}배",
                        "베팅액": f"{w_stake:,}원",
                        "적중 시 환급금": f"{t1_ret:,}원",
                        "실질 순수익": f"{t1_prof:+,}원",
                        "수익률(ROI)": f"{t1_roi:+.1f}%",
                        "결과 의미": "정배 승리 방어"
                    },
                    {
                        "티켓": "🎫 티켓 2 (무승부 🔥)",
                        "마킹 조합": f"[축 2경기] + [무승부 ({wm['odds_draw']}배)]",
                        "합산 배당률": f"{t2_odds}배",
                        "베팅액": f"{w_stake:,}원",
                        "적중 시 환급금": f"{t2_ret:,}원",
                        "실질 순수익": f"{t2_prof:+,}원",
                        "수익률(ROI)": f"{t2_roi:+.1f}%",
                        "결과 의미": "🔥 무캐기 고배당"
                    },
                    {
                        "티켓": "🎫 티켓 3 (원정승 ⚡)",
                        "마킹 조합": f"[축 2경기] + [{wm['away_team']} 승 ({wm['odds_away']}배)]",
                        "합산 배당률": f"{t3_odds}배",
                        "베팅액": f"{w_stake:,}원",
                        "적중 시 환급금": f"{t3_ret:,}원",
                        "실질 순수익": f"{t3_prof:+,}원",
                        "수익률(ROI)": f"{t3_roi:+.1f}%",
                        "결과 의미": "⚡ 역배 방어"
                    }
                ])
                st.dataframe(wheel_df, use_container_width=True, hide_index=True)

                min_prof = min(t1_prof, t2_prof, t3_prof)
                max_prof = max(t1_prof, t2_prof, t3_prof)
                min_roi = min(t1_roi, t2_roi, t3_roi)
                max_roi = max(t1_roi, t2_roi, t3_roi)

                if min_prof >= 0:
                    st.success(
                        f"✨ **결과 요약:** 축 2경기(승률 {bank_prob_pct}) 적중 시, 변수 경기 결과에 따라 "
                        f"**최소 {min_prof:+,}원 ({min_roi:+.1f}%) ~ 최대 {max_prof:+,}원 ({max_roi:+.1f}%) 순이익**이 발생합니다."
                    )
                else:
                    st.warning(
                        f"⚠️ **결과 요약:** 축 2경기(승률 {bank_prob_pct})가 적중하더라도, 저배당 발생 시 "
                        f"**최소 {min_prof:+,}원 ({min_roi:+.1f}%, 손실 가능) ~ 최대 {max_prof:+,}원 ({max_roi:+.1f}%)** 범위로 손실이 발생할 수 있습니다."
                    )

                # 개별 티켓 바구니 담기 버튼 (기존 바구니에 병합)
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("📥 티켓 1 경기들을 바구니에 담기", use_container_width=True):
                        st.session_state.setdefault("cart_picks", {}).update({
                            str(b1["match_seq"]): b1.get("chosen_selection", "home_win"),
                            str(b2["match_seq"]): b2.get("chosen_selection", "home_win"),
                            str(wm["match_seq"]): "home_win"
                        })
                        st.rerun()
                with btn_cols[1]:
                    if st.button("📥 티켓 2 경기들을 바구니에 담기", use_container_width=True):
                        st.session_state.setdefault("cart_picks", {}).update({
                            str(b1["match_seq"]): b1.get("chosen_selection", "home_win"),
                            str(b2["match_seq"]): b2.get("chosen_selection", "home_win"),
                            str(wm["match_seq"]): "draw"
                        })
                        st.rerun()
                with btn_cols[2]:
                    if st.button("📥 티켓 3 경기들을 바구니에 담기", use_container_width=True):
                        st.session_state.setdefault("cart_picks", {}).update({
                            str(b1["match_seq"]): b1.get("chosen_selection", "home_win"),
                            str(b2["match_seq"]): b2.get("chosen_selection", "home_win"),
                            str(wm["match_seq"]): "away_win"
                        })
                        st.rerun()

                with st.expander("📋 헷징 3장 티켓 전체 마킹 슬립 복사 (클릭하여 펼치기)", expanded=False):
                    wheel_slip_lines = [
                        "🛡️ [고정 축 + 변수 헷징 3장 티켓 마킹 슬립]",
                        f"🔒 고정 축 2경기: #{b1['match_seq']} [{b1['league']}] {b1['home_team']} vs {b1['away_team']} ({b1['pick']}) | #{b2['match_seq']} [{b2['league']}] {b2['home_team']} vs {b2['away_team']} ({b2['pick']})",
                        f"🎲 변수 마킹 경기: #{wm['match_seq']} [{wm['league']}] {wm['home_team']} vs {wm['away_team']}",
                        "────────────────────────────────────────",
                        f"🎫 티켓 1 (홈승 방어): [축 2경기] + {wm['home_team']} 승 ({wm['odds_home']}배) ➔ 합산 {t1_odds}배 (환급: {t1_ret:,}원 | 순익: {t1_prof:+,}원)",
                        f"🎫 티켓 2 (무승부 잭팟): [축 2경기] + 무승부 ({wm['odds_draw']}배) ➔ 합산 {t2_odds}배 (환급: {t2_ret:,}원 | 순익: {t2_prof:+,}원)",
                        f"🎫 티켓 3 (역배승 방어): [축 2경기] + {wm['away_team']} 승 ({wm['odds_away']}배) ➔ 합산 {t3_odds}배 (환급: {t3_ret:,}원 | 순익: {t3_prof:+,}원)",
                        "────────────────────────────────────────",
                        f"💰 티켓당 {w_stake:,}원 베팅 (총 투자 {total_w_stake:,}원)",
                        f"✨ 축 2경기(승률 {bank_prob_pct}) 적중 시 예상 순익: {min_prof:+,}원 ~ {max_prof:+,}원"
                    ]
                    st.code("\n".join(wheel_slip_lines), language="text")
        else:
            st.info(f"💡 {wheel_data.get('error', '현재 회차에서 조건에 맞는 축 또는 변수 경기를 찾지 못했습니다.')}")

        st.divider()

        # AI 추천 5대 프리셋 생성
        parlay_presets = build_recommended_parlays(predictions)

        st.markdown("### 🌟 AI 추천 5대 포트폴리오 (시뮬레이션)")
        st.write("승률, 기대가치(EV), 카테고리 분산 기준에 따라 구성된 조합 시뮬레이션입니다:")
        st.caption("⚠️ **복리 마진 안내:** 2경기 이상 조합 시 베트맨 하우스 마진(약 18%)이 곱으로 누적(2폴더 ~27.8%, 3폴더 ~38.6%)됩니다. 단일 경기 EV가 검증되기 전까지 복수 조합은 **'참고용 시뮬레이션'**으로 제공됩니다.")

        preset_labels = {
            "safety_double": ("🛡️ 안정형 2폴더 (Safety Double)", "개별 승률 50% 이상 중심, 안정적인 환급 지향 (합산 2~4배)"),
            "value_double": ("⚖️ 밸런스 가치 2폴더 (Value Double)", "양의 기대가치(EV)와 적중률의 균형 (합산 3~6배)"),
            "cross_double": ("🔀 크로스 2폴더 (승무패 + 언더오버)", "승패와 득점 기준점을 크로스 분산하여 경기 리스크 헤지"),
            "high_triple": ("🚀 고배당 실전 3폴더 (High-Yield Triple)", "소액 분산 투자용, 고배당 양의 EV 3폴더 조합"),
            "single_king": ("⚡ 단통 1경기 집중 픽 (Single King)", "다른 경기와 묶을 필요 없는 100% 단일 구매 추천 픽")
        }

        for key, (title, desc) in preset_labels.items():
            combo = parlay_presets.get(key)
            if not combo or not combo.get("valid"):
                continue

            with st.container(border=True):
                col_info, col_m1, col_m2, col_m3, col_btn = st.columns([2.5, 1, 1, 1, 1.2])
                with col_info:
                    st.markdown(f"#### {title}")
                    st.caption(desc)
                    if combo.get("compound_margin_pct"):
                        st.caption(f"누적 복리 마진: `~{combo['compound_margin_pct']:.1f}%`")
                with col_m1:
                    st.metric("합산 배당률", f"{combo['combined_odds']} 배")
                with col_m2:
                    st.metric("AI 결합 승률", combo["combined_prob_pct"])
                with col_m3:
                    st.metric("기대가치 (EV)", combo["combined_ev_pct"])
                with col_btn:
                    if st.button("📥 바구니 담기", key=f"copy_preset_{key}"):
                        for p in combo["picks"]:
                            st.session_state["cart_picks"][str(p["match_seq"])] = p.get("chosen_selection", "best")
                        st.rerun()

                # 포함된 경기 세부 내역 테이블
                rows = []
                for p in combo["picks"]:
                    rows.append({
                        "경기번호": f"#{p['match_seq']}",
                        "개최일시": p["match_time"],
                        "리그": p["league"],
                        "경기 대진": f"{p['home_team']} vs {p['away_team']}",
                        "게임 유형": p["category_label"],
                        "AI 추천 선택": p["selection_name"],
                        "배당률": f"{p['odds']}배",
                        "AI 승률": f"{p['prob']*100:.1f}%",
                        "기대가치": f"{p['ev']*100:+.1f}%"
                    })
                df_p = pd.DataFrame(rows)
                st.dataframe(df_p, use_container_width=True, hide_index=True)

                with st.expander(f"📋 {title} 마킹 슬립 복사", expanded=False):
                    preset_slip = generate_bet_slip_text(
                        combo["picks"],
                        combo["combined_odds"],
                        combo["combined_prob_pct"],
                        combo["combined_ev_pct"]
                    )
                    st.code(preset_slip, language="text")

        st.divider()

        # -------------------------------------------------------------
        # 실시간 맞춤 조합 바구니 (Interactive Bet Cart)
        # -------------------------------------------------------------
        st.markdown("### 🛒 실시간 내 맞춤 조합 바구니")
        st.caption("원하는 경기를 바구니에 담아 실시간 배당률, AI 승률, 예상 당첨금을 계산해보세요.")

        # 바구니 초기화 버튼
        cb_col1, cb_col2 = st.columns([4, 1])
        with cb_col2:
            if st.button("🗑️ 바구니 비우기", use_container_width=True):
                st.session_state["cart_picks"] = {}
                st.session_state["cart_seqs"] = []
                st.rerun()

        # 현재 담긴 경기 객체 추출
        pred_map = {str(m["match_seq"]): m for m in predictions}
        cart_picks = [
            get_pick_item(pred_map[seq], selection=st.session_state["cart_picks"][seq])
            for seq in st.session_state["cart_picks"]
            if seq in pred_map
        ]

        # 바구니 조합 지표 계산
        metrics = calculate_parlay_metrics(cart_picks)

        if not cart_picks:
            st.info("💡 아직 바구니에 담긴 경기가 없습니다. 위 프리셋의 [📥 바구니 담기] 버튼이나 '실시간 베트맨 경기' 탭에서 [승/무/패] 배당 버튼을 눌러보세요.")
        else:
            # 베트맨 규정 검증 메시지
            if not metrics["valid"]:
                st.error(metrics["error"])
            else:
                if metrics["folders"] == 1 and not cart_picks[0]["is_single"]:
                    st.warning("⚠️ 1경기만 선택되었습니다. 해당 경기는 '한경기 구매(단통)' 가능 경기가 아니므로, 베트맨 규정상 최소 2경기 이상 묶어야 발매가 가능합니다.")
                else:
                    st.success(f"✅ 베트맨 프로토 규정 통과! ({metrics['folders']}폴더 조합 가능)")

                # 지표 카드
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("총 조합 폴더 수", f"{metrics['folders']} 경기")
                with k2:
                    st.metric("최종 합산 배당률", f"{metrics['combined_odds']} 배")
                with k3:
                    st.metric("AI 결합 승률", metrics["combined_prob_pct"])
                with k4:
                    st.metric("조합 기대가치 (EV)", metrics["combined_ev_pct"], delta=f"권장 베팅 비중: {metrics['kelly_pct']}")

                # 담긴 경기 목록 테이블 (삭제 버튼 포함)
                st.markdown("##### 📌 바구니에 담긴 픽 목록")
                for p in cart_picks:
                    r1, r2, r3, r4, r5 = st.columns([1, 3, 2, 1, 0.8])
                    with r1:
                        st.write(f"**#{p['match_seq']}**")
                    with r2:
                        st.markdown(f"<div class='notranslate'>[{p['league']}] <b>{p['home_team']} vs {p['away_team']}</b></div>", unsafe_allow_html=True)
                    with r3:
                        st.markdown(f"<div class='notranslate' style='color: #2ed573; font-weight: bold;'>{p['category_label']}: {p['selection_name']}</div>", unsafe_allow_html=True)
                    with r4:
                        st.write(f"**{p['odds']}배** ({p['prob']*100:.1f}%)")
                    with r5:
                        if st.button("❌", key=f"del_cart_{p['match_seq']}"):
                            if p["match_seq"] in st.session_state["cart_picks"]:
                                del st.session_state["cart_picks"][p["match_seq"]]
                            st.rerun()

                # -------------------------------------------------------------
                # 📐 켈리 기준(Kelly Criterion) 자금관리 계산기
                # -------------------------------------------------------------
                st.markdown("##### 📐 켈리 기준(Kelly Criterion) 자금관리 계산기")
                st.caption("켈리 기준 공식에 따라 시드머니 규모와 기대가치(EV)에 맞춘 추천 베팅 금액을 산출합니다 (과도한 변동성 방지를 위해 1/4~1/2 켈리 권장).")

                k_col1, k_col2, k_col3 = st.columns([1.5, 1.5, 1.2])
                with k_col1:
                    total_bankroll = st.number_input(
                        "💼 총 보유 시드머니 (Bankroll, 원)",
                        min_value=10000,
                        max_value=100000000,
                        value=st.session_state.get("user_bankroll", 200000),
                        step=10000,
                        key="user_bankroll_input"
                    )
                    st.session_state["user_bankroll"] = total_bankroll

                with k_col2:
                    kelly_mode = st.selectbox(
                        "🛡️ 켈리 운용 전략 (위험 선호도)",
                        [
                            "1/4 켈리 (0.25x) - 안정 지향 / MDD 최소화 (권장 ⭐)",
                            "1/2 켈리 (0.50x) - 밸런스 베팅 (허용 상한선)",
                            "1/8 켈리 (0.125x) - 보수적 소액 테스트"
                        ],
                        index=0
                    )
                    multiplier = 0.25
                    if "1/2" in kelly_mode:
                        multiplier = 0.50
                    elif "1/8" in kelly_mode:
                        multiplier = 0.125

                with k_col3:
                    c_odds = metrics["combined_odds"]
                    c_prob = metrics["combined_prob"]
                    c_ev = metrics["combined_ev"]
                    folders_cnt = metrics.get("folders", len(cart_picks))
                    is_single_match = (folders_cnt == 1)
                    min_unit = 1000 if is_single_match else 100
                    unit_desc = "한경기 1,000원 단위" if is_single_match else "조합 100원 단위"
                    
                    raw_kelly = 0.0
                    if c_odds > 1.0 and c_ev > 0:
                        b_val = c_odds - 1.0
                        q_val = 1.0 - c_prob
                        raw_kelly = max(0.0, (c_prob * b_val - q_val) / b_val)
                        
                    applied_kelly = raw_kelly * multiplier
                    rec_stake = int(total_bankroll * applied_kelly)
                    rec_stake_proto = max(min_unit, (rec_stake // min_unit) * min_unit) if rec_stake >= min_unit else min_unit
                    rec_stake_proto = min(100000, rec_stake_proto)
                    
                    st.metric("추천 베팅 비중", f"{applied_kelly * 100:.2f}%", f"이론 100%: {raw_kelly * 100:.1f}%")

                with st.container(border=True):
                    res_k1, res_k2, res_k3 = st.columns([1.5, 1.5, 1])
                    with res_k1:
                        st.markdown(f"**💡 수학적 권장 베팅액:** <span style='font-size:1.25rem; color:#2ed573; font-weight:800;'>{rec_stake_proto:,} 원</span>", unsafe_allow_html=True)
                        st.caption(f"시드머니({total_bankroll:,}원)의 {rec_stake_proto/total_bankroll*100:.1f}% (베트맨 {unit_desc} 반영 | 회차 한도 10만 원)")
                    with res_k2:
                        est_ret = calculate_payout(c_odds, rec_stake_proto)
                        est_pnl = est_ret - rec_stake_proto
                        st.markdown(f"**적중 시 순익:** <span style='color:#60a5fa; font-weight:700;'>{est_pnl:+,} 원</span> (환급: {est_ret:,}원)", unsafe_allow_html=True)
                        st.caption(f"수학적 기대 순익(EV): {int(rec_stake_proto * c_ev):+,}원")
                    with res_k3:
                        if st.button("👉 권장 금액 적용", use_container_width=True):
                            st.session_state["applied_stake"] = rec_stake_proto
                            st.rerun()

                # 모의 베팅 환급금 계산기
                st.markdown("##### 💰 예상 환급금 & 기대 순익 시뮬레이터")
                default_val = st.session_state.get("applied_stake", 10000 if is_single_match else 1000)
                default_val = max(min_unit, (default_val // min_unit) * min_unit)
                default_val = min(100000, default_val)
                stake_krw = st.number_input(
                    f"베팅 희망 금액 (원, {unit_desc}, 최대 10만 원)",
                    min_value=min_unit,
                    max_value=100000,
                    value=default_val,
                    step=min_unit,
                    key="cart_stake_input"
                )
                if stake_krw > 100000:
                    st.warning("⚠️ 1인 1회차 구매 한도(100,000원)를 초과했습니다.")
                
                exp_return = calculate_payout(metrics["combined_odds"], stake_krw)
                exp_profit = exp_return - stake_krw
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"• 적중 시 예상 환급금: **`{exp_return:,} 원`** (순수익: {exp_profit:+,}원)")
                with res_col2:
                    st.markdown(f"• 수학적 기대 이익(EV): **`{int(stake_krw * metrics['combined_ev']):+,} 원`**")

                # 바구니 전체 베팅 슬립 복사
                st.markdown("##### 📋 마킹 슬립 원클릭 복사 (카톡/베트맨 입력용)")
                cart_slip = generate_bet_slip_text(
                    cart_picks,
                    metrics["combined_odds"],
                    metrics["combined_prob_pct"],
                    metrics["combined_ev_pct"],
                    stake=stake_krw
                )
                st.code(cart_slip, language="text")


# ==============================================================================
# TAB 3: 🏁 지난 경기 결과 vs AI 예측 채점 (실전 검증)
# ==============================================================================
with tab3:
    col_t_title, col_t_btn = st.columns([3, 1.2])
    with col_t_title:
        st.subheader("🏁 지난 베트맨 실제 경기 결과 vs AI 예측 실전 대조")
        st.caption("베트맨 공식 종료 경기들의 실제 최종 스코어 및 결과와 AI의 사전 예측 승률, 추천 픽(GRADE S/A/B), 적중 여부를 1:1로 비교 검증합니다.")
    with col_t_btn:
        st.write("")
        if st.button("🔄 실시간 결과 재채점", key="re_evaluate_btn_tab3", use_container_width=True):
            with st.spinner("베트맨 최근 경기 결과 전수 채점 중..."):
                try:
                    from tracker.result_evaluator import evaluate_betman_results
                    evaluate_betman_results()
                    st.cache_data.clear()
                    st.success("채점 완료! 최신 경기 결과와 적중률이 갱신되었습니다.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"채점 오류: {ex}")

    recent_picks = stats.get("recent_evaluated_picks", [])
    if not recent_picks:
        st.info("💡 아직 채점된 지난 경기 데이터가 없습니다. 상단의 '실시간 결과 재채점' 버튼을 눌러주세요.")
    else:
        # 1. 상단 KPI 카드
        total_hist = len(recent_picks)
        hit_hist = sum(1 for p in recent_picks if p.get("is_hit"))
        hit_rate = (hit_hist / total_hist * 100) if total_hist > 0 else 0.0
        total_pnl = sum(p.get("net_pnl_num", 0) for p in recent_picks)
        s_picks = [p for p in recent_picks if "S (" in p.get("grade", "")]
        s_hits = sum(1 for p in s_picks if p.get("is_hit"))
        s_rate = (s_hits / len(s_picks) * 100) if s_picks else 0.0

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.metric("총 종료 분석 경기", f"{total_hist:,} 건")
        with hc2:
            st.metric("AI 추천 적중 건수", f"{hit_hist:,} 건", f"전체 적중률 {hit_rate:.1f}%")
        with hc3:
            st.metric("👑 GRADE S 적중률", f"{s_rate:.1f}%", f"{s_hits}/{len(s_picks)} 적중 (주력)")
        with hc4:
            st.metric("실전 가상 총 손익", f"{total_pnl:+,.0f} 원", f"경기당 1만원 베팅 기준")

        st.divider()

        # 2. 인터랙티브 다차원 필터 바
        with st.container(border=True):
            st.markdown("##### 🔍 지난 경기 대조 필터 & 검색")
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 1.5, 2])
            
            with f_col1:
                avail_rounds = sorted(list(set(str(p.get("round", "")) for p in recent_picks)), reverse=True)
                round_filter = st.selectbox("📅 발매 회차 선택", ["전체 회차"] + [f"{r}회차" for r in avail_rounds])

            with f_col2:
                hit_filter = st.selectbox("🎯 적중 여부", ["전체 보기", "✅ 적중한 경기만", "❌ 미적중 경기만"])

            with f_col3:
                grade_filter = st.selectbox("👑 AI 추천 등급", ["전체 등급", "GRADE S (참고용 - 고승률)", "GRADE A (참고용 - 밸런스)", "GRADE B (참고용 - 고배당)"])

            with f_col4:
                team_search = st.text_input("🔎 팀 이름 검색", placeholder="예: 브리즈번, 플라멩구, 릴레스트룀")

            sub_f1, sub_f2 = st.columns([2, 1.5])
            with sub_f1:
                avail_leagues = sorted(list(set(p.get("league", "기타") for p in recent_picks if p.get("league"))))
                league_filter = st.selectbox("🏆 리그 선택", ["전체 리그"] + avail_leagues)
            with sub_f2:
                view_mode = st.radio("👀 결과 표시 방식", ["📇 카드형 대조 뷰", "📊 데이터 테이블 뷰"], horizontal=True)

        # 필터링 적용
        f_picks = recent_picks
        if round_filter != "전체 회차":
            target_r = round_filter.replace("회차", "")
            f_picks = [p for p in f_picks if str(p.get("round")) == target_r]
        if hit_filter == "✅ 적중한 경기만":
            f_picks = [p for p in f_picks if p.get("is_hit")]
        elif hit_filter == "❌ 미적중 경기만":
            f_picks = [p for p in f_picks if not p.get("is_hit")]
        if grade_filter != "전체 등급":
            g_tag = "S (" if "S" in grade_filter else ("A (" if "A" in grade_filter else "B (")
            f_picks = [p for p in f_picks if g_tag in p.get("grade", "")]
        if league_filter != "전체 리그":
            f_picks = [p for p in f_picks if p.get("league") == league_filter]
        if team_search:
            q = team_search.lower()
            f_picks = [p for p in f_picks if q in p.get("match", "").lower() or q in p.get("home_team", "").lower() or q in p.get("away_team", "").lower()]

        st.subheader(f"📋 채점 경기 대조 목록 ({len(f_picks)}경기)")

        if not f_picks:
            st.info("선택하신 조건에 일치하는 지난 경기가 없습니다.")
        elif view_mode == "📊 데이터 테이블 뷰":
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
        else:
            for p in f_picks[:60]:
                is_win = p.get("is_hit", False)
                status_badge = "<span style='background:#2ed573; color:white; padding:4px 10px; border-radius:6px; font-weight:800; font-size:13px;'>✅ 적중!</span>" if is_win else "<span style='background:#ff4757; color:white; padding:4px 10px; border-radius:6px; font-weight:800; font-size:13px;'>❌ 미적중</span>"

                grade_raw = p.get("grade", "")
                if "S (" in grade_raw:
                    g_badge = "<span class='badge-s'>GRADE S</span>"
                elif "A (" in grade_raw:
                    g_badge = "<span class='badge-a'>GRADE A</span>"
                else:
                    g_badge = "<span class='badge-b'>GRADE B</span>"

                with st.container(border=True):
                    st.markdown(
                        f"<div class='notranslate' style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
                        f"<div><b style='font-size:1.1rem;'>[{p.get('league', '기타')}] {p.get('match')}</b> &nbsp; {g_badge} &nbsp; <span class='badge-cat'>{p.get('game_type', '승무패')}</span></div>"
                        f"<div>{status_badge}</div>"
                        f"</div>"
                        f"<div class='match-meta-bar notranslate' style='margin-bottom:12px; font-size:12px;'>"
                        f"📅 {p.get('date')} &nbsp;|&nbsp; 🔢 베트맨 경기번호: #{p.get('match_seq')} &nbsp;|&nbsp; 🏷️ {p.get('round')}회차"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    c_res, c_ai, c_pnl = st.columns([1.5, 2.2, 1.5])
                    with c_res:
                        st.markdown("**🏁 실제 경기 결과:**")
                        st.markdown(f"<div style='font-size:1.5rem; font-weight:800; color:#f1f2f6;'>{p.get('score', '-')}</div>", unsafe_allow_html=True)
                        st.markdown(f"판정: <b style='color:#70a1ff;'>{p.get('actual_result', '-')}</b>", unsafe_allow_html=True)

                    with c_ai:
                        st.markdown("**🧠 당시 AI 예측 & 추천:**")
                        st.markdown(f"<div class='notranslate' style='color:#2ed573; font-weight:800; font-size:1.05rem;'>👉 {p.get('pick')}</div>", unsafe_allow_html=True)
                        st.caption(f"배당률: **{p.get('odds')}배** | AI 승률: **{p.get('ai_prob')}%** | 기대가치: **{p.get('ev_pct')}**")

                    with c_pnl:
                        st.markdown("**💰 실전 채점 손익:**")
                        pnl_color = "#2ed573" if is_win else "#ff4757"
                        st.markdown(f"<div style='font-size:1.3rem; font-weight:800; color:{pnl_color};'>{p.get('net_pnl', '0원')}</div>", unsafe_allow_html=True)
                        st.caption("10,000원 베팅 기준")

            if len(f_picks) > 60:
                st.info(f"💡 전체 {len(f_picks)}경기 중 상위 60경기를 표시했습니다. 전체 경기 내역은 위의 '📊 데이터 테이블 뷰'를 선택하시면 한눈에 보실 수 있습니다.")


# ==============================================================================
# TAB 4: 📊 AI 누적 적중률 & 수익률(ROI) 통계
# ==============================================================================
with tab4:
    st.subheader("📈 베트맨 공식 경기 결과 기반 AI 누적 적중률 & ROI 통계")
    source_txt = stats.get("source", "베트맨 공식 전수 채점")
    st.caption(f"📊 {source_txt} | 갱신 시각: {stats.get('updated_at', '')} (1경기당 10,000원 베팅 기준)")

    if not stats:
        st.warning("⚠️ 아직 검증 통계 데이터가 없습니다. Tab 3의 '실시간 결과 재채점' 버튼을 눌러주세요.")
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        roi_ci_l = stats.get("roi_ci_low", 0.0)
        roi_ci_h = stats.get("roi_ci_high", 0.0)
        with sc1:
            st.metric("분석 경기 (종료)", f"{stats.get('sample_matches_analyzed', 0):,} 건")
        with sc2:
            st.metric("AI 추천 베팅", f"{stats.get('total_bets_placed', 0):,} 건")
        with sc3:
            st.metric("전체 적중률", f"{stats.get('win_rate_pct', 0)}%", f"{stats.get('won_bets', 0)}/{stats.get('total_bets_placed', 0)} 적중")
        with sc4:
            st.metric("누적 수익률 (고정 1만원)", f"{stats.get('roi_pct', 0):+.2f}%", f"95% CI: [{roi_ci_l:+.1f}%, {roi_ci_h:+.1f}%]")

        st.divider()

        # 자금관리 비교 & CLV 성과
        st.markdown("### ⚖️ 자금 관리 방식 비교 & 마감 배당 대비 성과 (CLV)")
        stk = stats.get("staking_comparison", {})
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.metric("고정 베팅 (1만 원)", f"{stk.get('flat_roi_pct', stats.get('roi_pct', 0)):+.2f}%", f"순익: {stk.get('flat_net_profit_krw', 0):+,}원")
            st.caption("베트맨 1회차 1경기당 고정 10,000원 베팅")
        with c_col2:
            st.metric("1/4 켈리 동적 베팅", f"{stk.get('kelly_roi_pct', 0):+.2f}%", f"순익: {stk.get('kelly_net_profit_krw', 0):+,}원")
            st.caption(f"초기 100만 원 잔고 ➔ 최종: {stk.get('kelly_final_bankroll_krw', 0):,}원 (최대 2% 상한)")
        with c_col3:
            clv_cnt = stats.get("clv_sample_count", 0)
            clv_avg = stats.get("avg_clv_pct", 0.0)
            st.metric("CLV (마감 대비 우위)", f"{clv_avg:+.2f}%", f"스냅샷 표본: {clv_cnt}건")
            st.caption("예측 시점 배당 / 마감 배당 비율 (시장 장기 우위 지표)")

        st.divider()

        # 등급별 세부 성과 분석
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

        st.info("💡 **핵심 인사이트 (미래 누수 차단 전수 시계열 검증):** 미래 경기 데이터를 엄격히 차단하고 순수 사전 폼만으로 검증한 실전 성과입니다. 베트맨의 약 18% 하우스 마진으로 인해 전체 누적 ROI는 -2.98% (95% CI: [-9.61%, +3.74%])입니다. 특히 복수 조합(2~3폴더)은 하우스 마진이 곱으로 누적(2폴더 ~27.8%, 3폴더 ~38.6%)되므로, 단일 경기에서 확실한 EV/CLV 우위가 검증되기 전까지는 조합 베팅을 주의하고 '시뮬레이션 전용'으로 활용해야 합니다.")

        # 회차별 성과 비교 테이블
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
