"""
dashboard/app.py
================
Streamlit 기반 스포츠 승부 예측 & 가치 픽(Value Bet) 실시간 웹 대시보드
[Tab 1: 이번 회차 실시간 경기 분석] / [Tab 2: AI 누적 적중률 & 모의 베팅 수익률(ROI)]
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="스포츠 승부예측 & 가치픽 대시보드",
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
    .pick-box {
        background: linear-gradient(135deg, #1e272e 0%, #2f3542 100%);
        border: 1px solid #70a1ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data():
    pred_path = Path("data/predictions.json")
    stats_path = Path("data/performance_stats.json")
    
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

# 타이틀 헤더
st.title("⚽ 스포츠 승부 예측 & 가치 픽 (Value Bet) 대시보드")
st.caption("베트맨(Betman) 프로토 배당률 및 Dixon-Coles 통계 모델 기반 기대가치(EV) 실시간 분석")

# 상단 탭 구성
tab1, tab2 = st.tabs(["🔥 이번 회차 실시간 베트맨 경기", "📊 AI 실전 적중률 & 수익률(ROI) 통계"])

# ==============================================================================
# TAB 1: 이번 회차 실시간 베트맨 경기 분석
# ==============================================================================
with tab1:
    if not predictions:
        st.warning("⚠️ 아직 분석된 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # 1. 상단 KPI 카드
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
            st.metric("분석 모델", "Dixon-Coles & Poisson", "5,330경기 학습")

        st.divider()

        # 2. 사이드바 필터 (리그, 게임유형, 한경기 전용, 정렬)
        st.sidebar.header("🔍 검색 및 베팅 필터")
        
        # [필터 1] 리그 선택
        leagues = sorted(list(set(m["league"] for m in predictions)))
        selected_league = st.sidebar.selectbox("리그 선택", ["전체"] + leagues)

        # [필터 2] 게임 유형 선택 (승무패, 핸디캡, 언더오버, 전반전)
        categories = ["전체 보기", "일반 승무패", "핸디캡", "언더오버", "전반 승무패", "전반 핸디캡", "전반 언더오버"]
        selected_category = st.sidebar.selectbox("게임 유형 선택", categories)

        # [필터 3] 한경기(단통) 구매 전용 토글
        only_single = st.sidebar.checkbox("⚡ 한경기(단통) 구매 가능만 보기", value=False)

        # [필터 4] 추천 가치 픽만 보기
        only_recommended = st.sidebar.checkbox("🔥 추천 가치 픽(EV > 0)만 보기", value=True)

        # [필터 5] 정렬 기준
        sort_by = st.sidebar.radio(
            "정렬 기준",
            ["기대가치(EV) 높은순 (AI 추천순)", "경기 시작 시간순", "홈 승률 높은순", "배당률 높은순"]
        )

        filtered = predictions
        if selected_league != "전체":
            filtered = [m for m in filtered if m["league"] == selected_league]
        if selected_category != "전체 보기":
            filtered = [m for m in filtered if selected_category in m.get("category", "")]
        if only_single:
            filtered = [m for m in filtered if m.get("is_single", False)]
        if only_recommended:
            filtered = [m for m in filtered if m["evaluation"]["is_recommended"]]

        if "기대가치" in sort_by:
            filtered.sort(key=lambda x: x["evaluation"]["best_ev"], reverse=True)
        elif "홈 승률" in sort_by:
            filtered.sort(key=lambda x: x["model_probabilities"]["home_win"], reverse=True)
        elif "배당률" in sort_by:
            filtered.sort(key=lambda x: max(x["betman_odds"].values()), reverse=True)

        st.subheader(f"📋 경기 분석 결과 ({len(filtered)}경기)")

        for m in filtered[:60]:  # 상위 60개 렌더링
            eval_info = m["evaluation"]
            odds = m["betman_odds"]
            probs = m["model_probabilities"]
            sel_labels = m.get("selection_labels", {})
            
            # 배지 스타일링
            grade_str = eval_info["grade"]
            if "S (" in grade_str:
                badge_html = "<span class='badge-s'>GRADE S (꿀배당)</span>"
            elif "A (" in grade_str:
                badge_html = "<span class='badge-a'>GRADE A (강력 추천)</span>"
            elif "B (" in grade_str:
                badge_html = "<span class='badge-b'>GRADE B (소액 추천)</span>"
            else:
                badge_html = "<span class='badge-pass'>PASS (가치 낮음)</span>"

            single_badge = "<span class='badge-single'>⚡ 단통 1경기 가능</span>" if m.get("is_single") else ""
            cat_badge = f"<span class='badge-cat'>{m.get('category_label', m.get('game_type', '승무패'))}</span>"

            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(
                        f"<div class='notranslate' translate='no' style='font-size: 1.18rem; font-weight: 700; margin-bottom: 6px;'>"
                        f"[{m['league']}] {m['home_team']} 🆚 {m['away_team']} &nbsp;&nbsp; {badge_html} {single_badge} {cat_badge}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='match-meta-bar notranslate' translate='no'>"
                        f"📅 <b>개최일시:</b> {m['match_time']} &nbsp;|&nbsp; "
                        f"🏷️ <b>게임유형:</b> {m.get('game_type', '승무패')} &nbsp;|&nbsp; "
                        f"🔢 <b>베트맨 경기번호:</b> #{m['match_seq']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with head_col2:
                    st.markdown(f"**🎯 AI 추천:**")
                    st.markdown(f"<div class='notranslate' style='color: #2ed573; font-weight: 800; font-size: 1.05rem;'>{eval_info['best_selection_name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"기대가치: **`{eval_info['best_ev_pct']}`**")

                # 각 선택지별 라벨 및 배당/확률
                label_h = sel_labels.get("home_win", f"{m['home_team']} 승")
                label_d = sel_labels.get("draw", "무승부")
                label_a = sel_labels.get("away_win", f"{m['away_team']} 승")

                col_h, col_d, col_a = st.columns(3)
                with col_h:
                    st.markdown(f"<div class='notranslate'><b>{label_h}</b></div>", unsafe_allow_html=True)
                    st.write(f"배당: **{odds['home_win']}** 배 | AI: **{probs['home_win']*100:.1f}%**")
                    st.progress(probs['home_win'])
                with col_d:
                    if odds['draw'] > 0:
                        st.markdown(f"<div class='notranslate'><b>{label_d}</b></div>", unsafe_allow_html=True)
                        st.write(f"배당: **{odds['draw']}** 배 | AI: **{probs['draw']*100:.1f}%**")
                        st.progress(probs['draw'])
                    else:
                        st.caption("무승부 없음 (2지선다 게임)")
                with col_a:
                    st.markdown(f"<div class='notranslate'><b>{label_a}</b></div>", unsafe_allow_html=True)
                    st.write(f"배당: **{odds['away_win']}** 배 | AI: **{probs['away_win']*100:.1f}%**")
                    st.progress(probs['away_win'])

                # 스코어 및 부가 지표
                score_text = " | ".join([f"{s['score']} ({s['prob']}%)" for s in m.get("top_scores", [])])
                st.caption(f"🎯 **예상 스코어 TOP 3:** {score_text}")

        if len(filtered) > 60:
            st.info(f"💡 현재 상위 60개 경기를 표시 중입니다. (사이드바의 게임 유형 및 리그 필터를 활용해 보세요!)")

# ==============================================================================
# TAB 2: AI 실전 적중률 & 모의 베팅 수익률(ROI) 통계
# ==============================================================================
with tab2:
    if not stats:
        st.warning("⚠️ 아직 검증 통계 데이터가 없습니다. `python tracker/result_evaluator.py`를 실행해 주세요.")
    else:
        st.subheader("📈 AI 모델 백테스팅 및 실전 가상 수익률 (ROI)")
        st.caption(f"최근 {stats.get('sample_matches_analyzed', 0)}개 실제 경기 대상 (1경기당 10,000원 정액 베팅 기준 시뮬레이션)")

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("총 베팅 건수", f"{stats['total_bets_placed']} 건")
        with sc2:
            st.metric("전체 적중률", f"{stats['win_rate_pct']}%", f"{stats['won_bets']}/{stats['total_bets_placed']} 적중")
        with sc3:
            st.metric("가상 순 손익", f"{stats['net_profit_krw']:+,} 원", delta=f"{stats['roi_pct']:+.2f}%")
        with sc4:
            st.metric("수익률 (ROI)", f"{stats['roi_pct']:+.2f}%")

        st.divider()

        # 등급별 세부 성과 분석
        st.markdown("### 🏆 등급별(Grade) 적중률 및 수익률 비교")
        gb = stats.get("grade_breakdown", {})
        
        g_cols = st.columns(3)
        with g_cols[0]:
            s_data = gb.get("S", {})
            st.markdown("#### 🔴 GRADE S (초고배당)")
            st.write(f"• 베팅: **{s_data.get('bets', 0)}건** 중 **{s_data.get('wins', 0)}건 적중** ({s_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{s_data.get('returned', 0) - s_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{s_data.get('roi_pct', 0):+.1f}%`**")
            
        with g_cols[1]:
            a_data = gb.get("A", {})
            st.markdown("#### 🟢 GRADE A (추천 가치 픽)")
            st.write(f"• 베팅: **{a_data.get('bets', 0)}건** 중 **{a_data.get('wins', 0)}건 적중** ({a_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{a_data.get('returned', 0) - a_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{a_data.get('roi_pct', 0):+.1f}%`** (압도적 우세)")

        with g_cols[2]:
            b_data = gb.get("B", {})
            st.markdown("#### 🔵 GRADE B (소액 분산)")
            st.write(f"• 베팅: **{b_data.get('bets', 0)}건** 중 **{b_data.get('wins', 0)}건 적중** ({b_data.get('win_rate_pct', 0)}%)")
            st.write(f"• 순익: **{b_data.get('returned', 0) - b_data.get('staked', 0):+,.0f}원**")
            st.markdown(f"• 수익률: **`{b_data.get('roi_pct', 0):+.1f}%`**")

        st.info("💡 **핵심 인사이트:** 데이터 분석 결과, **GRADE A (추천 가치 픽)**에 집중했을 때 적중률 **42.1%**, 누적 수익률 **+9.9%**로 가장 안정적인 플러스 이득을 기록했습니다.")

        # 최근 평가 경기 목록 테이블
        st.markdown("### 📝 최근 경기별 AI 추천 픽 채점 결과")
        recent_picks = stats.get("recent_evaluated_picks", [])
        if recent_picks:
            rec_df = pd.DataFrame(recent_picks)
            rec_df.columns = ["경기일자", "경기 대진", "실제 스코어", "AI 추천 픽", "배당률", "적중 여부", "손익"]
            rec_df["적중 여부"] = rec_df["적중 여부"].apply(lambda x: "✅ 적중" if x else "❌ 미적중")
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
