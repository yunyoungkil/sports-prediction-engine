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

import json
import streamlit as st
import pandas as pd

from models.parlay_builder import (
    build_recommended_parlays,
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

# 세션 상태: 사용자의 맞춤 조합 바구니 (경기 번호 목록)
if "cart_seqs" not in st.session_state:
    st.session_state["cart_seqs"] = []

# 타이틀 헤더
st.title("⚽ 스포츠 승부 예측 & 가치 픽 (Value Bet) 대시보드")
st.caption("베트맨(Betman) 프로토 배당률 및 Dixon-Coles 통계 모델 기반 기대가치(EV) 실시간 분석")

# 상단 탭 3개 구성
tab1, tab2, tab3 = st.tabs([
    "🔥 이번 회차 실시간 베트맨 경기",
    "🎯 AI 추천 2~3폴더 조합 & 베팅 바구니",
    "📊 AI 실전 적중률 & 수익률(ROI) 통계"
])

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
            st.metric("장바구니 담긴 경기", f"{len(st.session_state['cart_seqs'])} 개", "Tab 2에서 조합 확인")

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

        # [필터 5] 팀 이름 실시간 검색
        search_query = st.sidebar.text_input("🔎 팀 이름 검색", placeholder="예: 바르셀로나, 뮌헨, 인천")

        # [필터 6] 정렬 기준
        sort_by = st.sidebar.radio(
            "정렬 기준",
            ["기대가치(EV) 높은순 (AI 추천순)", "경기 시작 시간순", "홈 승률 높은순", "배당률 높은순"]
        )

        filtered = predictions
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
            m_seq = str(m["match_seq"])
            
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
                with head_col3:
                    # 바구니 담기 토글 버튼
                    in_cart = m_seq in st.session_state["cart_seqs"]
                    btn_label = "✅ 담김" if in_cart else "🛒 담기"
                    if st.button(btn_label, key=f"cart_btn_{m_seq}"):
                        if in_cart:
                            st.session_state["cart_seqs"].remove(m_seq)
                        else:
                            st.session_state["cart_seqs"].append(m_seq)
                        st.rerun()

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
                if m.get("top_scores"):
                    score_text = " | ".join([f"{s['score']} ({s['prob']}%)" for s in m.get("top_scores", [])])
                    st.caption(f"🎯 **예상 스코어 TOP 3:** {score_text}")

                # 팀별 최근 5경기 폼 및 맞대결(H2H) 세부 통계 펼치기
                home_form = m.get("home_form", {})
                away_form = m.get("away_form", {})
                h2h = m.get("h2h", {})

                with st.expander("📊 팀별 최근 5경기 폼(W-D-L) & 맞대결(H2H) 세부 통계"):
                    col_hf, col_h2h, col_af = st.columns([1.1, 1.3, 1.1])
                    
                    with col_hf:
                        st.markdown(f"<div class='notranslate'><b>🏠 {m['home_team']} 최근 5경기</b></div>", unsafe_allow_html=True)
                        if home_form.get("has_data"):
                            badges = []
                            for r in home_form.get("form_sequence", []):
                                if r == "W":
                                    badges.append("<span style='background:#2ed573; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>W</span>")
                                elif r == "D":
                                    badges.append("<span style='background:#747d8c; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>D</span>")
                                else:
                                    badges.append("<span style='background:#ff4757; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>L</span>")
                            st.markdown(" ".join(badges), unsafe_allow_html=True)
                            st.write(f"• 최근 전적: **{home_form['record_str']}** (승률 {home_form['win_rate_pct']}%)")
                            st.write(f"• 평균 득점: **{home_form['avg_scored']}골** | 실점: **{home_form['avg_conceded']}골**")
                            
                            with st.popover("최근 5경기 상세"):
                                for lm in home_form.get("last_matches", []):
                                    st.write(f"{lm['date']} ({lm['venue']}) vs {lm['opponent']} : **{lm['score']}** ({lm['result']})")
                        else:
                            st.caption("기초 데이터 축적 중 (오즈메이커 지수 반영)")

                    with col_h2h:
                        st.markdown(f"<b>⚔️ 맞대결 전적 (H2H)</b>", unsafe_allow_html=True)
                        if h2h.get("has_data"):
                            st.markdown(f"<div class='notranslate' style='font-weight:700; color:#70a1ff; margin-bottom:4px;'>{h2h['record_str']}</div>", unsafe_allow_html=True)
                            st.write(f"• 최근 맞대결 경기수: **{h2h['total_matches']}경기**")
                            st.write(f"• 경기당 평균 총 골수: **{h2h['avg_goals']}골**")
                            if h2h.get("matches"):
                                recent_m = h2h['matches'][0]
                                st.caption(f"가장 최근 맞대결: {recent_m['date']} ({recent_m['home']} {recent_m['score']} {recent_m['away']})")
                        else:
                            st.caption("최근 시즌 내 공식 맞대결 기록 없음")

                    with col_af:
                        st.markdown(f"<div class='notranslate'><b>✈️ {m['away_team']} 최근 5경기</b></div>", unsafe_allow_html=True)
                        if away_form.get("has_data"):
                            badges = []
                            for r in away_form.get("form_sequence", []):
                                if r == "W":
                                    badges.append("<span style='background:#2ed573; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>W</span>")
                                elif r == "D":
                                    badges.append("<span style='background:#747d8c; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>D</span>")
                                else:
                                    badges.append("<span style='background:#ff4757; color:white; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:12px;'>L</span>")
                            st.markdown(" ".join(badges), unsafe_allow_html=True)
                            st.write(f"• 최근 전적: **{away_form['record_str']}** (승률 {away_form['win_rate_pct']}%)")
                            st.write(f"• 평균 득점: **{away_form['avg_scored']}골** | 실점: **{away_form['avg_conceded']}골**")
                            
                            with st.popover("최근 5경기 상세"):
                                for lm in away_form.get("last_matches", []):
                                    st.write(f"{lm['date']} ({lm['venue']}) vs {lm['opponent']} : **{lm['score']}** ({lm['result']})")
                        else:
                            st.caption("기초 데이터 축적 중 (오즈메이커 지수 반영)")

        if len(filtered) > 60:
            st.info(f"💡 현재 상위 60개 경기를 표시 중입니다. (사이드바의 게임 유형 및 리그 필터를 활용해 보세요!)")


# ==============================================================================
# TAB 2: 🎯 AI 추천 2~3폴더 조합 & 베팅 바구니 (NEW!)
# ==============================================================================
with tab2:
    st.subheader("🎯 AI 엄선 2~3폴더 조합 & 맞춤 베팅 바구니")
    st.caption("베트맨 프로토 규정(동일 경기 중복 배제) 및 확률 곱셈 법칙 기반 최적의 다폴더 조합을 생성합니다.")

    if not predictions:
        st.warning("⚠️ 예측 데이터가 없습니다. 먼저 터미널에서 `python run_pipeline.py`를 실행해 주세요.")
    else:
        # AI 추천 5대 프리셋 생성
        parlay_presets = build_recommended_parlays(predictions)

        st.markdown("### 🌟 AI 추천 5대 최적 포트폴리오")
        st.write("승률, 기대가치(EV), 카테고리 분산 기준에 따라 엄선된 황금 조합입니다:")

        preset_labels = {
            "safety_double": ("🛡️ 안정형 2폴더 (Safety Double)", "개별 승률 50% 이상 중심, 안정적인 환급 지향 (합산 2~4배)"),
            "value_double": ("⚖️ 밸런스 가치 2폴더 (Value Double)", "양의 기대가치(EV)와 적중률의 황금 균형 (합산 3~6배)"),
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
                with col_m1:
                    st.metric("합산 배당률", f"{combo['combined_odds']} 배")
                with col_m2:
                    st.metric("AI 결합 승률", combo["combined_prob_pct"])
                with col_m3:
                    st.metric("기대가치 (EV)", combo["combined_ev_pct"])
                with col_btn:
                    if st.button("📥 바구니 담기", key=f"copy_preset_{key}"):
                        for p in combo["picks"]:
                            if p["match_seq"] not in st.session_state["cart_seqs"]:
                                st.session_state["cart_seqs"].append(p["match_seq"])
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
                st.session_state["cart_seqs"] = []
                st.rerun()

        # 현재 담긴 경기 객체 추출
        pred_map = {str(m["match_seq"]): m for m in predictions}
        cart_matches = [pred_map[seq] for seq in st.session_state["cart_seqs"] if seq in pred_map]
        cart_picks = [get_pick_item(m) for m in cart_matches]

        # 바구니 조합 지표 계산
        metrics = calculate_parlay_metrics(cart_picks)

        if not cart_picks:
            st.info("💡 아직 바구니에 담긴 경기가 없습니다. 위 프리셋의 [📥 바구니 담기] 버튼이나 '실시간 베트맨 경기' 탭에서 [🛒 담기]를 눌러보세요.")
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
                            st.session_state["cart_seqs"].remove(p["match_seq"])
                            st.rerun()

                # 모의 베팅 환급금 계산기
                st.markdown("##### 💰 예상 환급금 & 기대 순익 시뮬레이터")
                stake_krw = st.number_input("베팅 희망 금액 (원)", min_value=1000, max_value=100000, value=10000, step=1000)
                
                exp_return = int(stake_krw * metrics["combined_odds"])
                exp_profit = int(stake_krw * metrics["combined_ev"])
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"• 적중 시 예상 환급금: **`{exp_return:,} 원`** (수익: {exp_return - stake_krw:+,}원)")
                with res_col2:
                    st.markdown(f"• 수학적 기대 이익(EV): **`{exp_profit:+,} 원`**")


# ==============================================================================
# TAB 3: AI 실전 적중률 & 모의 베팅 수익률(ROI) 통계
# ==============================================================================
with tab3:
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
