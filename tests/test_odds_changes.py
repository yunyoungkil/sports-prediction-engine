"""베트맨 '배당 변경' 내역 (경기 목록 응답의 tooltipList) — 꺼내기 · 배당 기록에 저장"""
import json
from datetime import datetime, timezone

from collector.betman_scraper import parse_odds_changes
from tracker.odds_history import load_history, update_odds_history


def _row(seq, at, b, a, ts=260128, deleted="N"):
    return {"GM_TS": ts, "GM_SEQ": seq, "CHG_DTM": at, "DEL_YN": deleted,
            "BCHG_W_ODDS": b[0], "BCHG_D_ODDS": b[1], "BCHG_L_ODDS": b[2],
            "ACHG_W_ODDS": a[0], "ACHG_D_ODDS": a[1], "ACHG_L_ODDS": a[2], "BCHG_W_HANDI_RT": 1, "ACHG_W_HANDI_RT": 1}


TOOLTIP = [  # 2026-10-28 1013 미토 홀리호크 vs 세레소 오사카 (베트맨 팝업과 같은 값) — 순서가 섞여 와도 시간순으로
    _row(1013, "20261028170733221676", (360, 325, 177), (420, 325, 166)),
    _row(1013, "20261028165310222034", (315, 320, 193), (360, 325, 177)),
    _row(1013, "20261028180000000000", (420, 325, 166), (430, 325, 160), deleted="Y"),   # 지워진 기록은 뺌
    _row(999, "20261020120000000000", (200, 300, 300), (210, 300, 290), ts=260120),      # 다른 회차는 뺌
]


def test_parse_changes_like_betman_popup():
    ch = parse_odds_changes(TOOLTIP, 260128)
    assert list(ch) == ["1013"]
    first, last = ch["1013"]
    assert first["at"] == "2026-10-28 16:53:10" and last["at"] == "2026-10-28 17:07:33"
    assert first["from"] == {"home_win": 3.15, "draw": 3.2, "away_win": 1.93}       # 1차 (처음) 배당
    assert last["to"] == {"home_win": 4.2, "draw": 3.25, "away_win": 1.66}          # 현재 배당
    assert first["line_from"] == 1.0
    assert parse_odds_changes(None) == {}


def test_changes_saved_in_odds_history_and_kept_after_kickoff(tmp_path):
    ch = parse_odds_changes(TOOLTIP, 260128)["1013"]
    m = {"match_seq": "1013", "odds": {"home_win": 4.2, "draw": 3.25, "away_win": 1.66}, "game_type": "축구 승무패",
         "kickoff_ts": datetime(2026, 10, 28, 10, 0, tzinfo=timezone.utc).timestamp() * 1000, "odds_changes": ch[:1]}
    update_odds_history(260128, [m], now=datetime(2026, 10, 28, 8, 0, tzinfo=timezone.utc), hist_dir=tmp_path)
    assert load_history(260128, tmp_path)["1013"]["changes"] == ch[:1]
    # 경기가 시작된 뒤에 받은 응답에도 전체 내역이 오면 갱신 (배당 점은 더 안 찍음)
    update_odds_history(260128, [{**m, "odds_changes": ch}], now=datetime(2026, 10, 28, 11, 0, tzinfo=timezone.utc),
                        hist_dir=tmp_path)
    e = load_history(260128, tmp_path)["1013"]
    assert e["changes"] == ch and len(e["points"]) == 1
    json.dumps(e)   # 파일에 그대로 쓸 수 있는 값
