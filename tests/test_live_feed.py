"""대시보드 실시간 받기 (collector/live_feed.py) — 가짜 베트맨·가짜 시계로: 기다림 · 뒤에서 받기 · 요청 수 · 실패 시 쉬기"""
import threading

from collector.live_feed import LiveFeed


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


def fake_betman(state):
    """베트맨 대신: 접속(세션) 수 · 요청 순서를 state에 남김"""
    class Scraper:
        def __init__(self):
            state["sessions"] += 1

        def find_current_round(self):
            state["calls"].append("round")
            return state["round"]

        def fetch_current_round_matches(self, gm_ts):
            state["calls"].append("list")
            if state.get("gate"):                 # 진짜 베트맨처럼 받는 데 시간이 걸리는 상황
                state["gate"].wait(5)
            if state["empty"]:
                state["empty"] -= 1
                return []
            return [{"match_seq": "1", "close_time": state["close_ms"], "odds": dict(state["odds"])}]
    return Scraper


def predict(matches, ts, models, sharp):
    return [{**m, "betman_odds": m["odds"], "gm_ts": ts, "sharp": sharp.get(m["match_seq"])} for m in matches]


def setup():
    clock = Clock()
    state = {"sessions": 0, "calls": [], "round": "260128", "empty": 0,
             "close_ms": (clock.t + 6 * 3600) * 1000, "odds": {"home_win": 1.5}}
    feed = LiveFeed(clock=clock)
    file_preds = [{"match_seq": "1", "gm_ts": 260128, "sharp": {"home_win": 1.6}}]
    return clock, state, feed, (fake_betman(state), predict, None, file_preds)


def wait(feed):
    feed.thread.join(5)


def test_never_waits_first_load_shows_saved_data_then_live():
    clock, state, feed, args = setup()
    state["gate"] = threading.Event()
    assert feed.get(*args) is None            # 처음: 기다리지 않음 → 화면은 저장된 데이터, 뒤에서 받기 시작
    state["gate"].set()
    wait(feed)
    state["gate"] = None
    d = feed.get(*args)
    assert (d["ts"], d["preds"][0]["sharp"]) == (260128, {"home_win": 1.6})
    assert (state["calls"], state["sessions"]) == (["round", "list"], 1)

    clock.t += 30                             # 1분 안: 받지 않고 그대로
    assert feed.get(*args) is d and state["calls"] == ["round", "list"]

    clock.t += 40                             # 1분 지남: 지난 데이터를 바로 주고, 뒤에서 새로 받음
    state["gate"] = threading.Event()
    assert feed.get(*args) is d and feed.busy()
    state["gate"].set()
    wait(feed)
    assert feed.data["got"] == clock.t
    # 같은 접속(세션)으로 경기 목록만 다시 (회차 확인은 10분마다) → 요청 1번
    assert (state["calls"], state["sessions"]) == (["round", "list", "list"], 1)


def test_round_rechecked_after_10_minutes_or_when_everything_closed():
    clock, state, feed, args = setup()
    feed.get(*args); wait(feed)
    clock.t += 11 * 60
    assert feed.get(*args) is not None        # 11분 묵은 데이터도 30분 안이면 바로 보여 줌 (시각 표시로 구분)
    wait(feed)
    assert state["calls"] == ["round", "list", "round", "list"]   # 10분 지나 회차도 다시 확인

    state["close_ms"] = (clock.t - 60) * 1000   # 경기가 모두 마감 → 다음 회차가 열렸는지 바로 확인
    clock.t += 61
    feed.get(*args); wait(feed)               # 이번에 받은 목록이 '모두 마감'
    clock.t += 61
    feed.get(*args); wait(feed)
    assert state["calls"][-2:] == ["round", "list"]


def test_new_session_once_when_list_comes_back_empty():
    clock, state, feed, args = setup()
    state["empty"] = 1                        # 세션이 끊긴 것처럼 빈 목록 → 새 접속으로 한 번 더
    feed.get(*args); wait(feed)
    assert feed.get(*args) is not None
    assert (state["calls"], state["sessions"]) == (["round", "list", "round", "list"], 2)


def test_failure_waits_a_minute_and_old_data_expires():
    clock, state, feed, args = setup()
    state["round"] = None                     # 베트맨 접속 실패 (회차 조회부터 안 됨) → 한 번만 시도
    feed.get(*args); wait(feed)
    assert feed.get(*args) is None and state["calls"] == ["round"] and feed.failed_recently()
    clock.t += 30                             # 1분 동안은 다시 시도하지 않음
    feed.get(*args)
    assert not feed.busy() and state["calls"] == ["round"]
    clock.t += 31
    state["round"] = "260128"
    feed.get(*args); wait(feed)               # 1분 뒤 다시 시도 → 성공
    assert feed.get(*args) is not None

    clock.t += 31 * 60                        # 30분 넘게 못 받음 → 낡은 실시간 대신 저장된 데이터(None)
    state["round"], feed.round = None, None
    assert feed.get(*args) is None


def test_screen_redraws_only_when_odds_or_state_change():
    from collector.live_feed import changed_matches
    clock, state, feed, args = setup()
    feed.get(*args); wait(feed)
    first = feed.data
    clock.t += 61
    feed.get(*args); wait(feed)
    assert feed.data["sig"] == first["sig"] and changed_matches(first, feed.data) == 0
    state["odds"]["home_win"] = 1.45
    clock.t += 61
    feed.get(*args); wait(feed)
    assert feed.data["sig"] != first["sig"] and changed_matches(first, feed.data) == 1
    assert changed_matches(None, feed.data) == 1 and changed_matches(first, None) == 0
