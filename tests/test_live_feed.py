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


def test_first_load_waits_then_refreshes_in_background_with_one_request():
    clock, state, feed, args = setup()
    assert feed.must_wait()
    d = feed.get(*args)                       # 처음: 기다려서 받음 (회차 확인 + 경기 목록)
    assert (d["ts"], d["preds"][0]["sharp"]) == (260128, {"home_win": 1.6})
    assert (state["calls"], state["sessions"]) == (["round", "list"], 1)

    clock.t += 30                             # 1분 안: 받지 않고 그대로
    assert feed.get(*args) is d and state["calls"] == ["round", "list"]

    clock.t += 40                             # 1분 지남: 기다리지 않고 지난 데이터를 바로 주고, 뒤에서 새로 받음
    assert not feed.must_wait()
    state["gate"] = threading.Event()
    assert feed.get(*args) is d
    state["gate"].set()
    feed.thread.join(5)
    assert feed.data["got"] == clock.t
    # 같은 접속(세션)으로 경기 목록만 다시 (회차 확인은 10분마다) → 요청 1번
    assert (state["calls"], state["sessions"]) == (["round", "list", "list"], 1)


def test_round_rechecked_after_10_minutes_or_when_everything_closed():
    clock, state, feed, args = setup()
    feed.get(*args)
    clock.t += 11 * 60                        # 오래 쉼(3분 넘음) → 기다려서 받음 · 10분 지나 회차도 다시 확인
    assert feed.must_wait()
    feed.get(*args)
    assert state["calls"] == ["round", "list", "round", "list"]

    state["close_ms"] = (clock.t - 60) * 1000   # 이번 목록의 경기가 모두 마감 → 다음 회차가 열렸는지 바로 확인
    feed.get(*args)                           # (방금 받은 목록에는 아직 열린 경기가 있었음 → 다음 새로고침부터)
    clock.t += 61
    feed.get(*args)
    feed.thread.join(5)
    clock.t += 61
    feed.get(*args)
    feed.thread.join(5)
    assert state["calls"][-2:] == ["round", "list"]


def test_new_session_once_when_list_comes_back_empty():
    clock, state, feed, args = setup()
    state["empty"] = 1                        # 세션이 끊긴 것처럼 빈 목록 → 새 접속으로 한 번 더
    d = feed.get(*args)
    assert d is not None
    assert (state["calls"], state["sessions"]) == (["round", "list", "round", "list"], 2)


def test_failure_waits_a_minute_and_falls_back():
    clock, state, feed, args = setup()
    state["round"] = None                     # 베트맨 접속 실패 (회차 조회부터 안 됨) → 한 번만 시도
    assert feed.get(*args) is None
    assert state["calls"] == ["round"]
    clock.t += 30                             # 1분 동안은 다시 시도하지 않음 (클릭마다 오래 기다리지 않게)
    assert not feed.must_wait()
    assert feed.get(*args) is None and state["calls"] == ["round"]
    clock.t += 31
    state["round"] = "260128"
    assert feed.get(*args) is not None        # 1분 뒤 다시 시도 → 성공

    clock.t += 200                            # 받은 지 3분 넘었는데 또 실패 → 낡은 데이터 대신 저장된 데이터(None)
    state["round"], feed.round = None, None
    assert feed.get(*args) is None


def test_screen_redraws_only_when_odds_or_state_change():
    clock, state, feed, args = setup()
    first = feed.get(*args)["sig"]
    clock.t += 61
    feed.get(*args)
    feed.thread.join(5)
    assert feed.data["sig"] == first          # 같은 배당 → 다시 그릴 필요 없음
    state["odds"]["home_win"] = 1.45
    clock.t += 61
    feed.get(*args)
    feed.thread.join(5)
    assert feed.data["sig"] != first          # 배당이 바뀜 → 다시 그림
