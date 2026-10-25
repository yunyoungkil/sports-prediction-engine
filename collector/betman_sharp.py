"""
collector/betman_sharp.py
=========================
[Phase 5.4] 베트맨 판매 경기 ↔ 해외 배당(The Odds API) 짝짓기 및 회차별 스냅샷 저장.

- 베트맨 리그명 → The Odds API sport key 매핑
- 리그당 1회 호출 (bookmakers 10개 이하 지정 시 호출당 1크레딧) → 회차당 약 10크레딧
- 짝짓기: 킥오프 시각(±75분) + 팀명 유사도(team_name_map 영문명) + 배당 확률 거리
- 결과: data/history/sharp_snapshots/{gm_ts}.json 에 실행 시각별 누적 저장
  (나중에 evaluation/sharp_vs_betman.py 가 실전 결과와 붙여 ROI를 검증)

API 키: 환경변수 ODDS_API_KEY (Windows는 사용자 환경변수도 확인)
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.odds_normalizer import remove_margin

API_BASE = "https://api.the-odds-api.com/v4/sports"
SNAPSHOT_DIR = Path("data/history/sharp_snapshots")
MIN_REMAINING_CREDITS = 30   # 남은 크레딧이 이보다 적으면 조회 중단 (월 500회 무료 한도 보호)
# 같은 회차는 이 시간 안에 다시 조회하지 않고 직전 스냅샷을 재사용 (하루 2회 실행 × 리그 수 → 월 한도 초과 방지)
MIN_REFETCH_HOURS = float(os.getenv("SHARP_MIN_REFETCH_HOURS", "20"))
KST = timezone(timedelta(hours=9))

# 베트맨 리그명(부분 일치) → The Odds API sport key
BETMAN_LEAGUE_TO_SPORT_KEY = [
    ("프리미어리그", "soccer_epl"),
    ("챔피언십", "soccer_efl_champ"),
    ("라리가", "soccer_spain_la_liga"),
    ("분데스리가", "soccer_germany_bundesliga"),
    ("세리에A", "soccer_italy_serie_a"),
    ("리그1", "soccer_france_ligue_one"),
    ("에레디비시", "soccer_netherlands_eredivisie"),
    ("메이저리그사커", "soccer_usa_mls"),
    ("J1리그", "soccer_japan_j_league"),
    ("A리그", "soccer_australia_aleague"),
    ("코파 리베르타도레스", "soccer_conmebol_copa_libertadores"),
    ("챔피언스리그", "soccer_uefa_champs_league"),
    ("유로파리그", "soccer_uefa_europa_league"),
    ("프리메이라리가", "soccer_portugal_primeira_liga"),
]
# K리그·J2·중국 슈퍼리그는 The Odds API 미지원

# 샤프 우선순위 + 시장 평균용 북메이커 (10개 이하 → 호출당 1크레딧)
SHARP_BOOKMAKERS = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk"]
CONSENSUS_BOOKMAKERS = SHARP_BOOKMAKERS + ["williamhill", "unibet_eu", "marathonbet", "onexbet", "betsson", "sport888", "matchbook"]


def get_api_key() -> Optional[str]:
    key = os.getenv("ODDS_API_KEY")
    if not key and sys.platform.startswith("win"):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "ODDS_API_KEY")[0]
        except OSError:
            key = None
    return key or None


def sport_key_for(league: str) -> Optional[str]:
    if not league or "K리그" in league:
        return None
    for token, key in BETMAN_LEAGUE_TO_SPORT_KEY:
        if token in league:
            # "리그1"이 "J1리그"·"K리그1" 등에 걸리지 않게 프랑스만 허용
            if token == "리그1" and "프랑스" not in league:
                continue
            if token == "챔피언스리그" and ("AFC" in league or "아시아" in league):
                continue
            return key
    return None


def parse_betman_kickoff(match_time: str) -> Optional[datetime]:
    """'26.10.24 (토) 03:00' (KST) → UTC datetime"""
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{2}).*?(\d{1,2}):(\d{2})", str(match_time or ""))
    if not m:
        return None
    yy, mo, dd, hh, mi = (int(x) for x in m.groups())
    return datetime(2000 + yy, mo, dd, hh, mi, tzinfo=KST).astimezone(timezone.utc)


def _norm_name(s: str) -> str:
    s = re.sub(r"\b(fc|cf|sc|afc|ac|cd|club|de|the)\b", " ", str(s).lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def _name_sim(a: Optional[str], b: str) -> float:
    if not a:
        return 0.0
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _fair(odds: Dict[str, float]) -> Optional[Dict[str, float]]:
    h, d, a = odds.get("home_win"), odds.get("draw"), odds.get("away_win")
    if not all(isinstance(x, (int, float)) and x > 1.0 for x in (h, d, a)):
        return None
    p = remove_margin(h, d, a)["true_probs"]
    return {"home_win": p["home"], "draw": p["draw"], "away_win": p["away"]}


def _h2h_odds(bookmaker: Dict[str, Any], home: str, away: str) -> Optional[Dict[str, float]]:
    for mk in bookmaker.get("markets", []):
        if mk.get("key") != "h2h":
            continue
        o = {x.get("name"): x.get("price") for x in mk.get("outcomes", [])}
        if o.get(home) and o.get(away) and o.get("Draw"):
            return {"home_win": float(o[home]), "draw": float(o["Draw"]), "away_win": float(o[away])}
    return None


def summarize_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """API 이벤트 1개 → 샤프 배당(우선순위 1개) + 시장 평균 확률"""
    home, away = ev.get("home_team", ""), ev.get("away_team", "")
    books = {b.get("key"): _h2h_odds(b, home, away) for b in ev.get("bookmakers", [])}
    books = {k: v for k, v in books.items() if v}
    if not books:
        return None
    sharp_bm = next((k for k in SHARP_BOOKMAKERS if k in books), None)
    fair_all = [f for f in (_fair(v) for v in books.values()) if f]
    consensus = None
    if fair_all:
        consensus = {k: sum(f[k] for f in fair_all) / len(fair_all) for k in ("home_win", "draw", "away_win")}
    return {
        "event_id": ev.get("id"),
        "home_team": home,
        "away_team": away,
        "commence_time": ev.get("commence_time"),
        "sharp_bookmaker": sharp_bm,
        "sharp_odds": books.get(sharp_bm) if sharp_bm else None,
        "sharp_probs": _fair(books[sharp_bm]) if sharp_bm else None,
        "consensus_probs": consensus,
        "n_bookmakers": len(books),
    }


def fetch_league_events(sport_key: str, api_key: str) -> Dict[str, Any]:
    params = {
        "apiKey": api_key,
        "markets": "h2h",
        "oddsFormat": "decimal",
        "bookmakers": ",".join(CONSENSUS_BOOKMAKERS),
    }
    url = f"{API_BASE}/{sport_key}/odds?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SportsPredictionEngine/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        data = json.loads(resp.read().decode("utf-8"))
    return {"events": data, "remaining": int(float(remaining)) if remaining is not None else None}


def match_betman_to_events(betman: Dict[str, Any], events: List[Dict[str, Any]], team_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    베트맨 경기 1개에 해당하는 해외 이벤트를 찾습니다. 확신이 없으면 None.
    1) 킥오프 ±75분 후보만
    2) 팀명(영문 매핑) 유사도가 홈·원정 모두 0.6 이상이면 확정
    3) 아니면 배당 확률 거리로: 후보 1개면 거리 ≤ 0.25, 여러 개면 최소 거리 ≤ 0.12 이고 2위와 0.05 이상 차이
    """
    ko = parse_betman_kickoff(betman.get("match_time"))
    p_bm = _fair(betman.get("odds", {}))
    if ko is None or p_bm is None:
        return None
    cands = []
    for ev in events:
        try:
            ct = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if abs((ct - ko).total_seconds()) > 75 * 60 or not ev.get("sharp_probs") and not ev.get("consensus_probs"):
            continue
        ref = ev.get("sharp_probs") or ev.get("consensus_probs")
        dist = sum(abs(ref[k] - p_bm[k]) for k in ("home_win", "draw", "away_win"))
        sim = min(_name_sim(team_map.get(betman.get("home_team")), ev["home_team"]),
                  _name_sim(team_map.get(betman.get("away_team")), ev["away_team"]))
        cands.append((ev, dist, sim))
    if not cands:
        return None

    by_name = [c for c in cands if c[2] >= 0.6]
    if len(by_name) == 1:
        return {**by_name[0][0], "match_method": "name", "prob_distance": round(by_name[0][1], 4)}

    cands.sort(key=lambda c: c[1])
    best = cands[0]
    if len(cands) == 1:
        return {**best[0], "match_method": "time_single", "prob_distance": round(best[1], 4)} if best[1] <= 0.25 else None
    if best[1] <= 0.12 and cands[1][1] - best[1] >= 0.05:
        return {**best[0], "match_method": "time_odds", "prob_distance": round(best[1], 4)}
    return None


def collect_round_sharp_odds(
    gm_ts: Any,
    matches: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    team_map_path: str = "data/team_name_map.json",
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    회차의 축구 3-way 경기에 해외 배당을 붙여 {match_seq: 정보}를 반환하고 스냅샷으로 저장합니다.
    키가 없거나 크레딧이 부족하면 빈 dict.
    """
    api_key = api_key or get_api_key()
    if not api_key:
        print("ℹ [5.4] ODDS_API_KEY 미설정 → 해외 배당 비교 건너뜀")
        return {}

    now = datetime.now(timezone.utc)
    targets = []
    for m in matches:
        if m.get("sport") != "SC" or m.get("game_type") != "축구 승무패":
            continue
        if not float((m.get("odds") or {}).get("draw") or 0.0) > 1.0:
            continue
        ko = parse_betman_kickoff(m.get("match_time"))
        key = sport_key_for(m.get("league", ""))
        if key and ko and ko > now:
            targets.append((m, key))
    if not targets:
        return {}

    # 최근 스냅샷이 충분히 새것이면 API를 부르지 않고 재사용
    path = snapshot_dir / f"{gm_ts}.json"
    if path.exists() and not force:
        try:
            prev = json.loads(path.read_text(encoding="utf-8")).get("snapshots", [])
            if prev:
                last = prev[-1]
                age_h = (now - datetime.fromisoformat(last["captured_at"])).total_seconds() / 3600
                if age_h < MIN_REFETCH_HOURS:
                    want = {str(m["match_seq"]) for m, _ in targets}
                    reused = {k: v for k, v in last.get("matches", {}).items() if k in want}
                    print(f"🌍 [5.4] 해외 배당: {age_h:.1f}시간 전 스냅샷 재사용 ({len(reused)}경기, API 호출 없음)")
                    return reused
        except Exception:
            pass

    try:
        team_map = json.loads(Path(team_map_path).read_text(encoding="utf-8"))
    except Exception:
        team_map = {}

    events_by_key: Dict[str, List[Dict[str, Any]]] = {}
    remaining = None
    for key in sorted({k for _, k in targets}):
        if remaining is not None and remaining < MIN_REMAINING_CREDITS:
            print(f"⚠️ [5.4] 남은 크레딧 {remaining} < {MIN_REMAINING_CREDITS} → 나머지 리그 조회 중단")
            break
        try:
            res = fetch_league_events(key, api_key)
        except Exception as e:
            print(f"⚠️ [5.4] {key} 조회 실패: {e}")
            continue
        remaining = res["remaining"]
        events_by_key[key] = [s for s in (summarize_event(ev) for ev in res["events"]) if s]

    matched: Dict[str, Dict[str, Any]] = {}
    for m, key in targets:
        ev = match_betman_to_events(m, events_by_key.get(key, []), team_map)
        if not ev:
            continue
        ref = ev.get("sharp_probs") or ev.get("consensus_probs")
        odds = m["odds"]
        matched[str(m["match_seq"])] = {
            **ev,
            "sport_key": key,
            "betman_odds": odds,
            "betman_home": m.get("home_team"),
            "betman_away": m.get("away_team"),
            "match_time": m.get("match_time"),
            # 해외 기준 확률로 본 베트맨 배당의 기대값
            "betman_ev_vs_sharp": {k: round(ref[k] * odds[k] - 1.0, 4) for k in ("home_win", "draw", "away_win")},
        }

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    data = {"gm_ts": gm_ts, "snapshots": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["snapshots"].append({
        "captured_at": now.isoformat(),
        "credits_remaining": remaining,
        "targets": len(targets),
        "matches": matched,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🌍 [5.4] 해외 배당 짝짓기 {len(matched)}/{len(targets)}경기 (리그 {len(events_by_key)}개 조회, 남은 크레딧 {remaining}) → {path}")
    return matched
