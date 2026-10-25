"""
collector/betman_results_scraper.py
===================================
베트맨(Betman) 프로토 승부식(G101) 지난 회차들의 실제 경기 결과 및 스코어를 
공식 API(/gamebuy/winrst/inqWinrstDetlBody.do)로부터 수집합니다.
- 최근 N개 회차의 실제 경기 결과(스코어, 승/무/패, 배당) 일괄 수집
- K리그, 유럽리그, J리그 등 베트맨 발매 전 경기 한글 팀명 기준 자동 축적
"""

import json
import sys
import http.cookiejar
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "https://www.betman.co.kr"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class BetmanResultsScraper:
    def __init__(self, gm_id: str = "G101"):
        self.gm_id = gm_id
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        self.headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"
        }

    def fetch_round_results(self, gm_ts: int) -> List[Dict[str, Any]]:
        """특정 회차의 경기 결과 및 실제 스코어 목록 수집"""
        # 세션 쿠키 갱신
        init_url = f"{BASE_URL}/main/mainPage/gamebuy/winrstDetl.do?gmId={self.gm_id}&gmTs={gm_ts}"
        try:
            req_init = urllib.request.Request(init_url, headers={"User-Agent": USER_AGENT})
            self.opener.open(req_init, timeout=8)
        except Exception:
            pass

        api_url = f"{BASE_URL}/gamebuy/winrst/inqWinrstDetlBody.do"
        payload = {
            "gmId": self.gm_id,
            "gmTs": gm_ts,
            "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}
        }

        try:
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self.headers,
                method="POST"
            )
            res = self.opener.open(req, timeout=10)
            data = json.loads(res.read().decode("utf-8"))
            raw_items = data.get("detlBody", [])
        except Exception as e:
            print(f"⚠️ {gm_ts}회차 결과 수집 실패: {e}")
            return []

        results = []
        for item in raw_items:
            score = item.get("MCH_SCORE")
            if not score or score == "-":
                continue  # 취소되거나 미개최 경기 제외

            game_type = item.get("BETTYP_NM", "")
            league = item.get("PR_LL_WIN_LEAG_NM", item.get("LEAG_CD_NM", "기타"))
            home_team = item.get("HOME_TEAM", "").strip()
            away_team = item.get("AWAY_TEAM", "").strip()
            match_dtm = item.get("MCH_DTM", "")
            win_txt = item.get("WIN_TXT", "")

            # 스코어 파싱 (예: "4:0" -> 4, 0)
            h_score, a_score = None, None
            if ":" in score:
                parts = score.split(":")
                try:
                    h_score = int(parts[0].strip())
                    a_score = int(parts[1].strip())
                except Exception:
                    pass

            results.append({
                "gm_ts": gm_ts,
                "match_seq": str(item.get("GM_SEQ", "")),
                "sport": item.get("MCH_SPORT_CD", "SC"),
                "league": league,
                "game_type": game_type,
                "bettyp_typ_nm": item.get("BETTYP_TYP_NM", ""),
                "game_result": item.get("GAME_RESULT", ""),
                "handi_val": item.get("HANDI_VAL", 0),
                "match_time": match_dtm,
                "home_team": home_team,
                "away_team": away_team,
                "score": score,
                "home_score": h_score,
                "away_score": a_score,
                "result_text": win_txt,
                "odds_win": item.get("ODDS_WIN", 0.0),
                "odds_draw": item.get("ODDS_DRAW", 0.0),
                "odds_lose": item.get("ODDS_LOSE", 0.0)
            })

        return results

    def sync_recent_results(
        self,
        current_ts: int = 260126,
        n_rounds: int = 8,
        filepath: str = "data/history/betman_recent_results.json"
    ) -> List[Dict[str, Any]]:
        """
        기존 결과 파일(betman_recent_results.json)을 로드하여,
        아직 수집되지 않은 최근 종료 회차만 API로 추가 수집하고 중복 없이(gm_ts + match_seq) 병합 저장합니다.
        """
        path = Path(filepath)
        existing_results: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            except Exception as e:
                print(f"⚠️ 기존 결과 파일 읽기 실패 ({e}), 새로 생성합니다.")
                existing_results = []

        # 기존 수집된 회차 및 유니크 키 맵 구성
        existing_rounds = set()
        existing_map = {}
        for item in existing_results:
            ts = int(item.get("gm_ts", 0)) if str(item.get("gm_ts", "")).isdigit() else item.get("gm_ts")
            existing_rounds.add(ts)
            key = (str(item.get("gm_ts")), str(item.get("match_seq")))
            existing_map[key] = item

        print(f"📂 기존 실전 경기 결과: {len(existing_results)}건 (수집 회차: {sorted(list(existing_rounds))})")

        # 확인할 최근 N개 회차 목록
        target_rounds = [current_ts - i for i in range(1, n_rounds + 1)]
        missing_rounds = [ts for ts in target_rounds if ts not in existing_rounds]

        if not missing_rounds:
            print("  ✓ 최근 종료 회차 데이터가 이미 모두 수집되어 최신 상태입니다.")
            return existing_results

        print(f"📥 신규 미수집 회차 발견: {missing_rounds} -> 결과 수집 시작...")
        new_added = 0
        for ts in missing_rounds:
            r_list = self.fetch_round_results(ts)
            print(f"  ✓ {ts}회차: 유효 종료 경기 {len(r_list)}건 수집 완료")
            for item in r_list:
                key = (str(item.get("gm_ts")), str(item.get("match_seq")))
                if key not in existing_map:
                    existing_map[key] = item
                    existing_results.append(item)
                    new_added += 1

        # 회차 및 경기 순번 기준 정렬
        def sort_key(x):
            try:
                return (int(x.get("gm_ts", 0)), int(x.get("match_seq", 0)))
            except Exception:
                return (0, 0)

        existing_results.sort(key=sort_key)
        self.save_results(existing_results, filepath=filepath)
        print(f"✅ 신규 {new_added}건 병합 완료 (총 누적: {len(existing_results)}건)")
        return existing_results

    def fetch_recent_rounds(self, current_ts: int = 260126, n_rounds: int = 4) -> List[Dict[str, Any]]:
        """현재 회차 직전 N개 회차의 결과를 일괄 수집"""
        all_results = []
        for i in range(1, n_rounds + 1):
            ts = current_ts - i
            print(f"📥 베트맨 프로토 승부식 {ts}회차 결과 수집 중...")
            r_list = self.fetch_round_results(ts)
            print(f"  ✓ {ts}회차: 유효 종료 경기 {len(r_list)}건 수집 완료")
            all_results.extend(r_list)
        return all_results

    def save_results(self, results: List[Dict[str, Any]], filepath: str = "data/history/betman_recent_results.json"):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 베트맨 최근 경기 결과 {len(results)}건 저장 완료: {filepath}")


if __name__ == "__main__":
    scraper = BetmanResultsScraper()
    # 최근 종료 회차 누적 동기화
    scraper.sync_recent_results(current_ts=260126, n_rounds=8)
