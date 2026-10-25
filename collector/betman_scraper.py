"""
betman_scraper.py
=================
베트맨(Betman) 사이트에서 프로토 승부식(G101) 최신 회차의 
실시간 경기 목록, 대진(홈팀 vs 원정팀), 경기 일시, 승/무/패 배당률, 
게임 유형(일반, 핸디캡, 언더오버, 전반전), 한경기(단통) 여부를 수집합니다.
"""

import json
import sys
import http.cookiejar
import urllib.request
from datetime import datetime
from pathlib import Path
from collector.net_retry import open_with_retry

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


BASE_URL = "https://www.betman.co.kr"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class BetmanScraper:
    def __init__(self, gm_id: str = "G101"):
        self.gm_id = gm_id
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))

    def find_current_round(self) -> str:
        """
        베트맨 공식 API에서 현재 발매(판매) 중인 프로토 승부식(gm_id) 최신 gmTs를 자동 조회합니다.
        """
        print(f"🔍 베트맨 현재 판매 회차 자동 조회 중({self.gm_id})...")
        init_url = f"{BASE_URL}/main/mainPage/gamebuy/buyableGameList.do"
        try:
            req_init = urllib.request.Request(init_url, headers={"User-Agent": DEFAULT_USER_AGENT})
            self.opener.open(req_init, timeout=8)
        except Exception as e:
            print(f"  ⚠ 세션 쿠키 초기화 경고: {e}")

        endpoints = [
            f"{BASE_URL}/buyPsblGame/inqBuyAbleGameInfoList.do",
            f"{BASE_URL}/buyPsblGame/inqCacheBuyAbleGameInfoList.do"
        ]
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": init_url,
            "Origin": BASE_URL,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest"
        }
        payload = json.dumps({"_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}}).encode("utf-8")

        for endpoint in endpoints:
            try:
                req = urllib.request.Request(endpoint, data=payload, headers=headers)
                with open_with_retry(self.opener, req, timeout=8, label="회차 조회") as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                proto_games = data.get("protoGames", [])
                matched_rounds = []
                for g in proto_games:
                    if g.get("gmId") == self.gm_id:
                        gm_ts = str(g.get("gmTs", "")).strip()
                        if gm_ts:
                            matched_rounds.append(gm_ts)
                if matched_rounds:
                    matched_rounds.sort(reverse=True)
                    round_found = matched_rounds[0]
                    print(f"  ✓ 현재 판매 중인 최신 회차 발견: {round_found}")
                    return round_found
            except Exception:
                pass

        print("  ⚠ 현재 판매 회차 자동 조회 실패")
        return None

    def fetch_current_round_matches(self, gm_ts: str = None) -> list:
        if not gm_ts:
            gm_ts = self.find_current_round()
            if not gm_ts:
                print("  ⚠ 회차 정보를 확인할 수 없어 경기 수집을 중단합니다.")
                return []

        print(f"🔍 베트맨 프로토 승부식({self.gm_id}, 회차:{gm_ts}) 실시간 API 연동 시작...")
        
        matches = []
        try:
            # 1. 세션 쿠키 획득
            ifr_url = f"{BASE_URL}/main/mainPage/gamebuy/gameScheduleDetlIFR.do?gmId={self.gm_id}&gmTs={gm_ts}"
            req_init = urllib.request.Request(ifr_url, headers={"User-Agent": DEFAULT_USER_AGENT})
            self.opener.open(req_init, timeout=10)

            # 2. 공식 비동기 스케줄 API 호출
            api_url = f"{BASE_URL}/buyPsblGame/detailSchedule.do"
            payload = {
                "gmId": self.gm_id,
                "gmTs": int(gm_ts) if str(gm_ts).isdigit() else 0,
                "yearMonth": "",
                "sportsCode": "",
                "betId": "",
                "league": "",
                "sglYn": "",
                "sortName": "gameDate",
                "sortOrder": "",
                "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}
            }
            json_body = json.dumps(payload).encode("utf-8")
            
            headers = {
                "User-Agent": DEFAULT_USER_AGENT,
                "Referer": ifr_url,
                "Origin": BASE_URL,
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest"
            }
            
            req_api = urllib.request.Request(api_url, data=json_body, headers=headers)
            with open_with_retry(self.opener, req_api, timeout=15, label="경기 목록 조회") as resp:
                data = json.loads(resp.read().decode("utf-8"))

            schedules = data.get("schedules", [])
            print(f"  ✓ 베트맨 실시간 서버로부터 {len(schedules)}개 경기 데이터 수신 완료!")

            # 3. 경기 정보 추출 및 카테고리 세분화
            for s in schedules:
                home_name = s.get("homeName") or s.get("homeShortName") or ""
                away_name = s.get("awayName") or s.get("awayShortName") or ""
                
                win_allot = float(s.get("winAllot") or 0.0)
                draw_allot = float(s.get("drawAllot") or 0.0)
                lose_allot = float(s.get("loseAllot") or 0.0)
                
                if home_name and away_name and (win_allot > 1.0 or lose_allot > 1.0):
                    # 베트맨 원본 개최일시 문자열 (예: 26.10.24 (토) 14:00)
                    ts = s.get("gameDate")
                    date_str = s.get("gameDateStr")
                    if not date_str and ts:
                        try:
                            date_str = datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            date_str = str(ts)

                    bet_nm = s.get("betNm") or s.get("betTypNm") or "승무패"
                    is_single = (str(s.get("sgl")) == "1")
                    line_val = float(s.get("winHandi") or 0.0)

                    # 카테고리 세분화 (일반 승무패, 핸디캡, 언더오버, 전반전 등)
                    if "전반" in bet_nm:
                        if "언더오버" in bet_nm:
                            category = "전반 언더오버"
                        elif "핸디캡" in bet_nm:
                            category = "전반 핸디캡"
                        else:
                            category = "전반 승무패"
                    elif "핸디캡" in bet_nm:
                        category = "핸디캡"
                    elif "언더오버" in bet_nm:
                        category = "언더오버"
                    else:
                        category = "일반 승무패"

                    matches.append({
                        "match_seq": str(s.get("matchSeq", "")),
                        "match_key": s.get("gameKey") or f"{home_name.strip()}:{away_name.strip()}",
                        "sport": s.get("itemCode", "축구"),
                        "league": s.get("leagueName") or s.get("leagueShortName") or "기타",
                        "game_type": bet_nm,
                        "category": category,
                        "is_single": is_single,
                        "line": line_val,
                        "win_txt": s.get("winTxt", "승"),
                        "draw_txt": s.get("drawTxt", "무"),
                        "lose_txt": s.get("loseTxt", "패"),
                        "match_time": date_str or "",
                        "close_time": s.get("endDate"),
                        "kickoff_ts": s.get("gameDate"),
                        "buy_reject": s.get("buyReject"),
                        "game_reject": s.get("gameReject"),
                        "proto_status": s.get("protoStatus"),
                        "match_state": s.get("matchState"),
                        "home_team": home_name.strip(),
                        "away_team": away_name.strip(),
                        "odds": {
                            "home_win": win_allot,
                            "draw": draw_allot,
                            "away_win": lose_allot
                        }
                    })

            print(f"  ✓ 유효 구매 대상 {len(matches)}개 경기 정제 완료!")

        except Exception as e:
            print(f"  ⚠ 베트맨 실시간 API 호출 중 오류 발생: {e}")
            matches = []

        return matches

    def save_matches(self, matches: list, target_path: str = "data/current_matches.json"):
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        print(f"💾 수집 결과 저장 완료: {dest} (총 {len(matches)}경기)")


def find_current_round(gm_id: str = "G101") -> str:
    """모듈 레벨 헬퍼 함수: 현재 판매 중인 최신 회차를 반환합니다."""
    return BetmanScraper(gm_id=gm_id).find_current_round()


if __name__ == "__main__":
    scraper = BetmanScraper()
    cur_round = scraper.find_current_round()
    matches = scraper.fetch_current_round_matches(gm_ts=cur_round)
    scraper.save_matches(matches)
