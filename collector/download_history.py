"""
download_history.py
===================
오픈소스 축구 데이터(football-data.co.uk)에서 최근 3시즌 주요 5대 리그의 
경기 결과, 스코어, 득실점, 배당률 데이터를 무료로 다운로드하여 로컬에 저장합니다.
"""

import os
import sys
import urllib.request
import pandas as pd
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 주요 리그 코드 및 이름 매핑
LEAGUES = {
    "E0": "EPL (잉글랜드 프리미어리그)",
    "SP1": "LaLiga (스페인 라리가)",
    "D1": "Bundesliga (독일 분데스리가)",
    "I1": "SerieA (이탈리아 세리에A)",
    "F1": "Ligue1 (프랑스 리그1)"
}

# 최근 시즌 목록 (예: 2324 = 2023-2024 시즌, 2425 = 2024-2025 시즌)
SEASONS = ["2223", "2324", "2425"]

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def download_historical_data(target_dir: str = "data/history") -> dict:
    """
    지정된 디렉터리에 최근 시즌 축구 경기 결과 CSV를 다운로드하고 요약 통계를 반환합니다.
    """
    save_path = Path(target_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    total_matches = 0
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print("🚀 [1/2] 오픈소스 공식 역사 축구 데이터 다운로드 시작 (football-data.co.uk)...")

    for season in SEASONS:
        for league_code, league_name in LEAGUES.items():
            file_name = f"{league_code}_{season}.csv"
            file_dest = save_path / file_name
            url = f"{BASE_URL}/{season}/{league_code}.csv"
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    content = resp.read()
                    with open(file_dest, "wb") as f:
                        f.write(content)
                    
                    # 데이터 검증 (행 수 확인)
                    df = pd.read_csv(file_dest, encoding="latin1")
                    # 필수 컬럼(날짜, 홈팀, 원정팀, 홈골, 원정골) 확인
                    if "HomeTeam" in df.columns and "FTHG" in df.columns:
                        valid_rows = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
                        match_count = len(valid_rows)
                        total_matches += match_count
                        downloaded_files.append({
                            "season": season,
                            "league": league_name,
                            "file": file_name,
                            "matches": match_count
                        })
                        print(f"  ✓ {league_name} [{season} 시즌]: {match_count}경기 다운로드 완료 ({file_name})")
            except Exception as e:
                print(f"  ⚠ {league_name} [{season}] 다운로드 건너뜀 (URL: {url}, 오류: {e})")

    # 전체 통합 데이터셋 생성 (cleaned_matches.csv)
    print("\n📦 [2/2] 다운로드된 경기 데이터 통합 및 표준화 중...")
    all_dfs = []
    for item in downloaded_files:
        fpath = save_path / item["file"]
        if fpath.exists():
            df = pd.read_csv(fpath, encoding="latin1")
            keep_cols = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"] if c in df.columns]
            subset = df[keep_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
            subset["Season"] = item["season"]
            subset["League"] = item["league"]
            all_dfs.append(subset)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_path = save_path / "all_historical_matches.csv"
        merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"✅ 통합 과거 데이터 생성 완료: {merged_path} (총 {len(merged_df):,}개 경기)")
    
    print("\n🎉 과거 데이터베이스 구축이 완료되었습니다!")
    return {
        "total_files": len(downloaded_files),
        "total_matches": total_matches
    }


if __name__ == "__main__":
    download_historical_data()
