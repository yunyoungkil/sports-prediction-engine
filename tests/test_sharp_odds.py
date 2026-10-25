import pandas as pd
from collector.sharp_odds import SharpOddsCollector


def test_extract_historical_sharp_odds():
    """샤프 배당 추출 시 Betfair Exchange와 Pinnacle 우선순위 및 폴백 정상 동작 검증"""
    sample_df = pd.DataFrame([
        # Case 1: Betfair Exchange 마감 배당 보유
        {
            "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "BFECH": 1.95, "BFECD": 3.60, "BFECA": 4.10,
            "PSCH": 1.90, "PSCD": 3.50, "PSCA": 4.00,
            "B365H": 1.85, "B365D": 3.40, "B365A": 3.90
        },
        # Case 2: Pinnacle 마감 배당 보유 (과거 시즌)
        {
            "HomeTeam": "Liverpool", "AwayTeam": "Everton",
            "BFECH": None, "BFECD": None, "BFECA": None,
            "PSCH": 1.45, "PSCD": 4.80, "PSCA": 7.50,
            "B365H": 1.40, "B365D": 4.50, "B365A": 7.00
        },
        # Case 3: Bet365 폴백
        {
            "HomeTeam": "Barcelona", "AwayTeam": "Real Madrid",
            "BFECH": None, "BFECD": None, "BFECA": None,
            "PSCH": None, "PSCD": None, "PSCA": None,
            "BFEH": None, "BFED": None, "BFEA": None,
            "PSH": None, "PSD": None, "PSA": None,
            "B365H": 2.20, "B365D": 3.50, "B365A": 3.20
        }
    ])

    result = SharpOddsCollector.extract_historical_sharp_odds(sample_df, prefer_closing=True)

    assert result.loc[0, "sharp_source"] == "Betfair_Exchange_Closing"
    assert result.loc[0, "sharp_home"] == 1.95
    assert result.loc[0, "sharp_away"] == 4.10

    assert result.loc[1, "sharp_source"] == "Pinnacle_Closing"
    assert result.loc[1, "sharp_home"] == 1.45
    assert result.loc[1, "sharp_away"] == 7.50

    assert result.loc[2, "sharp_source"] == "Bet365_Fallback"
    assert result.loc[2, "sharp_home"] == 2.20
