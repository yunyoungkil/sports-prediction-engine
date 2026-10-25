from models.poisson_model import normalize_league_name


def test_normalize_league_name_distinguishes_kleague_and_ligue1():
    """K리그1이 프랑스 리그1로 오매핑되지 않는지 검증 (REVIEW R7 회귀 방지)"""
    test_cases = [
        ("K리그1", ""),
        ("K리그2", ""),
        ("잉글랜드 챔피언십", ""),
        ("프랑스 리그1", "Ligue1"),
        ("Ligue 1", "Ligue1"),
        ("F1", "Ligue1"),
        ("EPL", "EPL"),
        ("잉글랜드 프리미어리그", "EPL"),
        ("스페인 라리가", "LaLiga"),
        ("LaLiga", "LaLiga"),
        ("독일 분데스리가", "Bundesliga"),
        ("이탈리아 세리에A", "SerieA")
    ]
    for league_input, expected in test_cases:
        actual = normalize_league_name(league_input)
        assert actual == expected, f"Failed for {league_input}: expected {expected}, got {actual}"
