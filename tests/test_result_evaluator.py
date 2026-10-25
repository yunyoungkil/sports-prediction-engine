import json
import tempfile
from pathlib import Path
from tracker.result_evaluator import evaluate_betman_results


def test_evaluate_betman_results_smoke():
    """소량 샘플 데이터로 evaluate_betman_results가 오류 없이 정상 종료되는지 확인하는 스모크 테스트"""
    sample_matches = [
        {
            "gm_ts": 260125,
            "match_seq": 1,
            "sport": "SC",
            "league": "EPL",
            "game_type": "일반",
            "bettyp_typ_nm": "일반",
            "game_result": "0",
            "home_team": "아스널",
            "away_team": "첼시",
            "home_score": 2,
            "away_score": 1,
            "odds_win": 1.85,
            "odds_draw": 3.40,
            "odds_lose": 3.80,
            "match_time": "26.10.20 (화) 20:00"
        },
        {
            "gm_ts": 260125,
            "match_seq": 2,
            "sport": "SC",
            "league": "LaLiga",
            "game_type": "일반",
            "bettyp_typ_nm": "일반",
            "game_result": "1",
            "home_team": "레알마드리드",
            "away_team": "바르셀로나",
            "home_score": 1,
            "away_score": 1,
            "odds_win": 2.10,
            "odds_draw": 3.30,
            "odds_lose": 3.10,
            "match_time": "26.10.21 (수) 21:00"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test_results.json"
        output_file = Path(tmpdir) / "test_stats.json"

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(sample_matches, f, ensure_ascii=False)

        stats = evaluate_betman_results(
            results_path=str(input_file),
            output_path=str(output_file)
        )

        assert stats is not None
        assert "updated_at" in stats
        assert "sample_matches_analyzed" in stats
        assert stats["sample_matches_analyzed"] == 2
        assert output_file.exists()
