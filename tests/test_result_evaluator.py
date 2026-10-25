import json
import tempfile
from pathlib import Path
from tracker.result_evaluator import evaluate_betman_results
from tracker.odds_snapshot import save_odds_snapshot, get_earliest_odds


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
            output_path=str(output_file),
            snapshots_dir=str(Path(tmpdir) / "snapshots")
        )

        assert stats is not None
        assert "updated_at" in stats
        assert "sample_matches_analyzed" in stats
        assert stats["sample_matches_analyzed"] == 2
        assert "clv_sample_count" in stats
        assert stats["clv_sample_count"] == 0
        assert stats["avg_clv_pct"] == 0.0
        assert output_file.exists()


def test_clv_calculation_with_snapshot():
    """예측 시점 배당 스냅샷이 있는 경기만 CLV 평균에 포함되는지 검증"""
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_dir = Path(tmpdir) / "snapshots"
        snap_dir.mkdir()

        # 1. 260125 회차 예측 시점 배당 스냅샷 저장
        # match 1: home_win 배당 2.00
        snap_matches = [
            {
                "match_seq": "1",
                "home_team": "아스널",
                "away_team": "첼시",
                "category": "일반 승무패",
                "line": 0.0,
                "odds": {
                    "home_win": 2.00,
                    "draw": 3.40,
                    "away_win": 3.80
                }
            }
        ]
        save_odds_snapshot(260125, snap_matches, snapshot_dir=str(snap_dir))

        # 2. 결과 데이터: match 1 마감 배당은 1.80 (배당 하락 -> CLV = (2.00/1.80)-1 = +11.11%)
        # match 2: 스냅샷에 없는 경기 -> CLV 계산에서 완전히 제외되어야 함
        results_data = [
            {
                "gm_ts": 260125,
                "match_seq": "1",
                "sport": "SC",
                "league": "EPL",
                "game_type": "일반",
                "bettyp_typ_nm": "일반",
                "game_result": "0",
                "home_team": "아스널",
                "away_team": "첼시",
                "home_score": 2,
                "away_score": 1,
                "odds_win": 1.80,
                "odds_draw": 3.40,
                "odds_lose": 3.80,
                "match_time": "26.10.20 (화) 20:00"
            },
            {
                "gm_ts": 260125,
                "match_seq": "2",
                "sport": "SC",
                "league": "LaLiga",
                "game_type": "일반",
                "bettyp_typ_nm": "일반",
                "game_result": "0",
                "home_team": "레알마드리드",
                "away_team": "바르셀로나",
                "home_score": 2,
                "away_score": 0,
                "odds_win": 2.50,
                "odds_draw": 3.30,
                "odds_lose": 2.80,
                "match_time": "26.10.21 (수) 21:00"
            }
        ]

        input_file = Path(tmpdir) / "results.json"
        output_file = Path(tmpdir) / "stats.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False)

        stats = evaluate_betman_results(
            results_path=str(input_file),
            output_path=str(output_file),
            snapshots_dir=str(snap_dir)
        )

        assert stats is not None
        # 추천된 경기 중 스냅샷이 있는 경기만 카운트
        picks = stats.get("recent_evaluated_picks", [])
        m1_pick = next((p for p in picks if p["match_seq"] == "1"), None)
        m2_pick = next((p for p in picks if p["match_seq"] == "2"), None)

        if m1_pick:
            assert m1_pick["init_odds"] == 2.00
            assert m1_pick["closing_odds"] == 1.80
            assert m1_pick["clv_pct"] == 11.11

        if m2_pick:
            assert m2_pick["init_odds"] is None
            assert m2_pick["clv_pct"] is None

        if m1_pick:
            assert stats["clv_sample_count"] == 1
            assert stats["avg_clv_pct"] == 11.11
