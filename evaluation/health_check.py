"""
evaluation/health_check.py
==========================
[Phase 6.1] 운영 점검 자동화 — 파이프라인·결과 갱신이 끝날 때마다 데이터 품질을 점검하고, 이상이 있으면 경고한다.

점검 항목
- match_count_drop : 같은 회차에서 수집 경기 수가 직전 실행의 절반 이하로 줄었다
- no_matches       : 예측 파일에 경기가 하나도 없다
- odds_missing     : 유효 배당(1.0 초과)이 2개 미만인 경기(배당 0·누락)가 5%를 넘는다
- count_mismatch   : 예측 파일과 경기 파일의 경기 수가 다르다
- pipeline_stale   : 파이프라인이 36시간 넘게 실행되지 않았다 (하루 2회 예약) — 결과 갱신 때 점검
- results_stale    : 결과 갱신이 6시간 넘게 실행되지 않았다 (1시간마다 예약) — 파이프라인 때 점검
- graded_decrease  : 성적표 채점 경기 수가 직전보다 줄었다 (기록 유실)

저장: data/health_state.json (직전 값 · 마지막 실행 시각), data/health_report.json (이번 점검 결과)

실행:
    python -m evaluation.health_check --source pipeline|results   # 점검 + 기록 (경고가 있어도 실패로 끝나지 않음)
    python -m evaluation.health_check --source pipeline --dry-run # 기록하지 않고 결과만 출력
    python -m evaluation.health_check --verify                    # 기록된 경고가 있으면 종료 코드 1 (Actions 실패 표시) + 텔레그램
    python -m evaluation.health_check --notify-failure 파이프라인   # 앞 단계가 실패했을 때 텔레그램 알림만

표준 라이브러리만 쓴다 (결과 갱신 Actions는 numpy만 설치한다).
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = Path("data")
PIPELINE_STALE_H = 36
RESULTS_STALE_H = 6
DROP_RATIO = 0.5
MIN_COUNT_FOR_DROP = 20
ODDS_MISSING_SHARE = 0.05


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _parse_ts(s: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _valid_odds(m: Dict[str, Any]) -> int:
    odds = m.get("betman_odds") or m.get("odds") or {}
    n = 0
    for v in odds.values():
        try:
            n += float(v) > 1.0
        except (TypeError, ValueError):
            pass
    return n


def _latest_round(predictions: List[Dict[str, Any]]) -> Optional[str]:
    rounds = sorted({str(m.get("gm_ts")) for m in predictions if m.get("gm_ts")})
    return rounds[-1] if rounds else None


def _warn(code: str, msg: str) -> Dict[str, str]:
    return {"code": code, "message": msg}


def run_checks(predictions: List[Dict[str, Any]], current_matches: Optional[List[Dict[str, Any]]],
               scorecard: Optional[Dict[str, Any]], state: Dict[str, Any], now: datetime,
               source: str = "pipeline") -> List[Dict[str, str]]:
    """경고 목록 (정상이면 빈 목록). state = 직전 실행 기록"""
    w: List[Dict[str, str]] = []
    n = len(predictions)
    gm = _latest_round(predictions)

    prev_n, prev_gm = state.get("match_count"), state.get("gm_ts")
    if n == 0:
        w.append(_warn("no_matches", "예측 파일에 경기가 하나도 없어요"))
    elif prev_n and prev_gm == gm and prev_n >= MIN_COUNT_FOR_DROP and n <= prev_n * DROP_RATIO:
        w.append(_warn("match_count_drop", f"{gm}회차 수집 경기 수가 직전 {prev_n} → 이번 {n}으로 절반 이하로 줄었어요"))

    bad = [str(m.get("match_seq")) for m in predictions if _valid_odds(m) < 2]
    if n and len(bad) / n > ODDS_MISSING_SHARE:
        w.append(_warn("odds_missing", f"배당이 0이거나 빠진 경기가 {len(bad)}/{n}경기예요 (예: {', '.join(bad[:5])})"))

    if current_matches is not None and len(current_matches) != n:
        w.append(_warn("count_mismatch", f"예측 파일 {n}경기 ≠ 경기 파일 {len(current_matches)}경기"))

    if source != "pipeline":
        last_p = _parse_ts(state.get("pipeline_last_run"))
        if last_p and now - last_p > timedelta(hours=PIPELINE_STALE_H):
            w.append(_warn("pipeline_stale", f"파이프라인이 {(now - last_p).total_seconds() / 3600:.0f}시간째 실행되지 않았어요 "
                                             f"(한도 {PIPELINE_STALE_H}시간, 마지막 {last_p.isoformat(timespec='minutes')})"))
    if source != "results":
        last_r = _parse_ts(state.get("results_last_run"))
        if last_r and now - last_r > timedelta(hours=RESULTS_STALE_H):
            w.append(_warn("results_stale", f"결과 갱신이 {(now - last_r).total_seconds() / 3600:.0f}시간째 실행되지 않았어요 "
                                            f"(한도 {RESULTS_STALE_H}시간, 마지막 {last_r.isoformat(timespec='minutes')})"))

    graded = (scorecard or {}).get("graded_matches")
    if graded is not None and state.get("graded") is not None and graded < state["graded"]:
        w.append(_warn("graded_decrease", f"성적표 채점 경기 수가 직전 {state['graded']} → 이번 {graded}으로 줄었어요"))
    return w


def next_state(state: Dict[str, Any], predictions: List[Dict[str, Any]], scorecard: Optional[Dict[str, Any]],
               source: str, now: datetime) -> Dict[str, Any]:
    out = dict(state)
    out["match_count"] = len(predictions)
    out["gm_ts"] = _latest_round(predictions)
    if (scorecard or {}).get("graded_matches") is not None:
        out["graded"] = scorecard["graded_matches"]
    out[f"{source}_last_run"] = now.isoformat(timespec="seconds")
    return out


def record(source: str, data_dir: Path = DATA_DIR, now: Optional[datetime] = None, dry_run: bool = False) -> Dict[str, Any]:
    """파일을 읽어 점검하고, (dry_run이 아니면) 상태·보고서를 저장한다"""
    now = now or datetime.now(timezone.utc)
    data_dir = Path(data_dir)
    predictions = _load(data_dir / "predictions.json", [])
    current = _load(data_dir / "current_matches.json", None)
    scorecard = _load(data_dir / "model_scorecard.json", None)
    state = _load(data_dir / "health_state.json", {})
    warnings = run_checks(predictions, current, scorecard, state, now, source)
    report = {"checked_at": now.isoformat(timespec="seconds"), "source": source, "ok": not warnings,
              "warnings": warnings, "matches": len(predictions), "round": _latest_round(predictions),
              "graded": (scorecard or {}).get("graded_matches")}
    if not dry_run:
        (data_dir / "health_state.json").write_text(
            json.dumps(next_state(state, predictions, scorecard, source, now), ensure_ascii=False, indent=2), encoding="utf-8")
        (data_dir / "health_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🩺 운영 점검 ({source}): " + ("이상 없음" if not warnings else f"경고 {len(warnings)}건"))
    for x in warnings:
        print(f"  ⚠️ [{x['code']}] {x['message']}")
    return report


def send_telegram(text: str) -> bool:
    """텔레그램 Secret(TELEGRAM_BOT_TOKEN · TELEGRAM_CHAT_ID)이 있을 때만 보낸다 (REVIEW R6: 채널 등록은 사용자 작업)"""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  (텔레그램 Secret이 없어 알림은 보내지 않아요)")
        return False
    try:
        body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=body, timeout=10)
        return True
    except Exception as e:
        print(f"  텔레그램 알림 실패: {e}")
        return False


def verify(data_dir: Path = DATA_DIR) -> int:
    """기록된 보고서에 경고가 있으면 텔레그램을 보내고 1을 돌려준다 (Actions에서 실패로 표시)"""
    report = _load(Path(data_dir) / "health_report.json", None)
    if not report:
        print("점검 보고서가 없어요")
        return 0
    if report.get("ok"):
        print(f"✅ 운영 점검 이상 없음 ({report.get('source')}, {report.get('checked_at')})")
        return 0
    lines = [f"⚠️ 스포츠 예측 엔진 운영 점검 경고 ({report.get('source')}, {report.get('checked_at')})"]
    lines += [f"- {x['message']}" for x in report.get("warnings", [])]
    print("\n".join(lines))
    send_telegram("\n".join(lines))
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="[Phase 6.1] 운영 점검")
    ap.add_argument("--source", choices=["pipeline", "results"], help="점검 + 기록")
    ap.add_argument("--dry-run", action="store_true", help="기록하지 않고 결과만 출력")
    ap.add_argument("--verify", action="store_true", help="기록된 경고가 있으면 종료 코드 1")
    ap.add_argument("--notify-failure", metavar="JOB", help="앞 단계가 실패했을 때 텔레그램 알림만 보냄")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    if args.notify_failure:
        msg = f"❌ 스포츠 예측 엔진: {args.notify_failure} 실행이 실패했어요. GitHub Actions 로그를 확인하세요."
        print(msg)
        send_telegram(msg)
    elif args.verify:
        sys.exit(verify(Path(args.data_dir)))
    elif args.source:
        record(args.source, Path(args.data_dir), dry_run=args.dry_run)
    else:
        ap.print_help()
