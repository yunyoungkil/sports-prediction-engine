"""
evaluation/briefing.py
======================
대화 시작 브리핑 — 테스트 · 자동 실행 · 판정 진행 상황을 한 번에 (2026-10-27 사용자 요청).
Claude Code 세션 시작 훅(.claude/settings.local.json)이 실행하고, 출력은 대화 맥락에 들어간다.

- 판정 숫자는 봇이 GitHub에 올린 최신 파일(origin/master)을 먼저 읽는다 → 로컬이 뒤처져 있어도 정확
- 작업 트리·저장소 파일은 바꾸지 않는다 (git fetch만)

실행:
    python -m evaluation.briefing
    python -m evaluation.briefing --no-tests
"""

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evaluation.gh_cli import gh_path as _gh

ROOT = Path(__file__).resolve().parent.parent
KST = timezone(timedelta(hours=9))
WORKFLOWS = [("pipeline.yml", "파이프라인 (하루 2번)"), ("results.yml", "결과 갱신 (1시간 예약, 실제 4~5시간)"),
             ("test.yml", "GitHub 테스트 (푸시 때)")]

# 외부 스케줄러(cron-job.org, 2026-10-29 설정): GitHub 무료 예약 실행이 5~7시간 늦거나 빠지는 문제를 피하려고
# 정해진 시각에 workflow_dispatch로 직접 깨운다. 작업: sports pipeline (매일 09:30·18:30 KST), sports results (2시간마다)
# (워크플로, 이름, 하루 기대 횟수, 이 시간 넘게 호출이 없으면 경고)
SCHEDULER_JOBS = [("pipeline.yml", "파이프라인", 2, 16.0), ("results.yml", "결과 갱신", 12, 3.5)]
# cron-job.org가 GitHub를 부를 때 쓰는 fine-grained 토큰(sports-engine-scheduler, Actions 쓰기 전용)의 만료일.
# 갱신하면 이 날짜도 같이 고칠 것: GitHub Settings → Personal access tokens → 해당 토큰 → Regenerate → cron-job.org 두 작업의 Authorization 헤더 교체
SCHEDULER_TOKEN_EXPIRES = date(2027, 10, 28)
TOKEN_WARN_DAYS = 30


def _run(cmd: List[str], timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except Exception:
        return None


def run_tests() -> str:
    r = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout=600)
    if r is None:
        return "⚠️ 테스트: 실행하지 못했어요"
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    last = lines[-1] if lines else "(출력 없음)"
    if r.returncode == 0:
        return f"✅ 테스트: {last}"
    failed = [line.replace("FAILED ", "") for line in lines if line.startswith("FAILED")]
    return f"❌ 테스트: {last}" + "".join(f"\n   - {f}" for f in failed[:10])


def git_sync() -> str:
    """원격 최신 정보만 가져와(작업 트리는 그대로) 로컬이 몇 커밋 뒤인지"""
    _run(["git", "fetch", "-q", "origin"], timeout=30)
    r = _run(["git", "rev-list", "--count", "HEAD..origin/master"])
    behind = int(r.stdout.strip()) if r and r.returncode == 0 and r.stdout.strip().isdigit() else None
    st = _run(["git", "status", "--porcelain"])
    dirty = len([line for line in (st.stdout.splitlines() if st and st.returncode == 0 else []) if line.strip()])
    parts = []
    if behind:
        parts.append(f"로컬이 GitHub보다 {behind}커밋 뒤 (주로 봇 데이터 갱신 — 작업 전 git pull --rebase)")
    elif behind == 0:
        parts.append("로컬이 GitHub 최신과 같음")
    if dirty:
        parts.append(f"커밋 안 된 변경 {dirty}개")
    return "📦 저장소: " + (" · ".join(parts) if parts else "확인 못 함")


def read_json(rel: str) -> Optional[Dict[str, Any]]:
    """봇이 GitHub에 올린 최신 파일(origin/master) → 없으면 로컬 파일"""
    r = _run(["git", "show", f"origin/master:{rel}"])
    if r and r.returncode == 0 and r.stdout.strip():
        try:
            return json.loads(r.stdout)
        except Exception:
            pass
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def runs_status(now: Optional[datetime] = None) -> List[str]:
    now = now or datetime.now(timezone.utc)
    out = ["⚙️ 자동 실행 (GitHub, 가장 최근):"]
    for wf, label in WORKFLOWS:
        r = _run([_gh(), "run", "list", "--workflow", wf, "--limit", "5", "--json", "conclusion,status,createdAt"], timeout=30)
        if r is None or r.returncode != 0:
            out.append(f"   - ⚠️ {label}: 상태를 못 가져왔어요 (gh 로그인 확인)")
            continue
        runs = json.loads(r.stdout or "[]")
        if not runs:
            out.append(f"   - {label}: 기록 없음")
            continue
        x = runs[0]
        t = datetime.fromisoformat(x["createdAt"].replace("Z", "+00:00"))
        mark = {"success": "✅", "failure": "❌", "cancelled": "⚪"}.get(x.get("conclusion") or "", "⏳")
        fails = sum(1 for y in runs if y.get("conclusion") == "failure")
        out.append(f"   - {mark} {label}: {t.astimezone(KST):%m-%d %H:%M} ({(now - t).total_seconds() / 3600:.0f}시간 전) "
                   f"{x.get('conclusion') or x.get('status')}" + (f" · 최근 5번 중 실패 {fails}번" if fails else ""))
    return out


def paper_lines(reader: Callable[[str], Optional[Dict[str, Any]]] = read_json) -> List[str]:
    """가상 베팅 리그 현황: 전략별 건수·적중·대기·손익·수익률과 판정 (표본이 적으면 판정 보류)"""
    pt = reader("data/paper_trading.json")
    if not pt or not pt.get("strategies"):
        return ["🧪 가상 베팅: 기록 없음"]
    unit = pt.get("stake_unit", 1000)
    out = [f"🧪 가상 베팅 (1건 {unit:,}원, 갱신 {str(pt.get('updated_at', ''))[:16].replace('T', ' ')} UTC):"]
    for s in pt["strategies"].values():
        pend = f" · 결과 대기 {s['pending']}" if s.get("pending") else ""
        if "total_deposited" in s:      # G: 월 10만 원 지갑 — 잔고·누적 입금·순손익·상태
            state = ("🚨 소진, 다음 달 1일까지 쉼" if s.get("is_broke")
                     else "⏳ 대기 중인 베팅에 돈이 묶임" if s.get("can_bet") is False else s.get("mode_label", ""))
            wallet = (f" · 지갑 {s['bankroll']:,}원 (누적 입금 {s['total_deposited']:,}원 · 순손익 {s.get('profit', 0):+,}원"
                      + (f" · {state}" if state else "") + ")")
            if "instinct_rounds" in s:  # G-본능: 본능대로 움직인 회차 수
                ir = s["instinct_rounds"]
                wallet += f" · 이긴 뒤 판돈 키움 {ir.get('win_boost', 0)}회차 · 진 뒤 고배당 {ir.get('chase', 0)}회차"
        elif "bankroll" in s:           # F: 금액이 매번 달라 지갑으로 봄
            wallet = f" · 지갑 {s['bankroll']:,}원 (쉰 회차 {s.get('rounds_skipped', 0)}/{s.get('rounds_planned', 0)})"
        else:
            wallet = ""
        if not s.get("bets"):
            out.append(f"   - {s['label']}: 채점된 베팅 없음{pend}{wallet}")
            continue
        out.append(f"   - {s['label']}: {s['bets']}건 중 {s['hits']}건 적중{pend} · 손익 {s['profit']:+,}원 "
                   f"(수익률 {s['roi_pct']:+.1f}%){wallet} · {s.get('verdict', '')}")
        if s.get("sizing"):             # G: 주력/복권 성적과 '판돈 조절이 도움이 되나' (같은 베팅 1,000원 고정과 비교)
            z = s["sizing"]
            slips = " · ".join(f"{v['label']} {v['bets']}건 {v['roi_pct']:+.1f}%" for v in (s.get("by_slip") or {}).values() if v["bets"])
            out.append(f"       ↳ {slips} · 판돈 조절 효과 {z['gain_pp']:+.1f}%p "
                       f"(AI 금액 {z['roi_ai_pct']:+.1f}% vs 1,000원 고정 {z['roi_flat_pct']:+.1f}%) · {z['verdict']}")
    g, gi = pt["strategies"].get("G_ai_bettor"), pt["strategies"].get("G_instinct")
    if g and gi:                        # 같은 경기, 다른 습관 — 원칙을 지킨 값 (과거 재현 28개월: G −618,790원 vs 본능 −1,359,720원)
        out.append(f"   ↳ 원칙 vs 본능 순손익: G {g.get('profit', 0):+,}원 vs G-본능 {gi.get('profit', 0):+,}원 "
                   f"(과거 재현 28개월: −618,790원 vs −1,359,720원)")
    return out


def scheduler_lines(now: Optional[datetime] = None,
                    runner: Callable[..., Optional[subprocess.CompletedProcess]] = _run) -> List[str]:
    """외부 스케줄러(cron-job.org)가 GitHub를 제때 깨우고 있는지 — workflow_dispatch 실행 기록으로 확인.
    (사람이 수동으로 돌린 실행도 같은 방식으로 기록돼 구분되지 않는다. 늦거나 적으면 경고)"""
    now = now or datetime.now(timezone.utc)
    out = ["⏰ 외부 스케줄러 (cron-job.org → GitHub, 최근 24시간):"]
    for wf, label, expected, late_h in SCHEDULER_JOBS:
        r = runner([_gh(), "run", "list", "--workflow", wf, "--event", "workflow_dispatch", "--limit", "40",
                    "--json", "createdAt"], timeout=30)
        if r is None or r.returncode != 0:
            out.append(f"   - ⚠️ {label}: 기록을 못 가져왔어요 (gh 로그인 확인)")
            continue
        times = sorted(datetime.fromisoformat(x["createdAt"].replace("Z", "+00:00")) for x in json.loads(r.stdout or "[]"))
        recent = [x for x in times if (now - x).total_seconds() <= 24 * 3600]
        if not times:
            out.append(f"   - ⚠️ {label}: 호출 기록이 없어요 — cron-job.org 작업이 저장·활성화됐는지 확인")
            continue
        age_h = (now - times[-1]).total_seconds() / 3600
        low = len(recent) < max(1, int(expected * 0.7))
        mark = "✅" if age_h <= late_h and not low else "⚠️"
        note = ""
        if age_h > late_h:
            note = f" · {late_h:g}시간 넘게 호출이 없어요 (cron-job.org 작업 상태·토큰 확인)"
        elif low:
            note = " · 평소보다 적어요"
        out.append(f"   - {mark} {label}: 24시간 {len(recent)}/{expected}회 · 마지막 {times[-1].astimezone(KST):%m-%d %H:%M} "
                   f"({age_h:.0f}시간 전){note}")
    return out


def token_expiry_line(today: Optional[date] = None) -> str:
    """스케줄러 토큰 만료가 가까우면 알린다 (30일 전부터 경고, 7일 전부터 긴급)"""
    today = today or datetime.now(KST).date()
    left = (SCHEDULER_TOKEN_EXPIRES - today).days
    how = "GitHub에서 토큰 Regenerate → cron-job.org 두 작업의 Authorization 값 교체 → 이 파일의 SCHEDULER_TOKEN_EXPIRES 수정"
    if left < 0:
        return f"🚨 스케줄러 토큰이 {-left}일 전에 만료됐어요 — cron-job.org 호출이 실패하고 있을 거예요 ({how})"
    if left <= 7:
        return f"🚨 스케줄러 토큰이 {left}일 뒤({SCHEDULER_TOKEN_EXPIRES}) 만료돼요 — 지금 갱신하세요 ({how})"
    if left <= TOKEN_WARN_DAYS:
        return f"⚠️ 스케줄러 토큰이 {left}일 뒤({SCHEDULER_TOKEN_EXPIRES}) 만료돼요 — 곧 갱신해야 해요 ({how})"
    return f"🔑 스케줄러 토큰 만료까지 {left}일 ({SCHEDULER_TOKEN_EXPIRES})"


def _commit_age_hours(rel: str, now: Optional[datetime] = None) -> Optional[float]:
    """origin/master에서 그 파일을 마지막으로 바꾼 커밋이 몇 시간 전인지"""
    r = _run(["git", "log", "-1", "--format=%ct", "origin/master", "--", rel])
    if r is None or r.returncode != 0 or not r.stdout.strip().isdigit():
        return None
    now = now or datetime.now(timezone.utc)
    return (now - datetime.fromtimestamp(int(r.stdout.strip()), timezone.utc)).total_seconds() / 3600


def data_intake_lines(reader: Callable[[str], Optional[Any]] = read_json,
                      age_hours: Callable[[str], Optional[float]] = _commit_age_hours) -> List[str]:
    """데이터를 잘 가져오고 있는지: 경기 수 · 회차 · 배당 이상 · 갱신 시각 · 채점 회차 결과가 비지 않았는지 (봇이 올린 최신 파일 기준)"""
    preds = reader("data/predictions.json")
    results = reader("data/history/betman_recent_results.json") or []
    card = reader("data/model_scorecard.json") or {}
    if not isinstance(preds, list) or not preds:
        return ["📥 데이터 수집: ⚠️ 예측 파일(predictions.json)이 없거나 비어 있어요"]
    problems: List[str] = []
    rounds = sorted({str(p.get("gm_ts")) for p in preds if p.get("gm_ts") is not None})
    no_round = sum(1 for p in preds if p.get("gm_ts") is None)
    if no_round:
        problems.append(f"회차(gm_ts)가 빈 경기 {no_round}건 — 베트맨 회차 조회가 실패해 옛 캐시를 쓴 흔적")
    bad_odds = sum(1 for p in preds if not float((p.get("betman_odds") or {}).get("home_win") or 0) > 1.0
                   or not float((p.get("betman_odds") or {}).get("away_win") or 0) > 1.0)
    if bad_odds:
        problems.append(f"배당이 비었거나 1 이하인 경기 {bad_odds}건")
    by_round: Dict[str, int] = {}
    for r in results:
        by_round[str(r.get("gm_ts"))] = by_round.get(str(r.get("gm_ts")), 0) + 1
    empty = [str(x) for x in (card.get("rounds") or []) if by_round.get(str(x), 0) == 0]
    if empty:
        problems.append(f"채점에 쓰이는 회차 {', '.join(empty)}의 결과가 비어 있어요 (결과 수집 실패)")
    ages = {"예측": age_hours("data/predictions.json"), "결과": age_hours("data/history/betman_recent_results.json")}
    limits = {"예측": 20.0, "결과": 6.0}   # 예측은 하루 2번, 결과는 2시간마다 갱신 (지연 감안)
    fresh = []
    for k, a in ages.items():
        if a is None:
            continue
        fresh.append(f"{k} 갱신 {a:.0f}시간 전")
        if a > limits[k]:
            problems.append(f"{k} 데이터가 {a:.0f}시간째 갱신되지 않았어요 (기준 {limits[k]:g}시간)")
    head = (f"경기 {len(preds):,}건 · 회차 {', '.join(rounds) if rounds else '없음'} · 결과 {sum(by_round.values()):,}건 "
            f"· 채점 {card.get('graded_matches', 0)}경기" + (" · " + " · ".join(fresh) if fresh else ""))
    if problems:
        return ["📥 데이터 수집: ⚠️ " + head] + [f"   - {p}" for p in problems]
    return [f"📥 데이터 수집: ✅ 정상 — {head}"]


def data_audit_lines(reader: Callable[[str], Optional[Any]] = read_json, root: Path = ROOT) -> List[str]:
    """🐞 데이터 점검 — 수집 · 사용 · 계산 · AI 모델 (evaluation/audit.py). 대화마다 버그 흔적을 먼저 찾는다"""
    import csv
    from evaluation.audit import audit_lines, run_audit
    last_ko, last_round = None, None
    try:
        with open(root / "data/history/betman_soccer_1x2.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ko, ts = int(float(r["kickoff_ms"] or 0)), int(r["gm_ts"] or 0)
                last_ko = ko if last_ko is None or ko > last_ko else last_ko
                last_round = ts if last_round is None or ts > last_round else last_round
    except Exception:
        pass
    rounds = sorted(p.stem for p in (root / "data/history/prediction_log").glob("*.json") if p.stem.isdigit())[-3:]
    logs = {ts: reader(f"data/history/prediction_log/{ts}.json") or {} for ts in rounds}
    ko_dt = datetime.fromtimestamp(last_ko / 1000, timezone.utc) if last_ko else None
    return audit_lines(run_audit(reader, logs, ko_dt, last_round))


def contract_lines(reader: Callable[[str], Optional[Any]] = read_json, now: Optional[datetime] = None) -> List[str]:
    """데이터 계약 점검(evaluation/data_contracts.py, 파이프라인이 실행마다 저장) 결과 — 봇이 올린 최신 보고서 기준"""
    rep = reader("data/contract_report.json")
    if not rep:
        return ["📜 데이터 계약: 보고서 없음 (파이프라인이 아직 돌리지 않았어요)"]
    now = now or datetime.now(timezone.utc)
    try:
        age = f"{(now - datetime.fromisoformat(rep['checked_at'])).total_seconds() / 3600:.0f}시간 전"
    except Exception:
        age = "시각 불명"
    rows = " · ".join(f"{k} {v:,}행" for k, v in (rep.get("rows") or {}).items())
    errs = rep.get("errors") or []
    out = [("📜 데이터 계약: ✅ 이상 없음" if not errs else f"📜 데이터 계약: ⚠️ 오류 {len(errs)}종") + f" — {rows} (점검 {age})"]
    out += [f"   - ⚠️ {e.get('message', e)}" for e in errs[:6]]
    out += [f"   - ℹ️ {n}" for n in (rep.get("notes") or [])[:3]]
    return out


def verdict_lines(reader: Callable[[str], Optional[Dict[str, Any]]] = read_json) -> List[str]:
    from evaluation.gates import status_from_data
    card = reader("data/model_scorecard.json") or {}
    stale = reader("data/stale_odds_report.json") or {}
    health = reader("data/health_report.json") or {}
    out = ["📊 판정 진행 (미리 정한 기준):"]
    fx = card.get("fixtures")
    soc = (card.get("by_game_type") or {}).get("축구 승무패") or {}
    out.append(f"   - AI 실전 채점: {card.get('graded_matches', 0)}/300경기 · {card.get('verdict', '기록 없음')}"
               + (f" · 서로 다른 대진 {fx}개" + (f" (축구 승무패 AI만 {soc.get('fixtures', soc.get('n'))}대진, "
                                               f"LL 차이 {soc['ll_diff']:+.4f})" if soc.get("ll_diff") is not None else "") if fx else ""))
    v = stale.get("value") or {}
    out.append(f"   - 지연 배당 (베트맨이 해외 배당을 늦게 따라가나): {v.get('matches', 0)}/100경기 · "
               f"{stale.get('verdict', '아직 기록 없음')}")
    for name, sh in (card.get("shadow") or {}).items():   # [Phase 7.1] 그림자 모델 — 300경기 뒤 교체 후보
        out.append(f"   - 그림자 모델 {name} (지금 AI와 같은 경기 비교, 음수 = 그림자가 더 정확): {sh['n']}경기 · "
                   f"LL 차이 {sh['ll_diff_vs_model']:+.5f} [{sh['ci'][0]:+.5f}, {sh['ci'][1]:+.5f}] · {sh['verdict']}")
    cm = card.get("calibration_metrics")
    if cm:
        out.append(f"   - 보정 (Brier·ECE): AI {cm['brier_ai']:.3f}·{cm['ece_ai']}%p vs 배당 {cm['brier_market']:.3f}·{cm['ece_market']}%p "
                   f"(Brier 차이 {cm['brier_diff']:+.4f} [{cm['brier_diff_ci'][0]:+.4f}, {cm['brier_diff_ci'][1]:+.4f}])")
    for h in card.get("hypotheses") or []:
        out.append(f"   - 사전 가설 {h['id']} {h['name']}: {h['n']}/300경기 · {h['verdict']}")
    st = status_from_data(reader("data/walk_forward.json"), card, reader("data/paper_trading.json"))
    feats = st["features"]
    out.append(f"   - {'🔓 일부 추천 켜짐' if st['any_on'] else '🔒 추천 잠금 중'}: "
               + " · ".join(f"{'✅' if s['ok'] else '⏳'} {s['name']}" for s in st["stages"])
               + f" · 실제 돈 시험 💎 {feats['gem']['profit']['n']}/300건, 조합 찾기 {feats['finder']['profit']['n']}/300건")
    out.extend(paper_lines(reader))
    warns = health.get("warnings") or []
    out.append("🩺 운영 점검 (마지막 실행 기준): "
               + ("이상 없음" if not warns else " / ".join(str(w.get("message", w)) for w in warns))
               + (f" ({health.get('source')}, {str(health.get('checked_at', ''))[:16]})" if health else ""))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="대화 시작 브리핑")
    ap.add_argument("--no-tests", action="store_true", help="테스트는 건너뛰기")
    args = ap.parse_args(argv)
    print(f"📋 [대화 시작 브리핑] {datetime.now(KST):%Y-%m-%d %H:%M} KST")
    print(git_sync())
    if not args.no_tests:
        print(run_tests())
    print("\n".join(runs_status()))
    print("\n".join(scheduler_lines()))
    print(token_expiry_line())
    print("\n".join(data_intake_lines()))
    print("\n".join(contract_lines()))
    from evaluation.docs_sync import sync_lines
    print("\n".join(sync_lines()))
    print("\n".join(data_audit_lines()))
    print("\n".join(verdict_lines()))


if __name__ == "__main__":
    main()
