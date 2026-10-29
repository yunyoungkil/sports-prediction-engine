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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
KST = timezone(timedelta(hours=9))
WORKFLOWS = [("pipeline.yml", "파이프라인 (하루 2번)"), ("results.yml", "결과 갱신 (1시간 예약, 실제 4~5시간)"),
             ("test.yml", "GitHub 테스트 (푸시 때)")]


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
        r = _run(["gh", "run", "list", "--workflow", wf, "--limit", "5", "--json", "conclusion,status,createdAt"], timeout=30)
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
        if not s.get("bets"):
            out.append(f"   - {s['label']}: 채점된 베팅 없음{pend}")
            continue
        out.append(f"   - {s['label']}: {s['bets']}건 중 {s['hits']}건 적중{pend} · 손익 {s['profit']:+,}원 "
                   f"(수익률 {s['roi_pct']:+.1f}%) · {s.get('verdict', '')}")
    return out


def verdict_lines(reader: Callable[[str], Optional[Dict[str, Any]]] = read_json) -> List[str]:
    from evaluation.gates import status_from_data
    card = reader("data/model_scorecard.json") or {}
    stale = reader("data/stale_odds_report.json") or {}
    health = reader("data/health_report.json") or {}
    out = ["📊 판정 진행 (미리 정한 기준):"]
    out.append(f"   - AI 실전 채점: {card.get('graded_matches', 0)}/300경기 · {card.get('verdict', '기록 없음')}")
    v = stale.get("value") or {}
    out.append(f"   - 지연 배당 (베트맨이 해외 배당을 늦게 따라가나): {v.get('matches', 0)}/100경기 · "
               f"{stale.get('verdict', '아직 기록 없음')}")
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
    print("\n".join(verdict_lines()))


if __name__ == "__main__":
    main()
