"""
evaluation/docs_sync.py
=======================
문서 동기화 점검 — 코드·워크플로·데이터 파일이 늘어도 문서가 따라가지 않아 낡는 것을 잡는다 (2026-10-30 도입).

규칙(SKILL.md·GEMINI.md '문서 최신화 규칙'): 새 모듈·워크플로·데이터 파일을 만들면 같은 커밋에서 ARCHITECTURE.md에 한 줄을 넣는다.
- undocumented(): ARCHITECTURE.md에 이름이 한 번도 안 나오는 모듈(collector/models/evaluation/tracker/notifier/dashboard),
  워크플로(.github/workflows/*.yml), data/ 바로 아래 파일
- stale_docs(): 문서의 마지막 수정 이후 코드 커밋이 너무 많이 쌓였는지 (git 이력, 문서마다 한도)
- broken_todo_links(): EXPERIMENTS.md의 TODO 줄 번호 링크(`[TODO 7.17](./TODO.md#L731)`)가 그 항목 줄을 안 가리키는 것
  (TODO에 항목이 늘면 줄 번호가 밀린다 — `--fix`가 항목 번호로 다시 찾아 고친다, 2026-11-01)
브리핑(`python -m evaluation.briefing`)이 매번 알려 준다. 표준 라이브러리만 쓴다.

실행:
    python -m evaluation.docs_sync          # 점검
    python -m evaluation.docs_sync --fix    # 밀린 TODO 링크 줄 번호 고치기
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ["collector", "models", "evaluation", "tracker", "notifier", "dashboard"]
ARCH = "ARCHITECTURE.md"
# 이 파일들은 문서 대상이 아니다 (패키지 표식 · 자동 생성 · 서로 다른 이름으로 이미 설명)
IGNORE_NAMES = {"__init__.py", ".gitkeep"}
# (문서, 마지막 수정 뒤 코드 커밋 한도): 설명 문서일수록 한도를 낮게
STALE_LIMITS = {"README.md": 40, "ARCHITECTURE.md": 40, "GEMINI.md": 60, ".agents/skills/sports-prediction-engine/SKILL.md": 60}
CODE_PATHS = ["collector", "models", "evaluation", "tracker", "notifier", "dashboard", "run_pipeline.py", ".github/workflows"]


def _read(rel: str, root: Path = ROOT) -> str:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def tracked_targets(root: Path = ROOT) -> Dict[str, List[str]]:
    """문서에 있어야 하는 이름 목록 {'모듈': [...], '워크플로': [...], '데이터 파일': [...]}"""
    mods = []
    for d in CODE_DIRS:
        for f in sorted((root / d).glob("*.py")) if (root / d).exists() else []:
            if f.name not in IGNORE_NAMES:
                mods.append(f"{d}/{f.name}")
    wfs = [f".github/workflows/{f.name}" for f in sorted((root / ".github" / "workflows").glob("*.yml"))] \
        if (root / ".github" / "workflows").exists() else []
    data = [f"data/{f.name}" for f in sorted((root / "data").glob("*")) if f.is_file() and f.name not in IGNORE_NAMES] \
        if (root / "data").exists() else []
    return {"모듈": mods, "워크플로": wfs, "데이터 파일": data}


def undocumented(root: Path = ROOT, doc: str = ARCH) -> Dict[str, List[str]]:
    """ARCHITECTURE.md에 파일 이름(확장자 포함, 폴더 경로는 있어도 없어도 됨)이 한 번도 안 나오는 것"""
    text = _read(doc, root)
    out: Dict[str, List[str]] = {}
    for kind, names in tracked_targets(root).items():
        missing = [n for n in names if Path(n).name not in text]
        if missing:
            out[kind] = missing
    return out


def _git(args: List[str], root: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def stale_docs(root: Path = ROOT, limits: Dict[str, int] = STALE_LIMITS) -> List[Dict[str, object]]:
    """문서 마지막 수정 이후 코드(CODE_PATHS) 커밋 수가 한도를 넘은 문서 (봇의 데이터 커밋은 data/라 세지 않음)"""
    out = []
    for rel, limit in limits.items():
        last = _git(["log", "-1", "--format=%H", "--", rel], root)
        if not last:
            continue
        n = _git(["rev-list", "--count", f"{last}..HEAD", "--", *CODE_PATHS], root)
        if n and n.isdigit() and int(n) > limit:
            out.append({"doc": rel, "commits": int(n), "limit": limit})
    return out


LINKED_DOCS = ["EXPERIMENTS.md"]   # TODO 항목으로 가는 줄 번호 링크를 쓰는 문서
_TODO_LINK = re.compile(r"\[([^\]]*?)\]\(\./TODO\.md#L(\d+)\)")
_TODO_ITEM = re.compile(r"- \[[ x]\] \*\*(\d+\.\d+) ")
PHASE6 = "Phase 6 결론"


def _todo_item_lines(todo_text: str) -> Dict[str, int]:
    """TODO 항목 번호 → 줄 번호 ('- [x] **7.17 …' 줄) + 'Phase 6 결론' 절"""
    out: Dict[str, int] = {}
    for i, line in enumerate(todo_text.splitlines(), 1):
        m = _TODO_ITEM.match(line)
        if m and m.group(1) not in out:
            out[m.group(1)] = i
        if f"**{PHASE6}" in line and PHASE6 not in out:
            out[PHASE6] = i
    return out


def _link_target(label: str, items: Dict[str, int]) -> Optional[int]:
    """링크 글자가 가리키려는 줄: 'Phase 6 …'은 Phase 6 결론, 그 밖에는 글자 속 항목 번호(범위 '5.6~5.7'이면 뒤 번호)"""
    if "Phase 6" in label:
        return items.get(PHASE6)
    nums = re.findall(r"\d+\.\d+", label)
    return items.get(nums[-1]) if nums else None


def broken_todo_links(root: Path = ROOT) -> List[Dict[str, object]]:
    """LINKED_DOCS 안 TODO 링크 중 그 항목 줄을 안 가리키는 것 (항목을 못 찾는 링크는 건너뜀)"""
    items = _todo_item_lines(_read("TODO.md", root))
    out = []
    for doc in LINKED_DOCS:
        for m in _TODO_LINK.finditer(_read(doc, root)):
            want = _link_target(m.group(1), items)
            if want and want != int(m.group(2)):
                out.append({"doc": doc, "label": m.group(1), "line": int(m.group(2)), "want": want})
    return out


def fix_todo_links(root: Path = ROOT) -> int:
    """밀린 TODO 링크 줄 번호를 항목 번호로 다시 찾아 고친다 (줄 끝 형식은 그대로). 고친 링크 수"""
    items = _todo_item_lines(_read("TODO.md", root))
    fixed = 0
    for doc in LINKED_DOCS:
        p = root / doc
        if not p.exists():
            continue
        text = p.read_bytes().decode("utf-8")

        def repl(m: "re.Match[str]") -> str:
            nonlocal fixed
            want = _link_target(m.group(1), items)
            if want and want != int(m.group(2)):
                fixed += 1
                return f"[{m.group(1)}](./TODO.md#L{want})"
            return m.group(0)

        new = _TODO_LINK.sub(repl, text)
        if new != text:
            p.write_bytes(new.encode("utf-8"))
    return fixed


def sync_lines(root: Path = ROOT) -> List[str]:
    """브리핑용 한 블록"""
    missing, stale, links = undocumented(root), stale_docs(root), broken_todo_links(root)
    if not missing and not stale and not links:
        return ["📚 문서 동기화: ✅ ARCHITECTURE.md에 모든 모듈·워크플로·데이터 파일이 있고, 문서가 코드보다 많이 뒤처지지 않았어요"]
    n_missing = sum(len(v) for v in missing.values())
    out = [f"📚 문서 동기화: ⚠️ 문서에 없는 항목 {n_missing}개 · 오래된 문서 {len(stale)}개 · 밀린 TODO 링크 {len(links)}개"
           " — 지금 최신화하세요 (SKILL.md '문서 최신화 규칙')"]
    for kind, names in missing.items():
        out.append(f"   - ARCHITECTURE.md에 없는 {kind}: {', '.join(names[:8])}" + (f" 외 {len(names) - 8}개" if len(names) > 8 else ""))
    for s in stale:
        out.append(f"   - {s['doc']}: 마지막 수정 뒤 코드 커밋 {s['commits']}개 (한도 {s['limit']})")
    if links:
        out.append(f"   - {', '.join(sorted({str(x['doc']) for x in links}))}의 TODO 링크 {len(links)}개가 다른 줄을 가리킴"
                   f" (예: '{links[0]['label']}' L{links[0]['line']} → L{links[0]['want']}) → `python -m evaluation.docs_sync --fix`")
    return out


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if "--fix" in sys.argv:
        print(f"🔗 밀린 TODO 링크 {fix_todo_links()}개를 고쳤어요")
    print("\n".join(sync_lines()))
