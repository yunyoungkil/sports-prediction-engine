"""CLAUDE.md · GEMINI.md의 '핵심 규칙' 블록 — 두 파일에서 같아야 하고, 짧게 유지돼야 한다 (긴 문서는 중간을 놓치기 때문)"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- CORE-RULES:START -->", "<!-- CORE-RULES:END -->"


def _block(name: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert START in text and END in text, f"{name}에 핵심 규칙 블록 표시가 없다"
    return text[text.index(START): text.index(END) + len(END)]


def test_core_rules_are_identical_in_claude_and_gemini_docs():
    assert _block("CLAUDE.md") == _block("GEMINI.md"), "CLAUDE.md와 GEMINI.md의 핵심 규칙 블록이 다르다 — 같이 고쳐야 한다"


def test_core_rules_stay_short_enough_to_be_read():
    block = _block("CLAUDE.md")
    rules = re.findall(r"^\d+\. ", block, flags=re.M)
    assert 5 <= len(rules) <= 12, "핵심 규칙은 12개 이하로 (많으면 중간을 놓친다)"
    assert all(len(line) <= 420 for line in block.splitlines()), "규칙 한 줄이 너무 길다 (420자 이하)"


def test_claude_md_is_short_and_points_to_the_detail_docs():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 60
    for doc in ("GEMINI.md", "SKILL.md", "ARCHITECTURE.md", "TODO.md", "REVIEW.md", "README.md"):
        assert doc in text


def test_the_most_important_rules_are_in_the_block():
    block = _block("CLAUDE.md")
    for must in ("evaluation.briefing", "git pull --rebase", "되돌릴 수 없는", "사전 등록", "미래 정보", "docs_sync", "pytest"):
        assert must in block, must
