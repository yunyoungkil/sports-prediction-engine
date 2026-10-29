"""대시보드 코드 정적 검사 — Streamlit '매직'(값만 있는 줄을 화면에 찍음)에 걸리는 줄 막기

2026-10-29: `st.caption(msg) if where == "side" else st.info(msg)` 한 줄이 매직에 걸려,
④ 탭에서 금액을 고치면 사이드바에 DeltaGenerator 내부 설명이 쏟아졌다. 조건식·and/or 식을 문장으로 쓰지 말고 if/else로."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def test_no_conditional_expression_statements():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    bad = [f"{n.lineno}: {ast.get_source_segment(APP.read_text(encoding='utf-8'), n)[:80]}"
           for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, (ast.IfExp, ast.BoolOp))]
    assert not bad, "if/else 문으로 바꾸세요 (Streamlit 매직이 값을 화면에 찍음):\n" + "\n".join(bad)
