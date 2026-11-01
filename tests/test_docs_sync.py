"""evaluation/docs_sync.py — 문서가 코드·워크플로·데이터 파일을 따라가는지 점검"""

from pathlib import Path

from evaluation.docs_sync import ROOT, stale_docs, sync_lines, tracked_targets, undocumented


def _mini_repo(tmp_path: Path, arch_text: str) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "alpha.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "models" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "run.yml").write_text("name: x", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "out.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text(arch_text, encoding="utf-8")
    return tmp_path


def test_missing_items_are_reported_by_kind(tmp_path):
    root = _mini_repo(tmp_path, "문서에는 아무것도 없음")
    miss = undocumented(root)
    assert miss == {"모듈": ["models/alpha.py"], "워크플로": [".github/workflows/run.yml"], "데이터 파일": ["data/out.json"]}


def test_documented_items_pass_and_init_is_ignored(tmp_path):
    root = _mini_repo(tmp_path, "`models/alpha.py` `.github/workflows/run.yml` `data/out.json`")
    assert undocumented(root) == {}
    assert "models/__init__.py" not in tracked_targets(root)["모듈"]


def test_file_name_alone_is_enough(tmp_path):
    root = _mini_repo(tmp_path, "alpha.py · run.yml · out.json")
    assert undocumented(root) == {}


def test_sync_lines_warn_with_the_rule_pointer(tmp_path):
    root = _mini_repo(tmp_path, "비어 있음")
    text = "\n".join(sync_lines(root))
    assert "⚠️" in text and "models/alpha.py" in text and "문서 최신화 규칙" in text


def test_real_repo_docs_are_in_sync():
    """실제 저장소: 새 모듈·워크플로·data/ 파일을 만들고 ARCHITECTURE.md에 안 적으면 이 테스트가 실패한다 (문서 최신화 규칙 강제)"""
    assert undocumented(ROOT) == {}, f"ARCHITECTURE.md에 없는 항목: {undocumented(ROOT)} — 같은 커밋에서 문서에 한 줄 추가하세요"


def test_stale_docs_handles_missing_git(tmp_path):
    assert stale_docs(tmp_path, {"nonexistent.md": 1}) == []


def test_todo_links_are_checked_and_fixed_by_item_number(tmp_path):
    """EXPERIMENTS.md의 TODO 줄 번호 링크가 밀리면 점검이 잡고, --fix가 항목 번호로 다시 찾아 고친다 (줄 끝 형식 유지)"""
    from evaluation.docs_sync import broken_todo_links, fix_todo_links
    root = _mini_repo(tmp_path, "`models/alpha.py` `.github/workflows/run.yml` `data/out.json`")
    (root / "TODO.md").write_bytes("# TODO\r\n- [x] **5.6 a**\r\n- [x] **5.7 b**\r\n✅ **Phase 6 결론 (x):**\r\n- [ ] **7.2 c**\r\n".encode("utf-8"))
    (root / "EXPERIMENTS.md").write_bytes(
        "| [TODO 5.6~5.7](./TODO.md#L3) · [Phase 6 결론](./TODO.md#L9) · [7.2](./TODO.md#L2) · [없는 9.9](./TODO.md#L1) |\r\n".encode("utf-8"))
    bad = broken_todo_links(root)
    assert {(b["label"], b["want"]) for b in bad} == {("Phase 6 결론", 4), ("7.2", 5)}   # 5.6~5.7은 뒤 번호(5.7, L3)라 맞음
    assert any("밀린 TODO 링크 2개" in line for line in sync_lines(root))
    assert fix_todo_links(root) == 2 and broken_todo_links(root) == []
    text = (root / "EXPERIMENTS.md").read_bytes().decode("utf-8")
    assert "(./TODO.md#L4)" in text and "(./TODO.md#L5)" in text and "\r\n" in text and "[없는 9.9](./TODO.md#L1)" in text
