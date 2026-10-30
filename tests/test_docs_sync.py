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
