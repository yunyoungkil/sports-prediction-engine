import json
from types import SimpleNamespace

import setup_skills as ss


def _lock(home, skills):
    p = ss.lock_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 3, "skills": skills}), encoding="utf-8")


def _skill(home, name, text="v1"):
    d = ss.agents_dir(home) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_sources_exclude_agentmemory_and_install_is_global_copy():
    assert not any("agentmemory" in s for s in ss.SOURCES)
    cmd = ss.install_cmd("obra/superpowers")
    assert cmd[:5] == ["npx", "-y", "skills", "add", "obra/superpowers"]
    for flag in ("-g", "--copy", "gemini-cli"):
        assert flag in cmd
    assert "claude-code" not in cmd   # Claude 쪽은 계정 플러그인이 깐다


def test_skills_from_filters_by_source(tmp_path):
    _lock(tmp_path, {"humanizer": {"source": "DaleSeo/korean-skills"},
                     "hyperframes": {"source": "heygen-com/hyperframes"},
                     "watch": {"source": "bradautomates/claude-video"}})
    assert ss.skills_from(ss.lock_path(tmp_path), ss.SOURCES) == ["humanizer", "watch"]
    assert ss.skills_from(tmp_path / "없음.json", ss.SOURCES) == []


def test_copy_to_antigravity_replaces_old_copy(tmp_path):
    _skill(tmp_path, "humanizer", "v2")
    dst = ss.antigravity_dir(tmp_path)
    (dst / "humanizer").mkdir(parents=True)
    (dst / "humanizer" / "SKILL.md").write_text("v1", encoding="utf-8")
    (dst / "humanizer" / "old.txt").write_text("x", encoding="utf-8")
    done = ss.copy_to_antigravity(["humanizer", "없는스킬"], ss.agents_dir(tmp_path), dst)
    assert done == ["humanizer"]
    assert (dst / "humanizer" / "SKILL.md").read_text(encoding="utf-8") == "v2"
    assert not (dst / "humanizer" / "old.txt").exists()


def test_dry_run_changes_nothing(tmp_path):
    calls = []
    assert ss.main(["--dry-run"], home=tmp_path, run=lambda *a, **k: calls.append(a)) == 0
    assert calls == [] and not ss.antigravity_dir(tmp_path).exists()


def test_main_installs_then_copies_and_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.shutil, "which", lambda _: "npx")
    _lock(tmp_path, {"watch": {"source": "bradautomates/claude-video"}})
    _skill(tmp_path, "watch")
    seen = []

    def fake_run(cmd, cwd, env):
        seen.append((cmd[4], cwd, env.get("DO_NOT_TRACK")))
        return SimpleNamespace(returncode=1 if cmd[4] == "obra/superpowers" else 0)

    assert ss.main([], home=tmp_path, run=fake_run) == 1          # 하나라도 실패하면 1
    assert [s for s, _, _ in seen] == ss.SOURCES
    assert all(cwd == str(tmp_path) and dnt == "1" for _, cwd, dnt in seen)
    assert (ss.antigravity_dir(tmp_path) / "watch" / "SKILL.md").exists()
