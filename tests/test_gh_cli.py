"""evaluation/gh_cli.py — PATH에 gh가 없어도(설치 직후 열려 있던 프로세스) 기본 설치 위치를 찾는다"""

from pathlib import Path

from evaluation import gh_cli


def test_gh_path_prefers_path(monkeypatch):
    monkeypatch.setattr(gh_cli.shutil, "which", lambda name: "/usr/bin/gh")
    assert gh_cli.gh_path() == "/usr/bin/gh"


def test_gh_path_falls_back_to_default_install_location(monkeypatch, tmp_path):
    fake = tmp_path / "gh.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(gh_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(gh_cli, "WINDOWS_DEFAULT", fake)
    assert gh_cli.gh_path() == str(fake)


def test_gh_path_returns_plain_name_when_not_installed(monkeypatch):
    monkeypatch.setattr(gh_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(gh_cli, "WINDOWS_DEFAULT", Path("/definitely/not/here/gh.exe"))
    assert gh_cli.gh_path() == "gh"
