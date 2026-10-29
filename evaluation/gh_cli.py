"""
evaluation/gh_cli.py
====================
GitHub CLI(gh) 실행 파일 위치 찾기. 설치 직후 이미 열려 있던 터미널·대시보드 프로세스는 PATH가 갱신되지 않아
`gh`라는 이름만으로는 못 찾는다 → PATH에서 찾고, 없으면 Windows 기본 설치 위치를 쓴다.
"""

import shutil
from pathlib import Path

WINDOWS_DEFAULT = Path("C:/Program Files/GitHub CLI/gh.exe")


def gh_path() -> str:
    found = shutil.which("gh")
    if found:
        return found
    return str(WINDOWS_DEFAULT) if WINDOWS_DEFAULT.exists() else "gh"
