"""새 PC에서 제미나이(Antigravity·Gemini CLI)에 Claude와 같은 공용 스킬을 깐다 (PC마다 한 번).

Claude Code 쪽은 Claude 계정에 등록한 플러그인이 같은 계정으로 로그인한 PC에 저절로 깔리므로 여기서 하지 않는다.

실행:  python setup_skills.py            # 설치 + Antigravity 폴더로 복사
       python setup_skills.py --dry-run  # 할 일만 보여 주고 아무것도 바꾸지 않음
필요:  Node.js 20 이상(npx). 저장소 파일·data/는 건드리지 않는다 (홈 폴더에만 설치).

어디에 들어가나:
  ~/.agents/skills/        skills 도구가 까는 공용 폴더 — Gemini CLI가 읽는다
  ~/.gemini/config/skills/ Antigravity 1.4 이상의 전역 스킬 폴더 — 위 폴더에서 복사한다
                           (프로젝트 안 .agents/skills/는 Antigravity의 작업 공간 스킬 폴더라 여기엔 두지 않는다)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# 2026-11-03 사용자가 고른 목록. agentmemory는 뺐다 (기존 기억 파일과 겹치고, 윈도우에서는 엔진·서버를 따로 띄워야 함).
SOURCES = [
    "daleseo/korean-skills",                 # 한국어 맞춤법·문체·AI 말투 고치기 (3개)
    "nextlevelbuilder/ui-ux-pro-max-skill",  # 화면·디자인 (7개)
    "multica-ai/andrej-karpathy-skills",     # 코딩 실수 줄이기 (1개)
    "obra/superpowers",                      # 계획·디버깅·검증 작업 방식 (15개)
    "bradautomates/claude-video",            # 동영상 보기 watch (1개)
]


def agents_dir(home: Path) -> Path:
    return home / ".agents" / "skills"


def antigravity_dir(home: Path) -> Path:
    return home / ".gemini" / "config" / "skills"


def lock_path(home: Path) -> Path:
    return home / ".agents" / ".skill-lock.json"


def install_cmd(source: str, npx: str = "npx") -> List[str]:
    """전역(-g)·Gemini CLI용·전체 스킬·복사본으로 설치하는 명령 (Gemini CLI는 ~/.agents/skills를 읽는다)"""
    return [npx, "-y", "skills", "add", source, "-g", "-a", "gemini-cli", "-s", "*", "-y", "--copy"]


def skills_from(lock_file: Path, sources: Iterable[str]) -> List[str]:
    """skills 도구의 전역 기록 파일에서 고른 출처의 스킬 이름만 고른다 (출처 대소문자 무시)"""
    if not lock_file.exists():
        return []
    wanted = {s.lower() for s in sources}
    data = json.loads(lock_file.read_text(encoding="utf-8"))
    return sorted(name for name, info in (data.get("skills") or {}).items()
                  if str(info.get("source", "")).lower() in wanted)


def copy_to_antigravity(names: Iterable[str], src_dir: Path, dst_dir: Path) -> List[str]:
    """공용 폴더의 스킬을 Antigravity 폴더로 복사한다 (같은 이름은 새것으로 바꿈). 복사한 이름을 돌려준다."""
    done = []
    for name in names:
        src = src_dir / name
        if not (src / "SKILL.md").exists():
            continue
        dst = dst_dir / name
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        done.append(name)
    return done


def main(argv: List[str] = None, home: Path = None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description="제미나이(Antigravity·Gemini CLI)에 공용 스킬 설치")
    ap.add_argument("--dry-run", action="store_true", help="할 일만 보여 줌")
    args = ap.parse_args(argv)
    home = home or Path.home()

    npx = shutil.which("npx")
    if not npx and not args.dry_run:
        print("❌ npx가 없습니다. Node.js 20 이상을 먼저 설치하세요: https://nodejs.org")
        return 1
    env = dict(os.environ, DO_NOT_TRACK="1")   # skills 도구의 사용 통계 전송 끄기

    failed = []
    for src in SOURCES:
        cmd = install_cmd(src, npx or "npx")
        print(f"▶ {src}")
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        r = run(cmd, cwd=str(home), env=env)
        if r.returncode != 0:
            failed.append(src)

    names = skills_from(lock_path(home), SOURCES)
    if args.dry_run:
        print(f"▶ Antigravity 폴더로 복사할 스킬 {len(names)}개 → {antigravity_dir(home)}")
        return 0
    antigravity_dir(home).mkdir(parents=True, exist_ok=True)
    copied = copy_to_antigravity(names, agents_dir(home), antigravity_dir(home))
    print(f"✅ Gemini CLI: {agents_dir(home)} · Antigravity: {antigravity_dir(home)} ({len(copied)}개)")
    if failed:
        print("⚠️ 설치 실패: " + ", ".join(failed) + " — 인터넷 연결을 확인하고 다시 실행하세요")
        return 1
    print("   Antigravity·Gemini CLI를 다시 열면 스킬이 보입니다. Claude Code는 같은 Claude 계정으로 로그인하면 저절로 깔립니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
