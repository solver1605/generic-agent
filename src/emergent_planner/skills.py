"""
Skill discovery and scoring utilities.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import yaml

from .models import SkillMeta


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def normalize_skill_key(name: str) -> str:
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def find_project_root(start: Path = Path.cwd()) -> Path:
    """
    Resolve project root by walking up until `.git` is found.
    If not found, return the provided start directory.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return start.resolve() if start.exists() else cur
        cur = cur.parent


def _skill_files_under_root(root: Path) -> List[Path]:
    """
    Collect SKILL.md files recursively under a root directory.
    """
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.name == "SKILL.md" else []
    return sorted([p for p in root.rglob("SKILL.md") if p.is_file()])


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in sorted(paths):
        k = p.resolve().as_posix()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _discover_skill_files(search_root: Path) -> List[Path]:
    """
    Discover SKILL.md files with project-root awareness.

    Behavior:
    - Scan search_root recursively when it exists.
    - Also scan all `.skills` directories recursively from project root.
    """
    if search_root.exists() and search_root.name != ".skills":
        return _skill_files_under_root(search_root)

    project_root = find_project_root(Path.cwd())
    out: List[Path] = []

    if search_root.exists():
        out.extend(_skill_files_under_root(search_root))

    for p in sorted(project_root.rglob(".skills")):
        if p.is_dir():
            out.extend(_skill_files_under_root(p))

    return _dedupe_paths(out)


def _parse_skill_files(skill_files: List[Path], *, include_body: bool) -> List[SkillMeta]:
    skills: List[SkillMeta] = []
    for skill_file in skill_files:
        try:
            text = skill_file.read_text(encoding="utf-8")
            sk = parse_skill_md(text, skill_file)
        except Exception:
            # Skip invalid SKILL.md files outside the skill contract.
            continue
        if not include_body:
            sk.body = None  # registry only; body loaded on-demand via load_skill tool
        skills.append(sk)
    skills.sort(key=lambda s: s.path.as_posix())
    return skills


def parse_skill_md(text: str, path: Path) -> SkillMeta:
    """Parse a SKILL.md file with YAML frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: Missing YAML frontmatter (--- ... ---)")
    fm_raw, body = m.group(1), m.group(2)
    meta = yaml.safe_load(fm_raw) or {}
    name = (meta.get("name") or "").strip()
    desc = (meta.get("description") or "").strip()
    if not name or not desc:
        raise ValueError(f"{path}: 'name' and 'description' are required in frontmatter")
    return SkillMeta(name=name, description=desc, path=path, meta=meta, body=body.strip())


def discover_skills(skills_root: Path = Path(".skills"), *, include_body: bool = False) -> List[SkillMeta]:
    """
    Discover SKILL.md files recursively and return SkillMeta records.

    - If `skills_root` exists, it is scanned recursively.
    - If `skills_root` does not exist, fallback scans `.skills` directories
      recursively from project root.
    """
    return _parse_skill_files(_discover_skill_files(skills_root), include_body=include_body)


def discover_skills_in_roots(
    roots: List[Path],
    *,
    include_body: bool = False,
    strict_scope: bool = True,
) -> List[SkillMeta]:
    """
    Discover skills across explicit roots.

    - strict_scope=True: scan only listed roots recursively.
    - strict_scope=False: if roots yield no files, fallback to project-wide discovery.
    """
    paths: List[Path] = []
    for root in roots or []:
        p = Path(root).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        paths.extend(_skill_files_under_root(p))

    deduped = _dedupe_paths(paths)
    if not deduped and not strict_scope:
        deduped = _discover_skill_files(Path(".skills"))

    return _parse_skill_files(deduped, include_body=include_body)


# ---------------------------------------------------------------------------
# Skills registry rendering (Top-K)
# ---------------------------------------------------------------------------

def score_skill(skill: SkillMeta, query: str) -> int:
    q = (query or "").lower()
    s = (skill.name + " " + skill.description).lower()
    score = 0
    for term in set(re.findall(r"[a-zA-Z0-9_]+", q)):
        if len(term) >= 3 and term in s:
            score += 2
    for tok in skill.name.lower().split():
        if tok in q:
            score += 3
    return score


def render_skills_topk(
    skills: List[SkillMeta],
    user_text: str,
    max_chars: int,
    k: int = 12,
) -> str:
    if not skills:
        return "Available skills: (none found)"
    ranked = sorted(skills, key=lambda sk: score_skill(sk, user_text), reverse=True)
    top = ranked[:k]
    lines = ["Available skills (load only when needed):"]
    for s in top:
        lines.append(f"- {s.name}: {s.description}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...[truncated]..."
    return out
