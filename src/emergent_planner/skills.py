"""
Skill discovery and scoring utilities.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

from .models import SkillMeta


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


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


def _discover_skill_files(search_root: Path) -> List[Path]:
    """
    Discover SKILL.md files with project-root awareness.

    Behavior:
    - Scan search_root recursively when it exists.
    - Also scan all `.skills` directories recursively from project root.
    """
    # If caller passes an explicit non-.skills directory, respect that scope.
    if search_root.exists() and search_root.name != ".skills":
        return _skill_files_under_root(search_root)

    project_root = find_project_root(Path.cwd())
    out: List[Path] = []

    if search_root.exists():
        out.extend(_skill_files_under_root(search_root))

    for p in sorted(project_root.rglob(".skills")):
        if p.is_dir():
            out.extend(_skill_files_under_root(p))
    # Deduplicate while preserving deterministic order.
    seen = set()
    deduped: List[Path] = []
    for p in sorted(out):
        k = p.resolve().as_posix()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(p)
    return deduped


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
    skills: List[SkillMeta] = []
    for skill_file in _discover_skill_files(skills_root):
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
