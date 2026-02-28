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


def discover_skills(skills_root: Path = Path(".skills")) -> List[SkillMeta]:
    """Scan <skills_root>/<dir>/SKILL.md files and return a list of SkillMeta."""
    skills: List[SkillMeta] = []
    if not skills_root.exists():
        return skills

    for d in sorted(skills_root.iterdir()):
        if not d.is_dir():
            continue
        skill_file = d / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8")
        sk = parse_skill_md(text, skill_file)
        sk.body = None  # registry only; body loaded on-demand via load_skill tool
        skills.append(sk)
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
