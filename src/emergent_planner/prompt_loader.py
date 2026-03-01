"""Prompt library loader with profile-aware merge/replace overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .config import AgentConfig, AgentProfileConfig, PromptCardConfig
from .models import PromptCard, PromptLibrary
from .prompts import make_default_prompt_lib


def _load_card_text(card_cfg: PromptCardConfig, *, profile_id: str, config_dir: Path) -> str:
    has_text = card_cfg.text is not None
    has_file = card_cfg.file is not None and str(card_cfg.file).strip() != ""
    if has_text == has_file:
        raise ValueError(
            f"Invalid prompt card '{card_cfg.name}' in profile '{profile_id}': exactly one of text or file is required"
        )

    if has_text:
        return str(card_cfg.text)

    p = Path(str(card_cfg.file)).expanduser()
    if not p.is_absolute():
        p = (config_dir / p).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt card file not found for '{card_cfg.name}' in profile '{profile_id}': {p.as_posix()}"
        )
    return p.read_text(encoding="utf-8")


def _to_prompt_card(card_cfg: PromptCardConfig, *, profile_id: str, config_dir: Path) -> PromptCard:
    return PromptCard(
        name=card_cfg.name,
        text=_load_card_text(card_cfg, profile_id=profile_id, config_dir=config_dir),
        tags=set(card_cfg.tags or []),
        priority=int(card_cfg.priority),
    )


def _merge_cards(base_cards: List[PromptCard], profile_cards: List[PromptCard], disabled: List[str]) -> List[PromptCard]:
    out = [c for c in base_cards if c.name not in set(disabled)]
    index: Dict[str, int] = {c.name: i for i, c in enumerate(out)}

    for card in profile_cards:
        if card.name in index:
            out[index[card.name]] = card
        else:
            out.append(card)
            index[card.name] = len(out) - 1

    return [c for c in out if c.name not in set(disabled)]


def build_prompt_lib_for_profile(cfg: AgentConfig, profile: AgentProfileConfig, *, config_dir: Path) -> PromptLibrary:
    """
    Build a PromptLibrary using profile-level prompt override settings.
    """
    prompt_cfg = profile.prompts
    strategy = str(prompt_cfg.strategy or "merge").strip().lower()
    if strategy not in {"merge", "replace"}:
        strategy = "merge"

    profile_cards = [
        _to_prompt_card(c, profile_id=profile.id, config_dir=config_dir)
        for c in list(prompt_cfg.cards or [])
    ]

    if strategy == "replace":
        cards = profile_cards
    else:
        base_cards = list(make_default_prompt_lib().cards)
        cards = _merge_cards(base_cards, profile_cards, list(prompt_cfg.disable_cards or []))

    return PromptLibrary(cards=cards)
