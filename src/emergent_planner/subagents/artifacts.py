"""
Persistence helpers for sub-agent outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def persist_task_artifact(
    *,
    artifact_root: Path,
    parent_run_id: str,
    request_id: str,
    task_id: str,
    payload: Dict[str, Any],
) -> Path:
    out_dir = artifact_root / parent_run_id / request_id
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task_id}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p
