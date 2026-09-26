# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Central path resolution: all relative paths anchor to the project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# MARL_code/ (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Offline vault datasets (download_and_unzip_vault default base)
VAULT_DIR = PROJECT_ROOT / "vaults"

# Default artifact directories under project root
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a config path relative to PROJECT_ROOT (or return absolute paths as-is)."""
    if path is None:
        return None
    text = str(path).strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    p = Path(text)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def ensure_project_root() -> str:
    """Change cwd to project root so legacy relative paths remain valid."""
    root = str(PROJECT_ROOT)
    os.chdir(root)
    return root
