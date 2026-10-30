"""Automatically expose project-local dependencies for the active Python version."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_TAG = f"py{sys.version_info.major}{sys.version_info.minor}"

for deps_dir in sorted(PROJECT_ROOT.glob(f".pydeps-{VERSION_TAG}*")):
    if deps_dir.is_dir():
        deps_path = str(deps_dir)
        if deps_path not in sys.path:
            sys.path.insert(0, deps_path)
