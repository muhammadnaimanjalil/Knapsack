"""JSON decoding for uploads and local files.

This module only handles transport and syntax. Domain rules are intentionally
kept in ``data.validation`` so every caller follows the same validation path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stochastic_knapsack.domain.exceptions import InputDataError


def load_json_bytes(content: bytes, source_name: str) -> Any:
    """Decode one UTF-8 JSON document and attach its name to useful errors."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{source_name} must be UTF-8 encoded.") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputDataError(
            f"{source_name} is not valid JSON (line {exc.lineno}, column {exc.colno})."
        ) from exc


def load_problem_paths(items_path: Path, packages_path: Path) -> tuple[Any, Any]:
    """Read the two problem documents from disk without applying domain rules."""
    try:
        items_content = items_path.read_bytes()
        packages_content = packages_path.read_bytes()
    except OSError as exc:
        raise InputDataError(f"Could not read the input files: {exc}") from exc
    return (
        load_json_bytes(items_content, items_path.name),
        load_json_bytes(packages_content, packages_path.name),
    )
