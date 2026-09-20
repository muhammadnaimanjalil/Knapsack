"""JSON serialization for downloadable typed result objects."""

from __future__ import annotations

import json
from typing import Any


def result_json(result: Any) -> str:
    """Serialize a result dataclass using its public dictionary contract."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
