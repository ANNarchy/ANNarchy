"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpikeProcessing:
    """
    Stores information and code snippets related to spike events.
    """
    condition: str
    condition_dependencies: list[str]
    reset: list[dict[str, Any]]
