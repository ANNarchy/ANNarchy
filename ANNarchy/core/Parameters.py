"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass

@dataclass
class localparam:
    value: float | int | bool
    type: str = 'float'

@dataclass
class globalparam:
    value: float | int | bool
    type: str = 'float'