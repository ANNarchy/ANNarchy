"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class RawNeuronFields:
    """
    Stores the original arguments passed to `ANNarchy.core.Neuron`,
    primarily for debugging purposes.
    """
    parameters: dict
    equations: str
    functions: str
    reset: str | None
    spike: str | None
