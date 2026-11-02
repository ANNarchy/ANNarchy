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
    functions: str | None
    reset: str | None
    spike: str | None

@dataclass(frozen=True)
class RawSynapseFields:
    """
    Stores the original arguments passed to `ANNarchy.core.Synapse`,
    primarily for debugging purposes.
    """
    parameters: dict
    equations: str
    functions: str | None
    pre_spike: str
    post_spike: str | None
    axon_spike: str | None