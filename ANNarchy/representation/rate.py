"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass

from .attributes import AttributeRepresentation


@dataclass(slots=True)
class RatePSP(AttributeRepresentation):
    """
    Stores information and code snippets related to postsynaptic potentials, which are
    part of the continuous signal transmission between neurons.

    Implementation note (Nov. 3rd, 2026):
    
        A dictionary seems to be required for expressions such as sum("FF"),
        but only in the GPU compilation path (Population/CUDAGenerator, _get_attr_and_type).
    """