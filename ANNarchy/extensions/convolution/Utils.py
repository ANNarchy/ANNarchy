"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Synapse import Synapse

class SharedSynapse(Synapse):
    """
    Shared synapse for report()
    """
    # For reporting
    _instantiated = []

    def __init__(self, psp, operation, name="Shared synapse", description="Weight shared over all synapses of the projection."):
        """
        """
        # Shared synapses are non-plastic.
        Synapse.__init__(self, 
            psp=psp, operation=operation,
            name=name, 
            description=description
        )

        # For reporting
        self._instantiated.append(True)