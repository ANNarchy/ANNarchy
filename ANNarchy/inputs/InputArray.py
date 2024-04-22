"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
from ANNarchy.intern.SpecificPopulation import SpecificPopulation

class InputArray(SpecificPopulation):
    """
    Population holding static inputs for a rate-coded network.

    The input values are stored in the recordable attribute `r`, without any further processing.

    ```python
    inp = ann.FixedArray(geometry=10)
    inp.r = np.linspace(1, 10, 10)

    pop = ann.Population(100, ...)

    proj = ann.Projection(inp, pop, 'exc')
    proj.connect_all_to_all(1.0)
    ```

    Note that this population is functionally equivalent to:

    ```python
    inp = ann.Population(geometry, ann.Neuron(parameters="r=0.0"))
    ```

    :param geometry: shape of the population, either an integer or a tuple.

    """
    def __init__(self, geometry: int|tuple=None, name:str=None, copied:bool=False):

        # Does nothing except declaring r as a variable to allow recording.
        neuron = Neuron(
            parameters="",
            equations="r=r",
            name="Fixed Array",
            description="Fixed array source."
        )

        SpecificPopulation.__init__(self, 
            geometry=geometry,  
            neuron=neuron, name=name, 
            copied= copied
        )


    def _generate_st(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['update_variables'] = ""

    def _generate_omp(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['update_variables'] = ""

    def _generate_cuda(self):
        """
        Code generation if the CUDA paradigm is set.
        """
        self._specific_template['update_variables'] = ""