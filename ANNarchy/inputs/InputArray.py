"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import numpy

from ANNarchy.core.Neuron import Neuron
from ANNarchy.intern import Messages
from ANNarchy.intern.SpecificPopulation import SpecificPopulation


class InputArray(SpecificPopulation):
    """
    Population holding static inputs for a rate-coded network.

    The input values are stored in the recordable attribute `r`, without any further processing.

    ```python
    inp = net.create(ann.InputArray(geometry=10))
    inp.r = np.linspace(1, 10, 10)

    pop = net.create(100, ann.LeakyIntegrator)

    proj = net.connect(inp, pop, 'exc')
    proj.all_to_all(1.0)
    ```

    Note that this population is functionally equivalent to:

    ```python
    inp = net.create(geometry, ann.Neuron(parameters="r=0.0"))
    ```

    but `r` is recordable.

    :param geometry: shape of the population, either an integer or a tuple.
    :param name: optional name of the population.
    """

    def __init__(
        self,
        geometry: int|tuple|None = None,
        name: str|None = None,
        copied: bool = False,
        net_id=0,
    ):
        # Does nothing except declaring r as a variable to allow recording.
        neuron = Neuron(
            parameters="",
            equations="r=r",
            name="Input Array",
            description="Input array source.",
        )

        # Register Population
        super().__init__(
            geometry=geometry, neuron=neuron, name=name, copied=copied, net_id=net_id
        )

    #
    # Special properties
    #

    @property
    def r(self):
        if self.initialized:
            return self.cyInstance.r.reshape(self.geometry)
        else:
            return self.init["r"]

    @r.setter
    def r(self, value):
        raise Messages.NoDirectAssignment("InputArray: `r` can only be set during construction or via update()")

    #
    # User-interface
    #

    def update(self, r: numpy.ndarray|list[float]):
        """
        Update the input array.
        """
        # Sanity check
        if isinstance(r, list):
            if len(r) != self.size:
                Messages.error("InputArray: length of provided `r` does not fit to population size.")
        elif isinstance(r, numpy.ndarray):
            if r.shape != self.geometry:
                Messages.error("InputArray: shape of provided `r` does not fit to population geometry.")
            r = r.reshape(self.size)

        # Update values
        if self.initialized:
            self.cyInstance.r = r
        else:
            self.init["r"] = r

    #
    # Internal API
    #
    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return InputArray(
            geometry=self.geometry,
            name=self.name,
            copied=True,
            net_id=self.net_id if net_id is None else net_id,
        )

    #
    # Code generation
    #

    def _generate_st(self):
        """
        Adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template["update_variables"] = ""

    def _generate_omp(self):
        """
        Adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template["update_variables"] = ""

    def _generate_cuda(self):
        """
        Code generation if the CUDA paradigm is set.
        """
        self._specific_template["update_variable_body"] = ""
        self._specific_template["update_variable_invoke"] = ""
        self._specific_template["update_variable_header"] = ""
        self._specific_template["update_variable_call"] = ""
