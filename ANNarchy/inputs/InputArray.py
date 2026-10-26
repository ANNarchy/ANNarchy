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

    The input values are stored in the recordable attribute `r`, without any further processing. They are set or changed with the `update()` method:

    ```python
    inp = net.create(ann.InputArray(geometry=10))

    pop = net.create(100, ann.LeakyIntegrator)

    proj = net.connect(inp, pop, 'exc')
    proj.all_to_all(1.0)

    net.compile()

    inp.update(r=np.linspace(1, 10, 10))

    net.simulate(100.)
    ```

    Note that this population is functionally equivalent to:

    ```python
    inp = net.create(geometry, ann.Neuron(parameters="r=0.0"))
    ```

    but `r` is recordable, and it can only be set with `update()`, not by a direct assignment such as `inp.r = 1.0`.

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
        raise Messages.NoDirectAssignment("InputArray: `r` can only be set via update()")

    #
    # User-interface
    #

    def update(self, r: numpy.ndarray|list[float]):
        """
        Update the input array.

        :param r: new input values, either a single value for all neurons or an array with the shape of the population.
        """
        # Sanity check
        if isinstance(r, list):
            if len(r) != self.size:
                Messages.error("InputArray: length of provided `r` does not fit to population size.")
        elif isinstance(r, numpy.ndarray):
            if r.shape != self.geometry:
                Messages.error("InputArray: shape of provided `r` does not fit to population geometry.")
            r = r.reshape(self.size)

        # single value -> list
        if isinstance(r, (int, float)):
            r = numpy.ones(self.size)*r

        # Update values
        if self.initialized:
            self.cyInstance.r = r
        else:
            self.init["r"] = r

    def reset(self, attributes: list|None = None) -> None:
        """
        Resets all parameters and variables of the population to the value they had before the call to `net.compile()`.

        :param attributes: list of attributes (parameter or variable) which should be reinitialized. Default: all attributes.

        Implementation note:

            Overload `Population.reset` as we need to set spike_times/period from hand.
        """
        # reset r
        if isinstance(self.init["r"], (int, float)):
            r = numpy.ones(self.size)*self.init["r"]
        else:
            r = self.init["r"]
        self.cyInstance.r = r

        # reset other properties
        super().reset(attributes)

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

    def _init_attributes(self):
        """
        Method used after compilation to initialize the attributes.

        Implementation note:

            Overload `Population._init_attributes` as we need to initialize r by hand.
        """
        super()._init_attributes()

        # initialize r, the parent function would call pop.r = self.init["r"]
        # which results in a NoDirectAssignment error and skips the initialization
        if isinstance(self.init["r"], (int, float)):
            r = numpy.ones(self.size)*self.init["r"]
        else:
            r = self.init["r"]
        self.cyInstance.r = r


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
