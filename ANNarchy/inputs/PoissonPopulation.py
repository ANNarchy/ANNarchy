"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy

from ANNarchy.core.Attributes import Parameter
from ANNarchy.core.Neuron import Neuron
from ANNarchy.intern import Messages
from ANNarchy.intern.SpecificPopulation import SpecificPopulation


class PoissonPopulation(SpecificPopulation):
    """
    Population of spiking neurons following a Poisson distribution.

    Each neuron of the population will randomly emit spikes, with a mean firing rate defined by the *rates* argument.

    The mean firing rate in Hz can be a fixed value for all neurons:

    ```python
    pop = net.create(ann.PoissonPopulation(geometry=100, rates=100.0))
    ```

    but it can be modified later as a normal parameter:

    ```python
    pop.update(rates = numpy.linspace(10, 150, 100))
    ```

    It is also possible to define a temporal equation for the rates, by passing a string to the argument:

    ```python
    pop = net.create(
        ann.PoissonPopulation(
            geometry=100,
            rates="100.0 * (1.0 + sin(2*pi*t/1000.0) )/2.0"
        )
    )
    ```

    The syntax of this equation follows the same structure as neural variables.

    It is also possible to add parameters to the population which can be used in the equation of `rates`:

    ```python
    pop = net.create(
        ann.PoissonPopulation(
            geometry=100,
            parameters = dict(
                amp = 100.0,
                frequency = 1.0,
            ),
            rates="amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0"
        )
    )
    ```

    **Note:** The preceding definition is fully equivalent to the definition of this neuron:

    ```python
    poisson = ann.Neuron(
        parameters = dict(
            amp = 100.0,
            frequency = 1.0,
        ),
        equations = [
            'rates = amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0',
            'p = Uniform(0.0, 1.0) * 1000.0 / dt',
        ],
        spike = "p < rates"
    )
    ```

    The refractory period can also be set, so that a neuron can not emit two spikes too close from each other.

    If the ``rates`` argument is not set, the population can be used as an interface from a rate-coded population. The ``target`` argument specifies which incoming projections will be summed to determine the instantaneous firing rate of each neuron.

    ```python
    net = ann.Network()

    rates = 10.*numpy.ones((2, 100))
    rates[0, :50] = 100.
    rates[1, 50:] = 100.
    inp = net.create(ann.TimedArray(rates = rates, schedule=50.))

    pop = net.create(ann.PoissonPopulation(100, target="exc"))

    proj = net.connect(inp, pop, 'exc')
    proj.one_to_one(1.0)
    ```

    :param geometry: population geometry as tuple.
    :param name: unique name of the population (optional).
    :param rates: mean firing rate of each neuron. It can be a single value (e.g. 10.0) or an equation (as string).
    :param target: the mean firing rate will be the weighted sum of inputs having this target name (e.g. "exc").
    :param parameters: additional parameters which can be used in the `rates` equation.
    :param refractory: refractory period in ms.
    """

    def __init__(
        self,
        geometry: int | tuple[int],
        name: str|None = None,
        rates: float|str|None = None,
        target: str|None = None,
        parameters: str | dict = "",
        refractory: float|None = None,
        copied: bool = False,
        net_id: int = 0,
    ):
        if rates is None and target is None:
            Messages.error("A PoissonPopulation must define either rates or target.")

        self.target = target
        self.parameters_init = parameters
        self.refractory_init = refractory
        self.rates_init = rates

        if target is not None:
            # hybrid population: rates depends on the pre-synaptic input
            #                    projected onto target
            poisson_neuron = Neuron(
                parameters=parameters,
                equations=f"""
                    rates = sum({target})
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                    _sum_{target} = 0.0
                """,
                spike="""
                    p < rates
                """,
                refractory=refractory,
                name="Hybrid",
                description="Hybrid spiking neuron emitting spikes according to a Poisson distribution at a frequency determined by the weighted sum of inputs.",
            )

        elif isinstance(rates, str):
            # rates depends on an equation, other dependencies being stored in
            # the `parameters` dict.
            poisson_neuron = Neuron(
                parameters=parameters,
                equations=f"""
                    rates = {rates}
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike="""
                    p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution.",
            )

        elif isinstance(rates, (float, int, numpy.ndarray)):
            # rates is initialized with a fixed value for all neurons (float/int)
            # or with a value for each individual neuron (numpy.ndarray)
            #
            # HD (Oct. 14, 2026): I use here a local variable on purpose to be flexible for later
            #                     re-initialization, even though it possibly wastes a bit of memory ...
            poisson_neuron = Neuron(
                parameters={
                    'rates': Parameter(rates, locality='local')
                },
                equations="""
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike="""
                    p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution.",
            )

        else:
            raise ValueError("PoissonPopulation: received unexpected type for `rates`:", type(rates))

        # Register Population
        super().__init__(
            geometry=geometry,
            neuron=poisson_neuron,
            name=name,
            copied=copied,
            net_id=net_id,
        )

        # Set the rates if necessary
        if isinstance(rates, (numpy.ndarray, float, int)):
            self.update(rates=rates)

    #
    # User-interface
    #

    def update(
        self,
        rates
    ):
        """
        Re-initialize the rates.

        :param rates: mean firing rate of each neuron. It can be a single value for all or a numpy.ndarray storing individual values.
        """
        if isinstance(rates, str):
            raise Messages.error("PoissonPopulation: providing an equation `rates` is only allowed on construction.")

        if isinstance(rates, list):
            rates = numpy.ndarray(rates)

        if isinstance(rates, numpy.ndarray) and rates.shape != self.geometry:
            raise Messages.error("PoissonPopulation: `rates` must have the same dimension as the population.")

        if self.initialized:
            if isinstance(self.rates_init, str):
                pass # nothing to do here
            elif isinstance(self.rates_init, (numpy.ndarray)):
                self.cyInstance.rates = self.rates_init
            elif isinstance(self.rates_init, (int, float)):
                self.cyInstance.rates = numpy.ones(self.geometry) * self.rates_init
            else:
                raise TypeError("PoissonPopulation: something went wrong ...")

        else:
            self.rates_init = rates

    def reset(self, attributes: list|None = None) -> None:
        """
        Resets all parameters and variables of the population to the value they had before the call to `net.compile()`.

        :param attributes: list of attributes (parameter or variable) which should be reinitialized. Default: all attributes.
        """
        # Implementation note:
        # - rates does not change over time -> no reset necessary
        # - other values provided within *parameters* are reset by parent class.

        super().reset(attributes=attributes)

    #
    # Internal API - object handling
    #

    def _copy(self, net_id=None):
        """
        Returns a copy of the population when creating networks.
        """
        return PoissonPopulation(
            self.geometry,
            name=self.name,
            rates=self.rates_init,
            target=self.target,
            parameters=self.parameters_init,
            refractory=self.refractory_init,
            copied=True,
            net_id=self.net_id if not net_id else net_id,
        )

    def _init_attributes(self):
        """
        Method used after compilation to initialize the attributes.
        """
        super()._init_attributes()

        # Set the rates if necessary
        if isinstance(self.rates_init, (numpy.ndarray, float, int)):
            self.update(rates=self.rates_init)

    def __getattr__(self, name):
        """
        Method called when reading an attribute.
        """
        if name == "rates":
            if self.initialized:
                return self.cyInstance.rates
            else:
                return self.rates_init

        else:
            return super().__getattr__(name=name)

    def __setattr__(self, name, value):
        """
        Method called when setting an attribute.
        """
        if name == "rates":
            raise Messages.NoDirectAssignment("PoissonPopulation: `rates` can only be set during construction or via update()")

        else:
            super().__setattr__(name=name, value=value)

    #
    # Internal API - code generation
    #

    def _generate_st(self):
        """
        Generate single thread code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """

    def _generate_omp(self):
        """
        Generate openMP code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """

    def _generate_cuda(self):
        """
        Generate CUDA code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
