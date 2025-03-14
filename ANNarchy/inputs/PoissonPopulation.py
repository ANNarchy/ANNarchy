"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.SpecificPopulation import SpecificPopulation
from ANNarchy.intern import Messages
from ANNarchy.core.Neuron import Neuron


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
    pop.rates = np.linspace(10, 150, 100)
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

    rates = 10.*np.ones((2, 100))
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

    def __init__(self, 
                 geometry:int|tuple[int], 
                 name:str=None, 
                 rates:float|str=None, 
                 target:str=None, 
                 parameters:dict={}, 
                 refractory:float=None, 
                 copied:bool=False,
                 net_id:int = 0):

        if rates is None and target is None:
            Messages._error('A PoissonPopulation must define either rates or target.')

        self.target = target
        self.parameters_init = parameters
        self.refractory_init = refractory
        self.rates_init = rates

        if target is not None: # hybrid population
            # Create the neuron
            poisson_neuron = Neuron(
                parameters = parameters,
                equations = """
                    rates = sum(%(target)s)
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                    _sum_%(target)s = 0.0
                """ % {'target': target},
                spike = """
                    p < rates
                """,
                refractory=refractory,
                name="Hybrid",
                description="Hybrid spiking neuron emitting spikes according to a Poisson distribution at a frequency determined by the weighted sum of inputs."
            )


        elif isinstance(rates, str):
            # Create the neuron
            poisson_neuron = Neuron(
                parameters = parameters,
                equations = """
                    rates = %(rates)s
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                    _sum_exc = 0.0
                """ % {'rates': rates},
                spike = """
                    p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )

        elif isinstance(rates, np.ndarray):
            poisson_neuron = Neuron(
                parameters = """
                rates = 10.0
                """,
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )
        else:
            poisson_neuron = Neuron(
                parameters = """
                rates = %(rates)s
                """ % {'rates': rates},
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )
        SpecificPopulation.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name, copied=copied, net_id=net_id)

        if isinstance(rates, np.ndarray):
            self.rates = rates

    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return PoissonPopulation(
            self.geometry, name=self.name, 
            rates=self.rates_init, target=self.target, 
            parameters=self.parameters_init, 
            refractory=self.refractory_init, 
            copied=True,
            net_id = self.net_id if not net_id else net_id,
        )

    def _generate_st(self):
        """
        Generate single thread code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
        pass

    def _generate_omp(self):
        """
        Generate openMP code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
        pass

    def _generate_cuda(self):
        """
        Generate CUDA code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
        pass