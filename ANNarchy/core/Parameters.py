"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass
from ANNarchy.core.Random import RandomDistribution

# Parameters

@dataclass
class parameter:
    """
    Dataclass to represent a parameter in a Neuron or Synapse definition.

    ```python
    neuron = ann.Neuron(
        parameters = dict(
            tau = ann.parameter(value=10.0, locality='global', type=float)
            baseline = ann.parameter(value=ann.Uniform(-1., 1.), locality='local', type=float)
            activated = ann.parameter(value=True, locality='global', type=bool)
        )
    )
    ```

    By default, parameters are global and use the float type, so tau could be simply defined as `ann.parameter(10.0)`, or even just `10.0`.

    :param value: Initial value of the parameter. It can be defined as a RandomDistribution, which will be sampled with the correct shape when the population is created.
    :param locality: Locality of the parameter. Must be in ['global', 'semiglobal', 'local'].
    :param type: Data type of the parameter. Must be in [float, int, bool] (or ['float', 'int', 'bool']).
    """
    value: float | int | bool | RandomDistribution
    locality: str = 'global'
    type: str = 'float'


# Variables
@dataclass
class variable:
    """
    Dataclass to represent a variable.
    """
    name: str
    equation: str
    init: float = 0.0
    min: float = None
    max: float = None
    method: str = None
    type: str = 'float'
    locality: str = 'local'