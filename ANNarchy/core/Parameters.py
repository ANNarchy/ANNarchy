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
    equation: str
    init: float = None
    min: float = None
    max: float = None
    method: str = None
    type: str = 'float'
    locality: str = 'local'

    def _to_string(self, object_type:str) -> str:
        "Returns a one-liner string with the flags. object_type is either 'neuron' or 'synapse'. to decide between population and projection"

        representation = self.equation

        # Any flag?
        has_flag = False
        flags = []

        if self.init is not None:
            if not has_flag: has_flag = True ; representation += " : "
            if isinstance(self.init, (RandomDistribution,)):
                init = repr(self.init)
            else: 
                init = self.init
            flags.append(f"init={init}") 

        if self.min is not None:
            if not has_flag: has_flag = True ; representation += " : "
            flags.append(f"min={self.min}") 

        if self.max is not None:
            if not has_flag: has_flag = True ; representation += " : "
            flags.append(f"max={self.max}") 

        if self.method is not None:
            if not has_flag: has_flag = True ; representation += " : "
            flags.append(f"{self.method}") 

        if self.type != 'float':
            if not has_flag: has_flag = True ; representation += " : "
            flags.append(f"{self.type}") 

        if self.locality != 'local':
            if not has_flag: has_flag = True ; representation += " : "
            if object_type == 'neuron':
                val = 'population' # 'global' and 'population' would work
            elif object_type == 'synapse':
                if self.locality in ['global', 'projection']: # both are OK
                    val = 'projection'
                else: val = 'postsynaptic'
            flags.append(f"{val}") 


        return representation + ', '.join(flags)