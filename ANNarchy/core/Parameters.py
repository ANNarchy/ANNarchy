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
            # Global parameter
            tau = ann.parameter(value=10.0, locality='global')

            # Local parameter
            baseline = ann.parameter(value=ann.Uniform(-1., 1.), locality='local')

            # Boolean global parameter
            activated = ann.parameter(value=True, type=bool)
        )
    )
    ```

    By default, parameters are global and use the float type, so tau could be simply defined as `ann.parameter(10.0)`, or even just `10.0`.

    :param value: Initial value of the parameter. It can be defined as a RandomDistribution, which will be sampled with the correct shape when the population/projection is created, or a float/int/bool, depending on `type`.
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
    Dataclass to represent a variable in a Neuron or Synapse definition.

    ```python
    neuron = ann.Neuron(
        equations = [
            # Global parameter
            tau = ann.parameter(value=10.0, locality='global')

            # Local parameter
            baseline = ann.parameter(value=ann.Uniform(-1., 1.), locality='local')

            # Boolean global parameter
            activated = ann.parameter(value=True, type=bool)
        ]
    )
    ```

    :param equation: string representing the equation.
    :param init: initial value of the variable. It can be defined as a RandomDistribution, which will be sampled with the correct shape when the population/projection is created, or a float/int/bool, depending on `type`.
    :param min: minimum value that the variable can take. 
    :param max: maximum value that the variable can take. 
    :param method: numerical method to be used when the equation is an ODE. Must be in ['explicit', 'implicit', 'semiimplicit', 'exponential','midpoint', 'rk4', 'event-driven']
    :param locality: Locality of the parameter. Must be in ['global', 'semiglobal', 'local'].
    :param type: Data type of the parameter. Must be in [float, int, bool] (or ['float', 'int', 'bool']).
    """
    equation: str
    init: float | int | bool | RandomDistribution = None
    min: float = None
    max: float = None
    method: str = None
    locality: str = 'local'
    type: str = 'float'

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
    

# Structural plasticity
@dataclass
class creating:
    """
    Dataclass to represent a creation condition for structural plasticity.

    When the condition is true, a synapse is created with the specified probability, using the weight $w$ and delay $d$.

    ```python
    StructuralPlasticSynapse = ann.Synapse(

        parameters = dict(eta = 0.1, T = 1.0),
    
        equations = ann.variable("dw/dt = eta * pre.r * post.r"),
        
        creating = ann.creating("pre.r * post.r > T", proba = 0.1, w = 0.01),
    )
    ```

    """
    equation: str
    proba: float = 1.0
    w: float = 0.0
    d: float = None 

@dataclass
class pruning:
    """
    Dataclass to represent a pruning condition for structural plasticity.

    """
    equation: str
    proba: float = 1.0