"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class AttributeRepresentation:
    """
    Stores common information shared between the different attribute types
    in ANNarchy, i.e., parameter, variables, and random distributions.
    """
    name: str
    ctype: str
    locality: Literal["local", "semiglobal", "global"]

    def get_type(self) -> Literal["parameter", "variable"]:
        """ Returns a short type declaration of the attribute. """
        if isinstance(self, ParameterRepresentation):
            return "parameter"
        if isinstance(self, VariableRepresentation):
            return "variable"

        raise ValueError(
            f"Unknown attribute type: {type(self).__name__}"
        )

@dataclass(slots=True)
class ParameterRepresentation(AttributeRepresentation):
    """
    Stores the information related to an ANNarchy parameter, i.e., an
    attribute whose value is independent from the neuron equations.
    """
    bounds: dict
    flags: list[str]
    init: int|float|str

    @property
    def eq(self) -> str:
        return f"{self.name} = {self.init}"

@dataclass(slots=True)
class VariableRepresentation(AttributeRepresentation):
    bounds: dict
    cpp: str
    dependencies: list[str]
    eq: str
    flags: list[str]
    init: int|float|str
    method: str
    num_flops: int
    pre_loop: dict
    switch: str|None
    transformed_eq: str
    untouched: list[dict[str,str]]

@dataclass(slots=True)
class RandomDistRepresentation(AttributeRepresentation):
    """
    Stores information and code snippets related to values drawn from random distributions.
    """
    args: str
    dependencies: list[str]
    dist: str
    template: str

    @property
    def definition(self) -> str:
        """ Creates initialization statement """
        return f"{self.template}({self.args})"

@dataclass(slots=True)
class FunctionRepresentation:
    """
    Stores information and code snippets related to functions. Note that, these functions
    can not access the neural/synaptic states, therefore, everything need to be provided
    as arguments.
    """
    cpp: str
    name: str
