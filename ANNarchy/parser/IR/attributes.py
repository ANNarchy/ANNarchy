"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class AttributeRepresentation:
    """
    Stores common information shared between the different attribute types
    in ANNarchy, i.e., parameter, variables, and random distributions.
    """
    name: str
    ctype: str
    locality: Literal["local", "global"]

@dataclass(frozen=True)
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

    @classmethod
    def from_dict(cls, param_dict: dict) -> "ParameterRepresentation":
        """ Temporary function: converts old-style dict into intermediate representation object """
        return cls(
            # common
            name=param_dict["name"],
            ctype=param_dict["ctype"],
            locality=param_dict["locality"],
            # parameter-specific
            bounds=param_dict["bounds"],
            flags=param_dict["flags"],
            init=param_dict["init"]
        )

@dataclass(frozen=True)
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

    @classmethod
    def from_dict(cls, variable_dict: dict) -> "VariableRepresentation":
        """ Temporary function: converts old-style dict into intermediate representation object """
        return cls(
            # common
            name=variable_dict["name"],
            ctype=variable_dict["ctype"],
            locality=variable_dict["locality"],
            # variable-specific
            bounds=variable_dict["bounds"],
            cpp=variable_dict["cpp"],
            dependencies=variable_dict.get("dependencies"),
            eq=variable_dict["eq"],
            flags=variable_dict["flags"],
            init=variable_dict["init"],
            method=variable_dict.get("method"),
            num_flops=variable_dict.get("num_flops"),
            pre_loop=variable_dict.get("pre_loop"),
            switch=variable_dict.get("switch"),
            transformed_eq=variable_dict.get("transformed_eq"),
            untouched=variable_dict.get("untouched")
        )

@dataclass(frozen=True)
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

    @classmethod
    def from_dict(cls, random_dict: dict) -> "RandomDistRepresentation":
        """ Temporary function: converts old-style dict into intermediate representation object """
        return cls(
            args = random_dict["args"],
            ctype = random_dict["ctype"],
            dependencies = random_dict["dependencies"],
            dist = random_dict["dist"],
            locality = random_dict["locality"],
            name = random_dict["name"],
            template = random_dict["template"]
        )
