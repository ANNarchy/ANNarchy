"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Literal

from ANNarchy.core.Random import RandomDistribution

from .attributes import (
    AttributeRepresentation,
    FunctionRepresentation,
    ParameterRepresentation,
    RandomDistRepresentation,
    VariableRepresentation,
)
from .raw import RawNeuronFields
from .spike import SpikeProcessing


@dataclass(slots=True)
class NeuronRepresentation:
    """
    Stores information extracted from the parsed neuron type for
    use in code generation.

    Implementation note:

        No default values are provided, so all fields must be explicitly
        initialized when an instance is created. Currently, instances are
        created via the IRConverter.
    """

    _type: Literal["rate", "spike"]
    _parameters: list[ParameterRepresentation]
    _variables: list[VariableRepresentation]
    _random_distributions: list[RandomDistRepresentation]
    _targets: list[str]
    _functions: list[FunctionRepresentation]
    _global_operations: list[str]
    _spike: SpikeProcessing | None
    _axon_spike: SpikeProcessing | None
    _raw: RawNeuronFields
    _refractory: str|int|float|RandomDistribution|None

#########################################################################################
# Create iterators for traversing the data fields.                                      #
#########################################################################################

    @property
    def attributes(self) -> Iterator[AttributeRepresentation]:
        """
        Iterate over parameters followed by variables.

        Random distributions are intentionally excluded; access them through
        `random_distributions`.

        Implementation note:

            Unlike list concatenation with `+`, this does not create a new list.
            The stored attribute representations are yielded directly.
        """
        return chain(
            self._parameters,
            self._variables
        )

    def filter_by_locality(self, locality: Literal['local','global']) -> Iterator[AttributeRepresentation]:
        """
        Return an iterator over all variables and parameters with the provided `locality`.

        Implementation note:

            Unlike list concatenation with `+`, this does not create a new list.
            The stored attribute representations are yielded directly.
        """
        return (
            attr
            for attr in self.attributes
            if attr.locality == locality
        )

    @property
    def global_attributes(self) -> Iterator[AttributeRepresentation]:
        """
        Return an iterator over all variables and parameters with locality 'global'.
        """
        return self.filter_by_locality("global")

    @property
    def local_attributes(self) -> Iterator[AttributeRepresentation]:
        """
        Return an iterator over all variables and parameters with locality 'local'.
        """
        return self.filter_by_locality("local")

#########################################################################################
# Functions for gathering specific type of informations from the data fields.           #
#########################################################################################

    @property
    def is_spike(self) -> bool:
        return self._type == "spike"

    @property
    def is_rate(self) -> bool:
        return self._type == "rate"

    @property
    def functions(self) -> list[FunctionRepresentation]:
        return self._functions

    def has_functions(self):
        """ Returns `True` if the neuron definition includes at least one locally defined function. """
        return len(self._functions) > 0

    @property
    def targets(self) -> list[str]:
        return self._targets

    @property
    def random_distributions(self) -> list[RandomDistRepresentation]:
        return self._random_distributions

    def has_random_distributions(self):
        """ Returns `True` if the neuron definition includes at least one random distribution attribute. """
        return len(self._random_distributions) > 0

    @property
    def refractory(self) -> str|int|float|RandomDistribution:
        """ Returns the assigned refractory time """
        return self._refractory

    @refractory.setter
    def refractory(self, value):
        self._refractory = value

    @property
    def parameters(self) -> list[ParameterRepresentation]:
        return self._parameters

    def parameters_by_locality(
        self,
        locality: Literal["local", "global"],
    ) -> Iterator[ParameterRepresentation]:
        """ Iterate over parameters with the requested locality. """
        return (param for param in self.parameters if param.locality == locality)

    def has_parameters(self):
        """ Returns `True` if the neuron definition includes at least one parameter. """
        return len(self._parameters) > 0

    @property
    def variables(self) -> list[VariableRepresentation]:
        return self._variables

    def has_variables(self):
        """ Returns `True` if the neuron definition includes at least one variable. """
        return len(self._variables) > 0

#########################################################################################
# Functions for gathering property names from the data fields.                          #
#########################################################################################

    @property
    def attr_names(self) -> list[str]:
        """ Returns a list of names of all parameters and variables. """
        return [attr.name for attr in self.attributes]

    @property
    def global_attr_names(self) -> list[str]:
        """ Returns a list of global parameters and variables. """
        return [attr.name for attr in self.global_attributes]

    @property
    def local_attr_names(self) -> list[str]:
        """ Returns a list of local parameters and variables. """
        return [attr.name for attr in self.local_attributes]

    @property
    def parameter_names(self) -> list[str]:
        """ Return a list of names of the declared parameters. """
        return [ tmp.name for tmp in self.parameters ]

    def parameter_names_by_locality(
        self,
        locality: Literal["local", "global"],
    ) -> list[str]:
        """ Return a list of names of parameters with the requested locality. """
        return [parameter.name for parameter in self.parameters_by_locality(locality)]

    @property
    def variable_names(self) -> list[str]:
        """ Return a list of names of the declared variables. """
        return [ tmp.name for tmp in self.variables ]

    @property
    def function_names(self) -> list[str]:
        """ Return a list of names of the declared functions. """
        return [ tmp.name for tmp in self.functions ]

    def spike_processing(self, spike_type: Literal["neuron", "axon"] = "neuron") -> SpikeProcessing|None:
        """
        Get the code fields associated with a spike event.

        Currently, we distinguish between two cases:
        - "neuron": spikes generated within the neuron itself.
        - "axon": events attributed to external currents acting on the axon rather than directly on the neuron.
        """
        if spike_type == "neuron":
            return self._spike
        elif spike_type == "axon":
            return self._axon_spike
        else:
            raise ValueError(f"Invalid spike type: {spike_type!r}")