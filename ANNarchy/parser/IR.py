"""
A collection of intermediate representations (IR) produced by the parser module.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

from ANNarchy.core.Attributes import Parameter, Variable

@dataclass(frozen=True)
class RawNeuronFields:
    """
    Stores the original arguments passed to `ANNarchy.core.Neuron`,
    primarily for debugging purposes.
    """
    parameters: dict
    equations: str
    functions: str
    reset: str | None
    spike: str | None

@dataclass(frozen=True)
class SpikeProcessing:
    """
    Stores information and code snippets related to spike events.
    """
    condition: str
    condition_dependencies: list[str]
    reset: list[dict[str, Any]]

@dataclass(frozen=True)
class NeuronRepresentation:
    """
    Stores information extracted from the parsed neuron type for
    use in code generation.

    Implementation note:

        No default values are provided, so all fields must be explicitly
        initialized when an instance is created. Currently, instances are
        created via the `from_dict` method.
    """

    _type: Literal["rate", "spike"]
    _parameters: list[dict[str, str|Parameter]]
    _variables: list[dict[str, str|Variable]]
    _random_distributions: list[dict]
    _targets: list[str]
    _functions: list[dict[str, str]]
    _global_operations: list[str]
    _spike: SpikeProcessing | None
    _axon_spike: SpikeProcessing | None
    _raw: RawNeuronFields

#########################################################################################
# Create iterators for traversing the data fields.                                      #
#########################################################################################

    @property
    def attributes(self) -> Iterator[dict[str, Any]]:
        """
        Return an iterator over all attributes.

        Implementation note:

            Unlike list concatenation with `+`, this does not create a new list.
            The original attribute dictionaries are yielded directly.
        """
        return chain(
            self._parameters,
            self._variables,
            self._random_distributions,
        )

    def filter_by_locality(self, locality: Literal['local','global']) -> Iterator[dict[str, Any]]:
        """
        Return an iterator over all attributes with the provided `locality`.

        Implementation note:

            Unlike list concatenation with `+`, this does not create a new list.
            The original attribute dictionaries are yielded directly.
        """
        return (
            attr
            for attr in self.attributes
            if attr["locality"] == locality
        )

    @property
    def global_attributes(self) -> Iterator[dict[str, Any]]:
        """
        Return an iterator over all attributes with locality 'global'.
        """
        return self.filter_by_locality("global")

    @property
    def local_attributes(self) -> Iterator[dict[str, Any]]:
        """
        Return an iterator over all attributes with locality 'local'.
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
    def functions(self) -> list[dict[str, str]]:
        return self._functions

    @property
    def targets(self) -> list[str]:
        return self._targets

    @property
    def random_distributions(self) -> list[dict[str, dict]]:
        return self._random_distributions

    @property
    def variables(self) -> list[dict[str, str | Variable]]:
        return self._variables

#########################################################################################
# Functions for gathering property names from the data fields.                          #
#########################################################################################

    @property
    def attr_names(self) -> list[str]:
        """ Returns a list of names of all attributes. """
        return [attr["name"] for attr in self.attributes]

    @property
    def global_attr_names(self) -> list[str]:
        """ Returns a list of global attributes. """
        return [attr["name"] for attr in self.global_attributes]

    @property
    def local_attr_names(self) -> list[str]:
        """ Returns a list of local attributes. """
        return [attr["name"] for attr in self.local_attributes]

    @property
    def variable_names(self) -> list[str]:
        """ Return a list of names of the declared variables. """
        return [ tmp["name"] for tmp in self.variables ]

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

    @classmethod
    def from_dict(cls, description) -> "NeuronRepresentation":
        """
        Temporary methods -> will be removed later

        Creates the new intermediate representation from the description object currently
        generated by the parser module.
        """
        if description['type']=="spike":
            raw_reset = description['raw_reset']
            raw_spike = description['raw_spike']

            spike_desc = SpikeProcessing(
                condition = description['spike']['spike_cond'],
                condition_dependencies = description['spike']['spike_cond_dependencies'],
                reset = description['spike']['spike_reset']
            )

            if description['axon_spike'] is not None:
                axon_spike_desc = SpikeProcessing(
                    condition = description['axon_spike']['spike_cond'],
                    condition_dependencies = description['axon_spike']['spike_cond_dependencies'],
                    reset = description['axon_spike']['spike_reset']
                )
            else:
                axon_spike_desc = None
        else:
            spike_desc = None
            axon_spike_desc = None
            raw_reset = None
            raw_spike = None

        return cls(
            _type = description['type'],
            _parameters = description['parameters'],
            _variables = description['variables'],
            _random_distributions = description['random_distributions'],
            _targets = description['targets'],
            _functions = description['functions'],
            _global_operations = description['global_operations'],
            _spike = spike_desc,
            _axon_spike = axon_spike_desc,
            _raw = RawNeuronFields(
                parameters = description['raw_parameters'],
                equations = description['raw_equations'],
                functions = description['raw_functions'],
                reset = raw_reset,
                spike = raw_spike
            )
        )