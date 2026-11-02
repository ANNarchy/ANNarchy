"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class SynapseRepresentation:
    """
    Stores information extracted from the parsed synapse type for
    use in code generation.

    Implementation note:

        No default values are provided, so all fields must be explicitly
        initialized when an instance is created. Currently, instances are
        created via the `from_dict` method.
    """
