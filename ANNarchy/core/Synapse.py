"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import copy
import hashlib
import json

from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern import Messages
from ANNarchy.parser.AnalyseSynapse import analyse_synapse


class Synapse:
    """
    Base class to define a synapse model.

    Synapses expect `parameters` as a dictionary and `equations` as a list of variable updates (including `w` if there is synaptic plasticity).

    Rate-coded synapses can define `psp` and `operation` to modify synaptic transmission:

    ```python
    nonlinear_synapse = ann.Synapse(
        psp = "log( (pre.r * w + 1 ) / (pre.r * w - 1) )",
        operation = 'max',
    )
    ```

    Spiking synapses can define event-based rules, such as `pre_spike` (a pre-synaptic spike arrives at the synapse) and `post_spike` (the post-synaptic neuron emits a spike):

    ```python
    STDP = ann.Synapse(
        parameters = dict(
            tau_pre = 10.0,
            tau_post = 10.0,
            cApre = 0.01,
            cApost = 0.0105,
            wmax = 0.01,
        ),
        equations = [
            ann.Variable('tau_pre * dApre/dt = - Apre', method='event-driven'),
            ann.Variable('tau_post * dApost/dt = - Apost', method='event-driven'),
        ],
        pre_spike = '''
            g_target += w
            Apre += cApre * wmax
            w = clip(w - Apost, 0.0 , wmax)
        ''',
        post_spike = '''
            Apost += cApost * wmax
            w = clip(w + Apre, 0.0 , wmax)
        '''
    )
    ```

    :param parameters: dictionary of parameters and their initial value.
    :param equations: list of equations defining the temporal evolution of variables.
    :param psp: continuous influence of a single synapse on the post-synaptic neuron (default for rate-coded: ``w*pre.r``). Synaptic transmission in spiking synapses occurs in ``pre_spike``.
    :param operation: operation (sum, max, min, mean) performed by the post-synaptic neuron on the individual psp (rate-coded only).
    :param pre_spike: updating of variables when a pre-synaptic spike is received (spiking only).
    :param post_spike: updating of variables when a post-synaptic spike is emitted (spiking only).
    :param pre_axon_spike: updating of variables when an axonal spike was emitted (spiking only, default None). The usage of this arguments prevents the application of learning rules.
    :param functions: additional functions used in the equations.
    :param pruning: Condition for pruning the synapse.
    :param creating: Condition for creating the synapse.
    :param name: name of the synapse type (used for reporting only).
    :param description: short description of the synapse type (used for reporting).
    """

    # Default name and description for reporting
    _default_names = {
        "rate": "Rate-coded synapse",
        "spike": "Spiking synapse"
    }

    # A list of created synapse types, either pre- or user-defined
    _instantiated_types = set()
    _synapse_type_ids = {}

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        # compute a hash on arguments, i.e., the model definition
        hash_key = cls._compute_hash_id(args, kwargs)

        # store it on the instance for later usage, e.g., _rk_synapses_type
        instance._hash_key = hash_key

        # add the type to global list currently only used by report.
        if hash_key not in Synapse._instantiated_types:
            # first time instantiated
            Synapse._instantiated_types.add(hash_key)
            GlobalObjectManager().add_synapse_type(instance)
            Synapse._synapse_type_ids[hash_key] = GlobalObjectManager().num_synapse_types()

        return instance

    def __copy__(self):
        """
        Create a shallow copy of this object without re-running initialization.

        The copied object retains the same model hash as the original object.
        This ensures that copying an instantiated model does not create a new
        model definition.
        """
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """
        Create a deep copy of this object without re-running initialization.

        The copied object retains the same model hash as the original object.
        Nested attributes are recursively deep-copied using the provided `memo`
        dictionary to correctly handle shared references and circular references.
        """
        result = object.__new__(type(self))
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))

        return result

    def __init__(
        self,
        parameters: str | dict = "",
        equations: str | list = "",
        psp: str = None,
        operation: str = "sum",
        pre_spike: str | list = None,
        post_spike: str | list = None,
        pre_axon_spike: str = None,
        functions: str = None,
        pruning: str = None,
        creating: str = None,
        name: str = None,
        description: str = None,
        extra_values: dict = {},
    ):
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.pre_spike = pre_spike
        self.post_spike = post_spike
        self.psp = psp
        self.pre_axon_spike = pre_axon_spike
        self.operation = operation
        self.extra_values = extra_values
        self.pruning = pruning
        self.creating = creating

        # Type of the synapse TODO: smarter
        self.type = "spike" if pre_spike else "rate"

        # Check the operation
        if self.type == "spike" and self.operation != "sum":
            Messages.error(
                "Spiking synapses can only perform a sum of presynaptic potentials."
            )

        if self.operation not in ["sum", "min", "max", "mean"]:
            Messages.error(
                "The only operations permitted are: sum (default), min, max, mean."
            )

        # Sanity check
        if self.pre_axon_spike and self.post_spike:
            Messages.error(
                "The usage of axonal spike events is currently not allowed for plastic connections."
            )

        # Description
        self.description = None

        if name:
            self.name = name
        else:
            self.name = self._default_names[self.type]

        if description:
            self.short_description = description
        else:
            if self.type == "spike":
                self.short_description = "User-defined spiking synapse."
            else:
                self.short_description = "User-defined rate-coded synapse."

    @classmethod
    def _compute_hash_id(cls, args, kwargs, key_length=24):
        """
        Compute a hash value to later (re-)identify a neuron model definition.
        """
        from ANNarchy.generator.Utils import canonicalize

        data = {
            #"type": f"{cls.__name__}",
            "args": canonicalize(args),
            "kwargs": canonicalize(kwargs),
        }

        serialized = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return hashlib.sha256(serialized).hexdigest()[:key_length]

    @property
    def _rk_synapses_type(self):
        """ for reporting """
        return self._synapse_type_ids[self._hash_key]

    def _analyse(self, net_id):
        # Analyse the synapse type
        if not self.description:
            self.description = analyse_synapse(self, net_id)

    def __add__(self, synapse):
        Messages.error("adding synapse models is not implemented yet.")

        # self._variables.update(synapse.variables)

    def __repr__(self):
        if self.type == "rate":
            text = (
                """Rate-coded synapse.

Parameters:
"""
                + str(self.parameters)
                + """
Equations of the variables:
"""
                + str(self.equations)
                + """

"""
                + """
Synaptic transmission (psp):

"""
                + "\tw*pre.r"
                if self.psp == None
                else str(self.psp)
            )

        else:
            text = (
                """Spiking synapse.

Parameters:
"""
                + str(self.parameters)
                + """
Equations of the variables:
"""
                + str(self.equations)
                + """
pre-synaptic spike:
"""
                + str(self.pre_spike)
                + """
post-synaptic spike:
"""
                + str(self.post_spike)
            )

        return text

    def __str__(self):
        import pprint

        return pprint.pformat(self, depth=4)  # TODO
