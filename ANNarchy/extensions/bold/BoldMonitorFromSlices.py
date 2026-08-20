"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import inspect

from ANNarchy.core.Population import Population
from ANNarchy.core.Monitor import Monitor
from ANNarchy.cython_ext import LILConnectivity
from ANNarchy.extensions.bold.AccProjection import AccProjection
from ANNarchy.extensions.bold.BoldModel import BoldModel
from ANNarchy.extensions.bold.PredefinedModels import balloon_RN
from ANNarchy.intern import Messages
from ANNarchy.intern.ConfigManagement import ConfigManager, get_global_config
from ANNarchy.intern.NetworkManager import NetworkManager


class BoldMonitorFromSlices:
    """
    Monitors the BOLD signal from one population using a computational model. Contrary to the default `BoldMonitor`
    implementation, we here allow the usage of sub-groups within the population.

    The monitor can be started and stopped with `start()` and `stop()`. The recorded data is retrieved with `get()`.
    """

    def __init__(
        self,
        population: Population,
        slices : list[list[int]]|None = None,
        bold_model: BoldModel = None,
        mapping: dict = {"I_CBF": "r"},
        normalize_input: int = 0,
        recorded_variables: list[str]|None = None,
        start: bool = False,
        copied: bool = False,
        net_id: int = 0,
    ):
        self.net_id = net_id

        if bold_model is None:
            bold_model = balloon_RN

        # instantiate if necessary, please note that population will make a deepcopy on this objects
        if inspect.isclass(bold_model):
            bold_model = bold_model()

        # for reporting
        bold_model._model_instantiated = True

        if normalize_input is None:
            normalize_input = []
        if recorded_variables is None:
            recorded_variables = []

        # argument check
        if not (isinstance(population, Population)):
            Messages.error("A population must be provided as recording target. If you want to record from multiple populations please use Network.boldmonitor()")
        if (normalize_input is not None) and (not isinstance(normalize_input, (int, float))):
            Messages.error("The time window for normalization must be one value.")
        if isinstance(recorded_variables, str):
            recorded_variables = [recorded_variables]
        if slices is None:
            Messages.error("A list of ranks to record from needs to be provided.")

        # Check mapping
        for target, input_var in mapping.items():
            if not target in bold_model._inputs:
                Messages.error(
                    "BoldMonitor: the key "
                    + target
                    + " of mapping is not part of the BOLD model."
                )

        # Check recorded variables
        if len(recorded_variables) == 0:
            recorded_variables = bold_model._output
        else:
            # Add the output variables (and remove doublons)
            l1 = bold_model._output
            l2 = (
                [recorded_variables]
                if isinstance(recorded_variables, str)
                else recorded_variables
            )
            recorded_variables = list(set(l2 + l1))
            recorded_variables.sort()

        if not copied:
            # Add the container to the object management
            NetworkManager().add_extension(net_id=self.net_id, extension=self)
            self.id = NetworkManager().number_extensions(net_id=self.net_id	)

            # create the population
            self._bold_pop = Population(
                len(slices), neuron=bold_model, name=bold_model.name
            )
            self._bold_pop.enabled = start

            # create the monitor
            self._monitor = Monitor(
                self._bold_pop, recorded_variables, start=start
            )

            # create the projection(s)
            self._acc_proj = []

            for target, input_var in mapping.items():
                Messages._debug(
                    "Creating ACCProjection between", population.name, self._bold_pop.name
                )

                # Create the projection
                tmp_proj = AccProjection(
                    pre=population,
                    post=self._bold_pop,
                    target=target,
                    variable=input_var,
                    scale_factor=1.0,
                    normalize_input=normalize_input
                )

                # Instead of 1-to-all as in BoldMonitor, here we generate one neuron for each sub-list
                lil = LILConnectivity()
                for idx, slice in enumerate(slices):
                    # only add the connectivity - weights=1.0, delays=0.0 are ignored anyways
                    lil.add(idx, slice, [1.0], [0.0])

                # HD (19th August 2026): this is a bit hacky ... I'm not sure, why we don't have a
                #                        from_lil method in the user interface.
                tmp_proj.connector_name = "Load from LIL"
                tmp_proj.connector_description = "Load from LIL"
                tmp_proj._store_connectivity(
                    tmp_proj._load_from_lil, (lil,), 0.0, "lil", "post_to_pre"
                )
                tmp_proj._single_constant_weight = True

                self._acc_proj.append(tmp_proj)

        else:  # TODO check
            # instances are assigned by the copying instance
            self._bold_pop = None
            self._monitor = None
            self._acc_proj = []

        self.name = "bold_monitor"

        # store arguments for copy
        self._populations = population
        self._bold_model = bold_model
        self._mapping = mapping
        self._normalize_input = normalize_input
        self._recorded_variables = recorded_variables
        self._start = start

        # Finalize initialization
        self._initialized = True if not copied else False

    #
    #   MONITOR functions
    #
    def start(self):
        """
        Starts recording as in `ANNarchy.core.Monitor.start()`.
        """
        self._monitor.start()

        # enable ODEs
        self._bold_pop.cyInstance.activate(True)

        # check if we have projections with baseline
        for proj in self._acc_proj:
            if proj._normalize_input > 0:
                proj.cyInstance.start(
                    int(proj._normalize_input / get_global_config('dt'))
                )

    def stop(self):
        """
        Stops recording as in `ANNarchy.core.Monitor.stop()`.
        """
        self._monitor.stop()

        # enable ODEs
        self._bold_pop.cyInstance.activate(False)

    def get(self, variable):
        """
        Retrieves recordings as in `ANNarchy.core.Monitor.get()`.
        """
        return self._monitor.get(variable)

    #
    #   POPULATION functions i. e. access to model parameter
    #

    # Method called when accessing an attribute.
    # We overload the default to allow access to monitor variables.
    def __getattr__(self, name):
        if name == "_initialized" or not hasattr(
            self, "_initialized"
        ):  # Before the end of the constructor
            return object.__getattribute__(self, name)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Messages.error(
                    "BoldMonitor: attributes can not modified before compile()"
                )

            if name in self._bold_pop.attributes:
                return getattr(self._bold_pop, name)

        return object.__getattribute__(self, name)

    # Method called when accessing an attribute.
    # We overload the default to allow access to monitor variables.
    def __setattr__(self, name, value):
        if name == "_initialized" or not hasattr(
            self, "_initialized"
        ):  # Before the end of the constructor
            return object.__setattr__(self, name, value)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Messages.error(
                    "BoldMonitor: attributes can not modified before compile()"
                )

            if name in self._bold_pop.attributes:
                setattr(self._bold_pop, name, value)
            else:
                raise AttributeError(
                    "the variable '"
                    + str(name)
                    + "' is not an attribute of the bold model."
                )

        else:
            object.__setattr__(self, name, value)

    #
    # Destruction
    def _clear(self):
        pass
