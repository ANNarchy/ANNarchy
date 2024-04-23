"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Population import Population
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core import Global
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages

from .PredefinedModels import balloon_RN
from .AccProjection import AccProjection

import inspect

class BoldMonitor(object):
    """
    Monitors the BOLD signal for several populations using a computational model.

    The BOLD monitor transforms one or two input population variables (such as the mean firing rate) into a recordable BOLD signal according to a computational model (for example a variation of the Balloon model).


    :param populations: list of recorded populations.

    :param bold_model: computational model for BOLD signal defined as a BoldModel object. Default is `balloon_RN`.

    :param mapping: mapping dictionary between the inputs of the BOLD model (`I_CBF` for single inputs, `I_CBF` and `I_CMRO2` for double inputs in the provided examples) and the variables of the input populations. By default, `{'I_CBF': 'r'}` maps the firing rate `r` of the input population(s) to the variable `I_CBF` of the BOLD model. 

    :param scale_factor: list of float values to allow a weighting of signals between populations. By default, the input signal is weighted by the ratio of the population size to all populations within the recorded region.

    :param normalize_input: list of integer values which represent a optional baseline per population. The input signals will require an additional normalization using a baseline value. A value different from 0 represents the time period for determing this baseline in milliseconds (biological time).

    :param recorded_variables: which variables of the BOLD model should be recorded? (by default, the output variable of the BOLD model is added, e.g. ["BOLD"] for the provided examples).

    :param start: whether to start recording directly.
    """
    def __init__(self,
        populations: list=None,
        bold_model: 'BoldModel' =balloon_RN,
        mapping: dict={'I_CBF': 'r'},
        scale_factor: list[float]=None,
        normalize_input: list[int]=None,
        recorded_variables: list[str]=None,
        start:bool=False,
        net_id:int=0,
        copied:bool=False):
        
        self.net_id = net_id

        # instantiate if necessary, please note
        # that population will make a deepcopy on this objects
        if inspect.isclass(bold_model):
            bold_model = bold_model()

        # for reporting
        bold_model._model_instantiated = True

        # The usage of [] as default arguments in the __init__ call lead to strange side effects.
        # We decided therefore to use None as default and create the lists locally.
        if populations is None:
            Messages._error("Either a population or a list of populations must be provided to the BOLD monitor (populations=...)")
        if scale_factor is None:
            scale_factor = []
        if normalize_input is None:
            normalize_input = []
        if recorded_variables is None:
            recorded_variables = []

        # argument check
        if not(isinstance(populations, list)):
            populations = [populations]
        if not(isinstance(scale_factor, list)):
            scale_factor = [scale_factor]*len(populations)
        if not(isinstance(normalize_input, list)):
            normalize_input = [normalize_input]*len(populations)
        if isinstance(recorded_variables, str):
            recorded_variables = [recorded_variables]

        if len(scale_factor) > 0:
            if len(populations) != len(scale_factor):
                Messages._error("BoldMonitor: Length of scale_factor must be equal to number of populations")

        if len(normalize_input) > 0:
            if len(populations) != len(normalize_input):
                Messages._error("BoldMonitor: Length of normalize_input must be equal to number of populations")

        # Check mapping
        for target, input_var in mapping.items():
            if not target in bold_model._inputs:
                Messages._error("BoldMonitor: the key " + target + " of mapping is not part of the BOLD model.")

        # Check recorded variables
        if len(recorded_variables) == 0:
            recorded_variables = bold_model._output
        else:
            # Add the output variables (and remove doublons)
            l1 = bold_model._output
            l2 = [recorded_variables] if isinstance(recorded_variables, str) else recorded_variables
            recorded_variables = list(set(l2+l1))
            recorded_variables.sort()

        if not copied:
            # Add the container to the object management
            NetworkManager().add_extension(net_id=0, extension=self)
            self.id = NetworkManager().number_extensions(net_id=0)

            # create the population
            self._bold_pop = Population(1, neuron=bold_model, name= bold_model.name )
            self._bold_pop.enabled = start

            # create the monitor
            self._monitor = Monitor(self._bold_pop, recorded_variables, start=start)

            # create the projection(s)
            self._acc_proj = []

            if len(scale_factor) == 0:
                pop_overall_size = 0
                for _, pop in enumerate(populations):
                    pop_overall_size += pop.size

                # the conductance is normalized between [0 .. 1]. This scale factor
                # should balance different population sizes
                for _, pop in enumerate(populations):
                    scale_factor_conductance = float(pop.size)/float(pop_overall_size)
                    scale_factor.append(scale_factor_conductance)

            if len(normalize_input) == 0:
                normalize_input = [0] * len(populations)
                # TODO: can we check if users used NormProjections? If not, this will crash ...

            for target, input_var in mapping.items():
                for pop, scale, normalize in zip(populations, scale_factor, normalize_input):

                    tmp_proj = AccProjection(pre = pop, post=self._bold_pop, target=target, variable=input_var, scale_factor=scale, normalize_input=normalize)

                    tmp_proj.connect_all_to_all(weights= 1.0)

                    self._acc_proj.append(tmp_proj)

        else:
            # Add the container to the object management
            NetworkManager().add_extension(net_id=self.net_id, extension=self)
            self.id = NetworkManager().number_extensions(net_id=self.net_id)

            # instances are assigned by the copying instance
            self._bold_pop = None
            self._monitor = None
            self._acc_proj = []

        self.name = "bold_monitor"

        # store arguments for copy
        self._populations = populations
        self._bold_model = bold_model
        self._mapping = mapping
        self._scale_factor = scale_factor
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
                proj.cyInstance.start(proj._normalize_input/get_global_config('dt'))

    def stop(self):
        """
        Stops recording as in `ANNarchy.core.Monitor.stop().
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

        if name == '_initialized' or not hasattr(self, '_initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Messages._error("BoldMonitor: attributes can not modified before compile()")

            if name in self._bold_pop.attributes:
                return getattr(self._bold_pop, name)

        return object.__getattribute__(self, name)

    # Method called when accessing an attribute.
    # We overload the default to allow access to monitor variables.
    def __setattr__(self, name, value):

        if name == '_initialized' or not hasattr(self, '_initialized'): # Before the end of the constructor
            return object.__setattr__(self, name, value)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Messages._error("BoldMonitor: attributes can not modified before compile()")

            if name in self._bold_pop.attributes:
                setattr(self._bold_pop, name, value)
            else:
                raise AttributeError("the variable '"+str(name)+ "' is not an attribute of the bold model.")

        else:
            object.__setattr__(self, name, value)

    #
    # Destruction
    def _clear(self):
        pass
