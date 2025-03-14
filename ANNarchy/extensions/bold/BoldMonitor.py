"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Population import Population
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core import Global
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

from .BoldModel import BoldModel
from .PredefinedModels import balloon_RN
from .AccProjection import AccProjection

import inspect

class BoldMonitor(object):
    """
    Monitors the BOLD signal for several populations using a computational model.

    Returned by `Network.boldmonitor()`.

    The monitor can be started and stopped with `start()` and `stop()`. The recorded data is retrieved with `get()`.
    """
    def __init__(self,
            populations: list=None,
            bold_model: BoldModel = None,
            mapping: dict={'I_CBF': 'r'},
            scale_factor: list[float]=None,
            normalize_input: list[int]=None,
            recorded_variables: list[str]=None,
            start:bool=False,
            copied:bool=False,
            net_id:int=0,
        ):
        
        self.net_id = net_id

        if bold_model is None:
            bold_model = bold.balloon_RN

        # instantiate if necessary, please note that population will make a deepcopy on this objects
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


        # Get the corresponding network
        self._net = NetworkManager().get_network(net_id=net_id)

        # Add the container to the object management
        self.id = self._net._add_extension(extension=self)


        if not copied:

            # create the population
            self._bold_pop = self._net.create(1, neuron=bold_model, name=bold_model.name )
            self._bold_pop.enabled = start

            # create the monitor
            self._monitor = self._net.monitor(self._bold_pop, recorded_variables, start=start)

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
                    Messages._debug("Creating ACCProjection between", pop.name, self._bold_pop.name)
                    tmp_proj = AccProjection(
                        pre = pop, 
                        post=self._bold_pop, 
                        target=target, 
                        variable=input_var, 
                        scale_factor=scale, 
                        normalize_input=normalize,
                        net_id = net_id
                    )
                    tmp_proj.all_to_all(weights= 1.0)

                    self._acc_proj.append(tmp_proj)

        else: # TODO check

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
                proj.cyInstance.start(int(proj._normalize_input/ConfigManager().get('dt', self.net_id)))

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
