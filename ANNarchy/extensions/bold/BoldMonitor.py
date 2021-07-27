#===============================================================================
#
#     BoldMonitor.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2021 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Oliver Maith <oli_maith@gmx.de>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core.Population import Population
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core import Global

from .PredefinedModels import balloon_RN
from .AccProjection import AccProjection

class BoldMonitor(object):
    """
    Create a Bold monitor to record from a pre-synaptic population.

    The monitor transforms, dependent on the applied model, one or two input signals into a recordable signal. The required accumulation of the input into one
    unified variable (output_variable, i. e. at the same time input to the model).

    :param populations: list of the recorded populations.
    :param scale_factor: Is a list of float values to allow a weighting of signals between populations. By default, the input signal is weighted by the ratio of the population size to all populations 
                         within the recorded region.
    :param normalize_input: Is a list of integer values which represent a optional baseline per population. In absence of NormProjections the input signals will require
                            an additional normalization using a baseline value. A value unequal to 0 represents the time period for determing this baseline in milliseconds biological time.
    :param input_variables: recorded variable either a neuron variable or the normalized conductance (result of a NormProjection)
    :param output_variables: intermediate sum of input which is then fed into the bold model
    :param bold_model: computational model for BOLD signal stored as BoldModel object (see ANNarchy.extensions.bold.PredefinedModels for more some predefined examples)
    :param recorded variables: which variables of the bold_model should be recorded? (default "BOLD")
    """
    def __init__(self, populations=[], scale_factor=[], normalize_input=[], input_variables="", output_variables="exc", bold_model=balloon_RN, recorded_variables=["BOLD"], start=False, net_id=0, copied=False):
        """
        Initialize several objects required to implement a BOLD recording.

        First we create:
        
         * a population with 1 neuron implementing the bold equations
         * a monitor which record from the single neuron

        For each record source (provided in *populations*) we then create:
        
         * a projection which contributes to the single neuron
        """
        self.net_id = net_id

        # for reporting
        bold_model._model_instantiated = True

        # argument check
        if not(isinstance(populations, list)):
            populations = [populations]
        if not(isinstance(scale_factor, list)):
            scale_factor = [scale_factor]*len(populations)
        if not(isinstance(normalize_input, list)):
            normalize_input = [normalize_input]*len(populations)
        if isinstance(recorded_variables, str):
            recorded_variables = [recorded_variables]

        if len(scale_factor) > 0:###TODO: this leads to errors, because scale_factor is somehow globally set
            if len(populations) != len(scale_factor):
                Global._error("Length of scale_factor must be equal to number of populations")

        if len(normalize_input) > 0:
            if len(populations) != len(normalize_input):
                Global._error("Length of normalize_input must be equal to number of populations")

        # The bold model relies on one input
        if isinstance(input_variables, str) and isinstance(output_variables, str):

            input_variables = [input_variables]
            output_variables = [output_variables]

        # The bold model relies on multiple inputs. For each input the user needs to define in->out
        elif isinstance(input_variables, list) and isinstance(output_variables, list):
            if len(input_variables) != len(output_variables):
                Global._error("BoldMonitor: the list of input_variables and output_variables must have the same length")

        else:
            Global._error("BoldMonitor: input_variables and output_variables must be either a string or a list of strings not mixed.")

        if not copied:
            # Add the container to the object management
            Global._network[0]['extensions'].append(self)
            self.id = len(Global._network[self.net_id]['extensions'])

            # create the population
            self._bold_pop = Population(1, neuron=bold_model, name= bold_model.name )
            self._bold_pop.enabled = start

            # create the monitor
            self._monitor = Monitor(self._bold_pop, recorded_variables, start=start)

            # create the projection(s)
            self._acc_proj = []

            if len(scale_factor) == 0:
                pop_overall_size = 0
                for idx, pop in enumerate(populations):
                    pop_overall_size += pop.size
                
                # the conductance is normalized between [0 .. 1]. This scale factor
                # should balance different population sizes
                for idx, pop in enumerate(populations):
                    scale_factor_conductance = float(pop.size)/float(pop_overall_size)
                    scale_factor.append(scale_factor_conductance)

            if len(normalize_input) == 0:
                normalize_input = [0] * len(populations)
                # TODO: can we check if users used NormProjections? If not, this will crash ...

            for input, output in zip(input_variables, output_variables):
                for pop, scale, normalize in zip(populations, scale_factor, normalize_input):
                    tmp_proj = AccProjection(pre = pop, post=self._bold_pop, target=output, variable=input, scale_factor=scale, normalize_input=normalize)
                    tmp_proj.connect_all_to_all(weights= 1.0)

                    self._acc_proj.append(tmp_proj)
        else:
            # Add the container to the object management
            Global._network[net_id]['extensions'].append(self)
            self.id = len(Global._network[self.net_id]['extensions'])

            # instances are assigned by the copying instance
            self._bold_pop = None
            self._monitor = None
            self._acc_proj = []

        self.name = "bold_monitor"

        # store arguments for copy 
        self._populations = populations
        self._input_variables = input_variables
        self._output_variables = output_variables
        self._recorded_variables = recorded_variables
        self._bold_model = bold_model
        self._start = start

        # Finalize initialization
        self._initialized = True if not copied else False

    #
    #   MONITOR functions
    #
    def start(self):
        """
        see also: ANNarchy.core.Monitor.start()
        """
        self._monitor.start()

        # enable ODEs
        self._bold_pop.cyInstance.activate(True)

        # check if we have projections with baseline
        for proj in self._acc_proj:
            if proj._normalize_input > 0:
                proj.cyInstance.start(proj._normalize_input/Global.config["dt"])

    def stop(self):
        """
        see also: ANNarchy.core.Monitor.stop()
        """
        self._monitor.stop()

        # enable ODEs
        self._bold_pop.cyInstance.activate(False)

    def get(self, variable):
        """
        see also: ANNarchy.core.Monitor.get()
        """
        return self._monitor.get(variable)


    #
    #   POPUATION functions i. e. access to model parameter
    #
    def __getattr__(self, name):
        """
        Method called when accessing an attribute.

        We overload the default to allow access to monitor variables.
        """
        if name == '_initialized' or not hasattr(self, '_initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Global._error("BoldMonitor: attributes can not modified before compile()")

            if name in self._bold_pop.attributes:
                return getattr(self._bold_pop, name)
        
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        """
        Method called when accessing an attribute.

        We overload the default to allow access to monitor variables.
        """
        if name == '_initialized' or not hasattr(self, '_initialized'): # Before the end of the constructor
            return object.__setattr__(self, name, value)

        if self._initialized:
            if self._bold_pop.initialized == False:
                Global._error("BoldMonitor: attributes can not modified before compile()")

            if name in self._bold_pop.attributes:
                setattr(self._bold_pop, name, value)
            else:
                raise AttributeError("the variable '"+str(name)+ "' is not an attribute of the bold model.")

        else:
            object.__setattr__(self, name, value)
