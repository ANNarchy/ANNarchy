#===============================================================================
#
#     BoldMonitor2.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Oliver Maith <>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

from .BoldModels import BoldNeuron
from .AccProjection import AccProjection

class BoldMonitor(object):
    """
    Create a Bold monitor to record from a pre-synaptic population.
    """
    def __init__(self, populations=[], input_variables="", output_variables="", recorded_variables=[], bold_model=BoldNeuron, start=False, net_id=0, copied=False):
        """
        Initialize several objects required to implement a BOLD recording.

        First we create:
        
         * a population with 1 neuron implementing the bold equations
         * a monitor which record from the single neuron

        For each record source (provided in *populations*) we then create:
        
         * a projection which contributes to the single neuron
        """
        self.net_id = net_id

        # argument check
        if len(populations) == 1:
            populations = [populations]
        if isinstance(recorded_variables, str):
            recorded_variables = [recorded_variables]

        # The bold model relies on one input
        if isinstance(input_variables, str) and isinstance(output_variables, str):

            input_variables = [input_variables]
            output_variables = [output_variables]

        # The bold model relies on multiple inputs. For each input the user needs to define in->out
        elif isinstance(input_variables, list) and isinstance(output_variables, list):
            if len(input_variables) != len(output_variables):
                _error("BoldMonitor: the list of input_variables and output_variables must have the same length")

        else:
            _error("BoldMonitor: input_variables and output_variables must be either a string or a list of strings not mixed.")

        if not copied:
            # Add the container to the object management
            Global._network[0]['extensions'].append(self)
            self.id = len(Global._network[self.net_id]['extensions'])

            # create the population
            self._bold_pop = Population(1, neuron=bold_model)

            # create the monitor
            self._monitor = Monitor(self._bold_pop, recorded_variables, start=start)

            # create the projection(s)
            self._acc_proj = []

            for input, output in zip(input_variables, output_variables):
                for pop in populations:

                    tmp_proj = AccProjection(pre = pop, post=self._bold_pop, target=output, variable=input)
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

    def stop(self):
        """
        see also: ANNarchy.core.Monitor.stop()
        """
        self._monitor.stop()

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
