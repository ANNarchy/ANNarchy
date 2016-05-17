"""

    Neuron.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from ANNarchy.core.Global import _error, _warning, _network, _objects
from ANNarchy.parser.SingleAnalysis import analyse_neuron
from ANNarchy.core.PopulationView import PopulationView
import numpy as np

class Neuron(object):
    """
    Base class to define a neuron.
    """
    def __init__(self, parameters="", equations="", spike=None, reset=None, refractory = None, functions=None, name="", description="", extra_values={} ):
        """
        *Parameters*:

            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.
            * **spike**: condition to emit a spike (only for spiking neurons).
            * **reset**: changes to the variables after a spike (only for spiking neurons).
            * **refractory**: refractory period of a neuron after a spike (only for spiking neurons).
            * **name**: name of the neuron type (used for reporting only).
            * **description**: short description of the neuron type (used for reporting).

        """

        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.spike = spike
        self.reset = reset
        self.refractory = refractory
        self.extra_values = extra_values

        # Find the type of the neuron
        self.type = 'spike' if self.spike else 'rate'

        # Reporting
        if not hasattr(self, '_instantiated') : # User-defined
            _objects['neurons'].append(self)
        elif len(self._instantiated) == 0: # First instantiated of the class
            _objects['neurons'].append(self)

        if name:
            self.name = name
        else:
            self.name = 'Spiking neuron' if self.type == 'spike' else 'Rate-coded neuron'
        if description:
            self.short_description = description
        else:
            self.short_description = "Specific model of a spiking point-neuron." if self.type == 'spike' else "Specific model of a rate-coded neuron."

        # Analyse the neuron type
        self.description = None

    def _analyse(self):
        # Analyse the neuron type
        if not self.description:
            self.description = analyse_neuron(self)

    def __repr__(self):
        if self.type == 'rate':
            text= """Rate-coded neuron.

Parameters:
""" + str(self.parameters) + """
Equations of the variables:
""" + str(self.equations) + """

"""
        else:
            text= """Spiking neuron.

Parameters:
""" + str(self.parameters) + """
Equations of the variables:
""" + str(self.equations) + """
Spiking condition:
""" + str(self.spike) + """
Reset after a spike:
""" + str(self.reset)

        return text



class RateNeuron(Neuron):
    """
    Base class to define a rate-coded neuron.
    """
    def __init__(self, parameters="", equations="", functions=None, name="", description="", extra_values={}):
        """
        *Parameters*:

            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.

        """
        _warning("The use of RateNeuron or SpikeNeuron is deprecated, use Neuron instead.")
        Neuron.__init__(self, parameters=parameters, equations=equations, functions=functions, name=name, description=description, extra_values=extra_values)

class SpikeNeuron(Neuron):
    """
    Base class to define a spiking neuron.
    """
    def __init__(self, parameters="", equations="", spike=None, reset=None, refractory = None, functions=None, name="", description="", extra_values={} ):
        """
        *Parameters*:

            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.
            * **spike**: condition to emit a spike.
            * **reset**: changes to the variables after a spike
            * **refractory**: refractory period of a neuron after a spike.

        """
        _warning("The use of RateNeuron or SpikeNeuron is deprecated, use Neuron instead.")
        Neuron.__init__(self, parameters=parameters, equations=equations, functions=functions, spike=spike, reset=reset, refractory=refractory, name=name, description=description, extra_values=extra_values)

class IndividualNeuron(object):
    """
    Neuron object returned by the Population.neuron(rank) method.

    This only a wrapper around the Population data. It has the same attributes (parameter and variable) as the original population.
    """
    def __init__(self, population, rank):
        self.population  = population
        self.rank  = rank

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'population':
            return object.__getattribute__(self, name)
        elif hasattr(self.population, 'attributes'):
            if name in self.population.attributes:
                if not self.population.initialized: # Store it in the temporary array
                    val = (self.population.get(name))
                    if isinstance(val, np.ndarray):
                        return val[self.population.coordinates_from_rank(self.rank)]
                    else:
                        return val
                else:
                    if name in self.population.neuron_type.description['local']:
                        return getattr(self.population.cyInstance, 'get_single_'+name)(self.rank)
                    else:
                        return getattr(self.population.cyInstance, 'get_'+name)()
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name == 'population':
            object.__setattr__(self, name, value)
        elif name == 'rank':
            object.__setattr__(self, name, value)
        elif hasattr(self.population, 'attributes'):
            if name in self.population.attributes:
                if name in self.population.neuron_type.description['local']:
                    if not self.population.initialized: # Store it in the temporary array
                        newval = self.population.get(name)
                        newval[self.population.coordinates_from_rank(self.rank)] = value
                        self.population.__setattr__(name, newval)
                    else: # Access the C++ data
                        getattr(self.population.cyInstance, 'set_single_'+name)(self.rank, value)
                else:
                    self.population.__setattr__(name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        desc = 'Neuron of the population ' + self.population.name + ' with rank ' + str(self.rank) + ' (coordinates ' + str(self.population.coordinates_from_rank(self.rank)) + ').\n'
        desc += 'Parameters:\n'
        for param in self.population.parameters:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + '\n'
        desc += '\nVariables:\n'
        for param in self.population.variables:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + '\n'
        return desc

    def __add__(self, other):
        """Allows to join two neurons if they have the same population."""
        if other.population == self.population:
            if isinstance(other, IndividualNeuron):
                return PopulationView(self.population, list(set([self.rank, other.rank])))
            elif isinstance(other, PopulationView):
                return PopulationView(self.population, list(set([self.rank] + other.ranks)))
        else:
            _error("can only add two PopulationViews of the same population.")
            return None
