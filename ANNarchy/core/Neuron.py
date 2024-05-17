"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.parser.AnalyseNeuron import analyse_neuron
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern import Messages
import numpy as np

class Neuron :
    """
    Base class to define a neuron.

    :param parameters: parameters of the neuron and their initial value.
    :param equations: equations defining the temporal evolution of variables.
    :param functions: additional functions used in the variables' equations.
    :param spike: condition to emit a spike (only for spiking neurons).
    :param axon_spike: condition to emit an axonal spike (only for spiking neurons and optional). The axonal spike can appear additional to the spike and is independent from refractoriness of a neuron.
    :param reset: changes to the variables after a spike (only for spiking neurons).
    :param axon_reset: changes to the variables after an axonal spike (only for spiking neurons).
    :param refractory: refractory period of a neuron after a spike (only for spiking neurons).
    :param name: name of the neuron type (used for reporting only).
    :param description: short description of the neuron type (used for reporting).
    """
    # Default name and description for reporting
    _default_names = {'rate': "Rate-coded neuron", 'spike': "Spiking neuron"}

    def __init__(self, parameters:str="", equations:str="", spike:str=None, axon_spike:str=None, reset:str=None, axon_reset:str=None, refractory:str = None, functions:str=None, name:str="", description:str="", extra_values:dict={} ):

        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.spike = spike
        self.axon_spike = axon_spike
        self.reset = reset
        self.axon_reset = axon_reset
        self.refractory = refractory
        self.extra_values = extra_values

        # Find the type of the neuron
        self.type = 'spike' if self.spike else 'rate'

        # Not available by now ...
        if axon_spike and get_global_config('paradigm') != "openmp":
            Messages._error("Axonal spike conditions are only available for openMP by now.")

        # Reporting
        if not hasattr(self, '_instantiated') : # User-defined
            GlobalObjectManager().add_neuron_type(self)
        elif len(self._instantiated) == 0: # First instantiated of the class
            GlobalObjectManager().add_neuron_type(self)
        self._rk_neurons_type = GlobalObjectManager().num_neuron_types()

        if name:
            self.name = name
        else:
            self.name = self._default_names[self.type]

        if description:
            self.short_description = description
        else:
            self.short_description = "User-defined model of a spiking neuron." if self.type == 'spike' else "User-defined model of a rate-coded neuron."

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
Axonal spiking condition:
""" + str(self.axon_spike) + """
Reset after a spike:
""" + str(self.reset)

        return text


class IndividualNeuron :
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
                    ctype = self.population._get_attribute_cpp_type(name)
                    if name in self.population.neuron_type.description['local']:
                        return self.population.cyInstance.get_local_attribute(name, self.rank, ctype)
                    else:
                        return self.population.cyInstance.get_global_attribute(name)
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
                        ctype = self.population._get_attribute_cpp_type(name)
                        self.population.cyInstance.set_local_attribute(name, self.rank, value, ctype)
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
            Messages._error("can only add two PopulationViews of the same population.")
            return None
