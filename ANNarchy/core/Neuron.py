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
import numpy as np
from ANNarchy.core.Global import _error
from ANNarchy.core.PopulationView import PopulationView

import pprint

class RateNeuron(object):
    """
    Python definition of a mean rate coded neuron in ANNarchy. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters="", equations="", extra_values={}, functions=None):
        """ 
        The user describes the initialization of variables / parameters. Neuron parameters are described as Variable object consisting of key - value pairs 
        <name> = <initialization value>. The update rule executed in each simulation step is described as equation.
        
        *Parameters*:
        
            * TODO

        """        
        
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.extra_values = extra_values    

    def __add__(self, neuron):
        if not isinstance(neuron, RateNeuron):
            return
        
        self._variables.update(neuron.variables) 

    def __str__(self):
        """
        Customized print.
        """
        return pprint.pformat( self, depth=4 ) 
        
class SpikeNeuron(object):
    """
    Python definition of a mean rate coded neuron in ANNarchy. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters="", equations="", spike=None, reset=None, refractory = None, extra_values={}, functions=None ):
        """ 
        The user describes the initialization of variables / parameters. Neuron parameters are described as Variable object consisting of key - value pairs 
        <name> = <initialization value>. The update rule executed in each simulation step is described as equation.
        
        *Parameters*:
        
            * *parameters*: stored as *key-value pairs*. For example:

                .. code-block:: python
        
                    parameters = \"\"\"
                        a = 0.2
                        b = 2
                        c = -65 
                    \"\"\"

                initializes a parameter ``tau`` with the value 10. Please note, that you may specify several constraints for a parameter:
            
                * *population* : 
                
                * *min*:
                
                * *max*:

            * *equations*: simply as a string contain the equations
            
                .. code-block:: python
        
                    equations = \"\"\"
                        dv/dt = 0.04 * v * v + 5*v + 140 -u + I
                    \"\"\"

                spcifies a variable ``rate`` bases on his excitory inputs.
                
            * *spike*: denotes the conditions when a spike should be emited.

                .. code-block:: python
        
                    spike = \"\"\"
                        v > treshold
                    \"\"\"

            * *reset*: denotes the equations executed after a spike

                .. code-block:: python
        
                    reset = \"\"\"
                        u = u + d
                        v = c
                    \"\"\"
                    
            * *refractory*: denotes the refractory time of a neuron after a spike was emitted
            
                .. code-block:: python
        
                    refractory = 5    # 5 ms
                
        """        
        
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.spike = spike
        self.reset = reset
        self.refractory = refractory
        self.extra_values = extra_values
        
    def __str__(self):
        return pprint.pformat( self, depth=4 )
        
class IndividualNeuron(object):
    """Neuron object returned by the Population.neuron(rank) method.
    
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
                val = (self.population.get(name))
                if isinstance(val, np.ndarray):
                    return val[self.population.coordinates_from_rank(self.rank)]
                else:
                    return val
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
                if name in self.population.description['local']:
                    if not self.population.initialized: # Store it in the temporary array
                        newval = self.population.get(name)
                        newval[self.population.coordinates_from_rank(self.rank)] = value
                        self.population.__setattr__(name, newval)
                    else: # Access the C++ data 
                        eval('self.population.cyInstance._set_single_'+name+'('+str(self.rank)+', '+str(value)+')')
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
