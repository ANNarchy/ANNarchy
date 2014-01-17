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
from ANNarchy4.core.Master import Master
from ANNarchy4.core.Variable import Variable, SpikeVariable
from ANNarchy4.core.Global import _error

import pprint

class RateNeuron(Master):
    """
    Python definition of a mean rate coded neuron in ANNarchy4. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters, equations, extra_values={}, functions=None):
        """ 
        The user describes the initialization of variables / parameters. Neuron parameters are described as Variable object consisting of key - value pairs 
        <name> = <initialization value>. The update rule executed in each simulation step is described as equation.
        
        *Parameters*:
        
            * *parameters*: stored as *key-value pairs*. For example:

                .. code-block:: python
        
                    parameters = \"\"\"
                        tau = 10, 
                    \"\"\"

                initializes a parameter ``tau`` with the value 10. Please note, that you may specify several constraints for a parameter:
            
                * *population* : 
                
                * *min*:
                
                * *max*:

            * *equations*: simply as a string contain the equations
            
                .. code-block:: python
        
                    equations = \"\"\"
                        tau * drate / dt + rate = sum(exc)
                    \"\"\"

                spcifies a variable ``rate`` bases on his excitory inputs.

        """        
        Master.__init__(self)
        
        self._convert(parameters, equations, extra_values) 

    def __str__(self):
        """
        Customized print.
        """
        return pprint.pformat( self._variables, depth=4 ) 
        
class SpikeNeuron(Master):
    """
    Python definition of a mean rate coded neuron in ANNarchy4. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters="", equations="", spike=None, reset=None, extra_values={}, functions=None ):
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
        """        
        Master.__init__(self)
        
        # Analyse the provided strings in parameters, equations and extra_values
        self._convert(parameters, equations, extra_values)
        
        # Check if the reset and spike arguments were passed
        if not spike or not reset:
            _error('The *spike* and *reset* arguments must be defined for a spiking neuron')
            exit(0)
        spike_eq = self._prepare_string(spike)[0]
        reset_eq = self._prepare_string(reset)
        
        lside, rside = spike_eq.split('>')
        var_name = lside.replace(' ','')
        var_eq = rside.replace(' ','')
        
        old_var = self._variables[var_name]['var']
        new_var = SpikeVariable(
                            init = old_var.init,
                            eq = old_var.eq,
                            min = old_var.min,
                            max = old_var.max,
                            threshold = rside,
                            reset = reset_eq
                        )
        
        self._variables[var_name]['var'] = new_var 
        
    def __str__(self):
        return pprint.pformat( self._variables, depth=4 )
        
class IndividualNeuron(object):
    """Neuron object returned by the Population.neuron(rank) method.
    
    This only a wrapper around the Population data. It has the same attributes (parameter and variable) as the original population.
    """
    def __init__(self, pop, rank):
        self.__dict__['pop']  = pop
        self.__dict__['rank']  = rank
        self.__dict__['__members__'] = pop.parameters + pop.variables
        self.__dict__['__methods__'] = []
        
    def __getattr__(self, name):
        if name in self.pop.variables:
            return eval('self.pop.cyInstance._get_single_'+name+'(self.rank)')
        elif name in self.pop.parameters:
            return self.pop.__getattribute__(name)
        print('Error: population has no attribute called', name)
        print('Parameters:', self.pop.parameters)
        print('Variables:', self.pop.variables) 
                       
    def __setattr__(self, name, val):
        if hasattr(getattr(self.__class__, name, None), '__set__'):
            return object.__setattr__(self, name, val)
        
        # old version:
        #if name in self.pop.variables:
        #    eval('self.pop.cyInstance._set_single_'+name+'(self.rank, val)')
            
        #TODO: check if this works !!!
        if name in self.pop.variables:
            getattr(self.pop.cyInstance, '_set_single_'+name)(self.rank, val)
        elif name in self.pop.parameters:
            print('Warning: parameters are population-wide, this will affect all other neurons.')
            self.pop.__setattr__(name, val)
            
    def __repr__(self):
        desc = 'Neuron of the population ' + self.pop.name + ' with rank ' + str(self.rank) + ' (coordinates ' + str(self.pop.coordinates_from_rank(self.rank)) + ').\n'
        desc += 'Parameters:\n'
        for param in self.pop.parameters:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + ';'
        desc += '\nVariables:\n'
        for param in self.pop.variables:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + ';'
        return desc
