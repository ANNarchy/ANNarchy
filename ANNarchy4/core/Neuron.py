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
from .Master import Master

class Neuron(Master):
    """
    Python definition of a neuron in ANNarchy4. This object is intended to encapsulate neuronal equations and further used in population class.
    """
    
    def __init__(self, debug=False, order=[], **key_value_args ):
        """ The user describes the initialization of variables / parameters as *key-value pairs* 'variable'='value'. 
        Neuron variables are described as Variable object consisting of 'variable'='"update rule as string"' and 'init'='initialization value'.
        
        *Parameters*:
        
            * *key_value_args*: dictionary contain the variable / parameter declarations as *key-value pairs*. For example:

                .. code-block:: python
        
                    tau = 5.0, 

                initializes a parameter ``tau`` with the value 5.0 

                .. code-block:: python
        
                    rate = Variable( init=0.0, rate="tau * drate / dt + rate = sum(exc)" )

                and variable ``rate`` bases on his excitory inputs.

                .. warning::
                    
                    Please note, that automatically all key-value pairs provided to the function, except ``debug`` and ``order``, are assigned to *key_value_args*.

            * *order*: execution order of update rules. All variables of one neuron are stored in a dictionary but are not necessarily evaluated in the order how the designer may provide them to this object. If the equations depend on each other you may provide the right execution order. For example:
                
                .. code-block:: python
                
                    order = ['mp', 'rate'] 
              
                leads to execution of the mp update and then the rate will updated.
                
                .. warning::
                    
                    if you use the order key, the value need to contain **all** variable names.
            
            * *debug*: prints all defined variables/parameters to standard out (default = False)
            
                .. hint::            
                    
                    An experimental feature, currently not fully implemented.
                
        """
        Master.__init__(self, debug, order, key_value_args)
        
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
