"""

    Synapse.py

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
import pprint

class RateSynapse(object):
    """
    Base class to define a rate-coded synapse.
    """
    
    def __init__(self, parameters="", equations="", psp=None, functions=None, extra_values=None):
        """ 
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **psp**: post-synaptic potential summed by the post-synaptic neuron.
            * **functions**: additional functions used in the variables' equations.
            
        """                
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.psp = psp
        self.extra_values = extra_values        
         
    def __add__(self, synapse):
        if not isinstance(synapse, RateSynapse):
            return
        
        self._variables.update(synapse.variables) 

    def __str__(self):
        return pprint.pformat( self, depth=4 ) #TODO
        
class SpikeSynapse(object):
    """
    Bae class to define a spiking synapse.
    """

    def __init__(self, parameters="", equations="", pre_spike=None, post_spike=None, functions=None, extra_values=None ):
        """ 
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **pre_spike**: updating of variables when a pre-synaptic spike is received.
            * **post_spike**: updating of variables when a post-synaptic spike is emitted.
            * **functions**: additional functions used in the variables' equations.
            
        """  
        
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.pre_spike = pre_spike
        self.post_spike = post_spike
        self.psp = None
        self.extra_values = extra_values

    def __add__(self, synapse):
        if not isinstance(synapse, SpikeSynapse):
            return
        
        self._variables.update(synapse.variables) 

    def __str__(self):
        return pprint.pformat( self, depth=4 ) #TODO

