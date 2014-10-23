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
import ANNarchy.core.Global as Global
from ANNarchy.parser.SingleAnalysis import analyse_synapse

        
class Synapse(object):
    """
    Base class to define a synapse.
    """
    def __init__(self, parameters="", equations="", psp=None, operation='sum', pre_spike=None, post_spike=None, functions=None, name=None, description=None ):
        """ 
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **psp**: influence of a single synapse on the post-synaptic neuron (default for rate-coded: w*pre.r).
            * **operation**: operation (sum, max, min, mean) performed by the post-synaptic neuron on the individual psp (rate-coded only, default=sum).
            * **pre_spike**: updating of variables when a pre-synaptic spike is received (spiking only).
            * **post_spike**: updating of variables when a post-synaptic spike is emitted (spiking only).
            * **functions**: additional functions used in the equations.
            * **name**: name of the synapse type (used for reporting only).
            * **description**: short description of the synapse type (used for reporting).
            
        """  
        
        # Store the parameters and equations
        self.parameters = parameters
        self.equations = equations
        self.functions = functions
        self.pre_spike = pre_spike
        self.post_spike = post_spike
        self.psp = psp
        self.operation = operation

        # Type of the synapse
        self.type = 'spike' if pre_spike else 'rate'

        # Check the operation
        if self.type == 'spike' and self.operation != 'sum':
            Global._error('spiking synapses can only perform a sum of presynaptic potentials.')
            exit(0)
        if not self.operation in ['sum', 'min', 'max', 'mean']:
            Global._error('the only operations permitted are: sum (default), min, max, mean.')
            exit(0)

        # Description
        self.description = None

        # Reporting
        if name:
            self.name = name
        else:
            self.name = 'Spiking synapse' if self.type == 'spike' else 'Rate-coded synapse'
        if description:
            self.short_description = description
        else:
            self.short_description = "User-defined model of a spiking synapse." if self.type == 'spike' else "User-defined model of a rate-coded synapse."

    def _analyse(self):
        # Analyse the synapse type
        if not self.description:
            self.description = analyse_synapse(self)

    def __add__(self, synapse):        
        self._variables.update(synapse.variables) 

    def __str__(self):
        import pprint
        return pprint.pformat( self, depth=4 ) #TODO


class RateSynapse(Synapse):
    """
    Base class to define a rate-coded synapse.
    """
    
    def __init__(self, parameters="", equations="", psp=None, functions=None):
        """ 
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **psp**: post-synaptic potential summed by the post-synaptic neuron.
            * **functions**: additional functions used in the variables' equations.
            
        """         
        Synapse.__init__(self, parameters=parameters, equations=equations, psp=psp, functions=functions)
        
class SpikeSynapse(Synapse):
    """
    Bae class to define a spiking synapse.
    """

    def __init__(self, parameters="", equations="", psp=None, pre_spike=None, post_spike=None, functions=None):
        """ 
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **psp**: post-synaptic potential summed by the post-synaptic neuron.
            * **pre_spike**: updating of variables when a pre-synaptic spike is received.
            * **post_spike**: updating of variables when a post-synaptic spike is emitted.
            * **functions**: additional functions used in the variables' equations.
            
        """  
        Synapse.__init__(self, parameters=parameters, equations=equations, psp=psp, pre_spike=pre_spike, post_spike=post_spike, functions=functions)

