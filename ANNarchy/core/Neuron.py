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
from ANNarchy.core.Global import _error
from ANNarchy.parser.SingleAnalysis import analyse_neuron


class Neuron(object):
    """
    Base class to define a neuron.
    """    
    def __init__(self, parameters="", equations="", spike=None, reset=None, refractory = None, functions=None, extra_values={} ):
        """         
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.                
            * **spike**: condition to emit a spike (only for spiking neurons).
            * **reset**: changes to the variables after a spike (only for spiking neurons).                  
            * **refractory**: refractory period of a neuron after a spike (only for spiking neurons).
                
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

        # Analyse the neuron type
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
    def __init__(self, parameters="", equations="", functions=None, extra_values={}):
        """        
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.

        """
        Neuron.__init__(self, parameters=parameters, equations=equations, functions=functions, extra_values=extra_values) 
        
class SpikeNeuron(Neuron):
    """
    Base class to define a spiking neuron.
    """    
    def __init__(self, parameters="", equations="", spike=None, reset=None, refractory = None, functions=None, extra_values={} ):
        """         
        *Parameters*:
        
            * **parameters**: parameters of the neuron and their initial value.
            * **equations**: equations defining the temporal evolution of variables.
            * **functions**: additional functions used in the variables' equations.                
            * **spike**: condition to emit a spike.
            * **reset**: changes to the variables after a spike                    
            * **refractory**: refractory period of a neuron after a spike.
                
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, functions=functions, spike=spike, reset=reset, refractory=refractory, extra_values=extra_values) 

