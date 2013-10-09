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
from Master import Master
import Global

from ANNarchy4 import parser

class Synapse(Master):
    """
    Definition of a synapse in ANNarchy4. This object is intended to encapsulate synapse equations, for learning or modified post-synaptic potential, and is further used in projection class.
    """

    def __init__(self, debug=False, order=[], **key_value_args):
        """ The user describes the initialization of variables / parameters as *key-value pairs* 'variable'='value'. 
        Synapse variables are described as Variable object consisting of 'variable'='"update rule as string"' and 'init'='initialzation value'.
        
        *Parameters*:
        
            * *key_value_args*: dictionary contain the variable / parameter declarations as key-value pairs. For example:

                .. code-block:: python
        
                    tau = 5.0, 

                initializes a parameter ``tau`` with the value 5.0 

                .. code-block:: python
        
                    value = Variable( init=0.0, rate="tau * drate / dt + value = pre.rate * 0.1" )

                and a simple update of the synaptic weight.
                
                .. warning::
                    
                    Please note, that automatically all key-value pairs provided to the function, except ``debug`` and ``order``, are assigned to *key_value_args*.

            * *order*: execution order of update rules.

                .. warning::
                    
                    if you use the order key, the value need to contain **all** variable names.
                            
            * *debug*: prints all defined variables/parameters to standard out (default = False)

                .. hint::            
                    
                    An experimental feature, currently not fully implemented.
            
        """
        Master.__init__(self, debug, order, key_value_args)

    def _global_operations(self):
        var, g_op = parser.SynapseAnalyser(self.variables).parse()        
        return g_op