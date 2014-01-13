from Variable import Variable
from SpikeVariable import SpikeVariable

from Master2 import Master2

import pprint

class RateNeuron(Master2):
    def __init__(self, parameters, equations, extra_values=None, functions=None):
        Master2.__init__(self)
        
        self._convert(parameters, equations) 

    def __str__(self):
        str = pprint.pformat( self._variables, depth=4 )
        return str
        
class SpikeNeuron(Master2):

    def __init__(self, parameters, equations, spike, reset, extra_values=None, functions=None ):
        Master2.__init__(self)
        
        self._convert(parameters, equations)
        
        spike_eq = self._prepare_string(spike)[0]
        reset_eq = self._prepare_string(reset)
        
        lside, rside = spike_eq.split(':')
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
        str = pprint.pformat( self._variables, depth=4 )
            
        #str += 'spike:\n', self._spike
        #str += 'reset:\n', self._reset
        return str
