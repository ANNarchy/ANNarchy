from Variable import Variable
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
        
        self._spike = spike
        self._reset = reset 

    def __str__(self):
        str = pprint.pformat( self._variables, depth=4 )
            
        #str += 'spike:\n', self._spike
        #str += 'reset:\n', self._reset
        return str
