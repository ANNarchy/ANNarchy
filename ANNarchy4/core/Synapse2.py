from Variable import Variable
from Master2 import Master2

from ANNarchy4 import parser

import re
import pprint

class RateSynapse(Master2):
    def __init__(self, parameters="", equations="", psp=None, extra_values=None, functions=None):
        Master2.__init__(self)
        
        self._convert(parameters, equations)
        
        if psp:
            psp = 'psp = ' + self._prepare_string(psp)
            var = Variable(init=0.0, eq=psp)
            var._validate() 
            
            self._variables[ 'psp' ] = {'type' : 'local' ,'var': var } 
         
    def __str__(self):
        return pprint.pformat( self._variables, depth=4 )
            
    def _global_operations(self):
        var, g_op = parser.SynapseAnalyser(self._variables, [], []).parse()
        return g_op
        
class SpikeSynapse(Master2):

    def __init__(self, parameters="", equations="", psp = None, extra_values=None, functions=None ):
        Master2.__init__(self)
        
        self._convert(parameters, equations)
        
        if psp:
            psp = 'psp = ' + ''.join(self._prepare_string(psp))
            var = Variable(init=0.0, eq=psp)
            var._validate() 
            
            self._variables['psp'] = { 'type':'local', 'var': var }

    def __str__(self):
        return pprint.pformat( self._variables, depth=4 )
            
    def _global_operations(self):
        var, g_op = parser.SynapseAnalyser(self._variables, [], []).parse()        
        return g_op
