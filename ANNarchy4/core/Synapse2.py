from Variable import Variable
from Master2 import Master2

import pprint

class RateSynapse(Master2):
    def __init__(self, parameters, equations, psp=None, extra_values=None, functions=None):
        Master2.__init__(self)
        
        print 'variables:'
        self._convert(parameters, equations) 
        pprint.pprint( self._variables, depth=4 )
        print '\n'
        
        print 'psp:'
        print psp
        
class SpikeSynapse(Master2):

    def __init__(self, parameters, equations, spike, reset, extra_values=None, functions=None ):
        Master2.__init__(self)
        
        print 'variables:'
        self._convert(parameters, equations) 
        pprint.pprint( self._variables, depth=4 )
        print '\n'
            
        print 'spike:\n', spike
        print 'reset:\n', reset
