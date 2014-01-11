from Variable import Variable
from Master2 import Master2

import re
import pprint

class RateSynapse(Master2):
    def __init__(self, parameters, equations, psp=None, extra_values=None, functions=None):
        Master2.__init__(self)
        
        self._convert(parameters, equations)
        
        if psp:
            psp = re.sub('\#[\s\S]+', ' ', psp) # remove comment
            var = Variable(init=0.0, eq=psp)
            var._validate() 
            
            self._variables.append({'name': 'psp', 'var': var })
         
    def __str__(self):
        print 'variables:'
        pprint.pprint( self._variables, depth=4 )
        print '\n'
        
class SpikeSynapse(Master2):

    def __init__(self, parameters, equations, psp = None, extra_values=None, functions=None ):
        Master2.__init__(self)
        
        self._convert(parameters, equations)
        
        self._spike = spike
        self._reset = reset 

        if psp:
            psp = re.sub('\#[\s\S]+', ' ', psp) # remove comment
            var = Variable(init=0.0, eq=psp)
            var._validate() 
            
            self._variables.append({'name': 'psp', 'var': var })

    def __str__(self):
        print 'variables:'
        pprint.pprint( self._variables, depth=4 )
        print '\n'
            
        print 'spike:\n', self._spike
        print 'reset:\n', self._reset
