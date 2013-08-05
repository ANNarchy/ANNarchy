import parser
import pprint
from Random import *
from Variable import *

#
# maybe we find  a better name for this class
#
class Master:
    """
    Internal base class.
    """
    def __init__(self, debug, order, keyValueArgs):
        """
        extract variable, initializer values and store them locally.
        """
        self.debug = debug
        self.variables = []
        self.order = order

        #
        # sort out all initialization values                
        for key in keyValueArgs:
            alreadyContained, v = self.keyAlreadyContained(key, keyValueArgs[key])

            if not alreadyContained:
                self.variables.append(v)
        
        # debug
        if debug:
            print 'Object '+self.__class__.__name__
            pprint.pprint(self.variables)

    def set(self, keyValueArgs):
        """
        update variable/parameter locally.
        """        
        print keyValueArgs

    def keyAlreadyContained(self, key, value):
        """
        check if a variable/parameter already stored locally.
        If the value is not listed a new object is returned.
        """        
        for v in self.variables:
            if v['name'] == key:
                return True, v

        if isinstance(value, Variable):
            return False, {'name': key, 'var': value }
        else:
            return False, {'name': key, 'init': value }
