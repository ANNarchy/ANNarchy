import parser
import pprint
from Random import *
from Variable import *

#
# maybe we find  a better name for this class
#
class Master:
    """
    Intern base class.
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
            alreadyContained, v = self.keyAlreadyContained(key)

            if not alreadyContained:
                self.variables.append(v)

            if isinstance(keyValueArgs[key], Variable):
                v['var'] = keyValueArgs[key]
            else:
                v['init'] = keyValueArgs[key]
        
        # debug
        if debug:
            print 'Object '+self.__class__.__name__
            pprint.pprint(self.variables)

    def set(self, keyValueArgs):
        print keyValueArgs

    def keyAlreadyContained(self, key):
        for v in self.variables:
            if v['name'] == key:
                return True, v

        return False, {'name': key, 'init': None, 'var': None}
