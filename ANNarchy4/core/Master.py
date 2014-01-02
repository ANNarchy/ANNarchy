"""

    Master.py
    
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
import pprint

from .Variable import Variable
from .SpikeVariable import SpikeVariable

#
# maybe we find  a better name for this class
#
class Master(object):
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
        self.spike_vars = 0
        
        #
        # sort out all initialization values                
        for key in keyValueArgs:
            alreadyContained, v = self.keyAlreadyContained(key, keyValueArgs[key])

            if not alreadyContained:
                self.variables.append(v)
        
        # debug
        if debug:
            print('Object '+self.__class__.__name__)
            pprint.pprint(self.variables)

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
        elif isinstance(value, SpikeVariable):
            self.spike_vars +=1
            return False, {'name': key, 'var': value }
        else:
            return False, {'name': key, 'init': value }
