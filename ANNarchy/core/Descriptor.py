"""

    Descriptor.py
    
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
from ANNarchy.core import Global

class Descriptor(object):

    def __init__(self):
        
        object.__setattr__(self, 'compiled', False) # distinguish pre and post compilation values
        
    def _compile(self):
        object.__setattr__(self, 'compiled', True)
        
    def __getattr__(self, name):
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif name == 'attributes':
            return object.__getattribute__(self, 'attributes')
        elif hasattr(self, 'attributes'):
            if name in self.parameters:
                return self.param_init[name]
            elif name in self.variables:
                return self.var_init[name]
        return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            object.__setattr__(self, name, value)
        elif name == 'attributes':
            return object.__setatt__(self, name, value)
        
        return object.__setattr__(self, name, value)
        

