"""

    Variable.py
    
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

class Variable(object):
    """
    Variable representation in ANNarchy.
    """
    def __init__(self,  **keyValueArgs):
        """
        Set of key-value pairs defining this variable.
        
        Possible keys are:
        
        * *eq*: variable equation
        * *init*: initialization
        * *min*: minimal value for this variable
        * *max*: maximal value for this variable
        * *type*: data type, available are: float, int, bool (default is float)            
            
        other keys will be rejected.
        """
        self.eq = None
        self.init = None
        self.min = None
        self.max = None
        self.type = None
        
        for key in keyValueArgs:

            if key == 'eq':
                self.eq = keyValueArgs[key]
            elif key=='init':
                self.init = self._convert_str_to_value(keyValueArgs[key])
            elif key=='min':
                self.min = self._convert_str_to_value(keyValueArgs[key])
            elif key=='max':
                self.max = self._convert_str_to_value(keyValueArgs[key])
            elif key=='type':
                self.type = self._convert_str_to_type(keyValueArgs[key])
            else:
                print 'unknown key: '+key

    def _convert_str_to_value(self, string):
        if type(string) != str:
            return string 
        
        if string.find('True') != -1:
            init = True
        elif string.find('False') != -1:
            init = False
        else:
            try:
                init = int(string)
            except ValueError:
                try:
                    init = float(string)
                except ValueError:
                    init = string
                
        return init

    def _convert_str_to_type(self, string):
        string = string.replace(' ','')
        
        if string.find('int') != -1:
            type = int
        elif string.find('float') != -1:
            type = float
        elif string.find('bool') != -1:
            type = bool
        else:
            print 'Error: unknown type', string
            
        return type

    def _validate(self):
        #
        # for later operations it's not positive if no
        # default is set
        if self.init == None:
            if self.type != None:
                self.init = self.type(0.0)
            else:
                self.type = float
                self.init = self.type(0.0)
 
        if type(self.init) != self.type:
            if isinstance(self.init, (bool, float, int)):
                if self.type != bool and type(self.init) != bool:
                    self.init = float(self.init)
                    self.type = float
            else:
                self.type = 'eq'
    
    def __add__(self, other):
        """
        Called if two Variable objects are added up.
        """
        if not isinstance(other, Variable):
            print 'Error: ...'
            return

        if self.init != None and other.init != None:
            print 'WARNING: init value will be overwritten.'

        if other.init:
            self.init = other.init
            self.type = type(self.init)

        self.eq = other.eq
        self.min = other.min
        self.max = other.max
        
    #
    # some customization stuff, maybe needed later.
    def __str__(self):
        itemDir = self.__dict__
        str = '['
        for i in itemDir:
            str += '{0} : {1}, '.format(i, itemDir[i])
        str+= ']'
         
        return str
     
    def __repr__(self):
        return "<%s instance at %li> %s" % (self.__class__.__name__, id(self), self.__str__())  
