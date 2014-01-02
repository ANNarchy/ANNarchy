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
from . import Global

#===============================================================================
# class Descriptor(object):
#     """
#     Base class for Projection and Population class to 
#     extend these with attributes after instantiation.
#     
#     refer: blog.brianbeck.com/post/74086029/instance-descriptors
#     """
#     def __getattribute__(self, name):
#         """
#         getter
#         """
#         
#         try:
#             value = object.__getattribute__(self, name)
#             print('value',value)
#             
#         except Exception as err:
#             if not Global._compiled :
#                 if name in object.__getattribute__(self, 'generator')._variable_names():
#                     tmp = object.__getattribute__(self, 'generator')._get_value(name)
#                     if tmp != None:
#                         value = tmp
#                         return value
#             else:
#                 print(err)
#                 pass
#         else:
#             if hasattr(value, '__get__'):
#                 value = value.__get__(self)
#                 return value
# 
#     def __setattr__(self, name, value):
#         """
#         setter
#         """        
#         try:
#             obj = object.__getattribute__(self, name)
#         except AttributeError:
#             pass
#         else:
#             if hasattr(obj, '__set__'):
#                 return obj.__set__(self, value)
#         if not Global._compiled :
#             print('hasattr', self, name)
#             if hasattr(self, 'initialized'):
#                 if name in self.generator._variable_names():
#                     self.generator._update_value(name, value)
#                     return None
#                 else:
#                     self.generator._add_value(name, value)
#                     return None
#         return object.__setattr__(self, name, value)
#===============================================================================
          
class Descriptor(object):
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if hasattr(value, '__get__'):
            value = value.__get__(self, self.__class__)
        return value

    def __setattr__(self, name, value):
        try:
            obj = object.__getattribute__(self, name)
        except AttributeError:
            pass
        else:
            if hasattr(obj, '__set__'):
                return obj.__set__(self, value)
        return object.__setattr__(self, name, value)

      
class Attribute(object):
    """
    Descriptor object, needed to extend Population and 
    Projection classes with attributes.
    """
    def __init__(self, variable):
        """
        Initialise Attribute object.
        
        * variable:     variable name as string
        """
        self.variable = variable
        
    def __get__(self, instance):
        return instance.get(self.variable)
        
    def __set__(self, instance, value):
        instance.set({self.variable: value})

    def __delete__(self, instance):
        pass


# -*- coding: utf-8 -*-

