import Global

class Descriptor(object):
    """
    Base class for Projection and Population class to 
    extend these with attributes after instantiation.
    
    refer: blog.brianbeck.com/post/74086029/instance-descriptors
    """
    def __getattribute__(self, name):
        """
        getter
        """
        value = object.__getattribute__(self, name)
        if hasattr(value, '__get__'):
            value = value.__get__(self)
        return value

    def __setattr__(self, name, value):
        """
        setter
        """        
        try:
            obj = object.__getattribute__(self, name)
        except AttributeError:
            pass
        else:
            if hasattr(obj, '__set__'):
                return obj.__set__(self, value)

        if not Global._compiled :
            if hasattr(self, 'initialized'):
                if name in self.generator._variable_names():
                    self.generator._update_value(name, value)
                else:
                    self.generator._add_value(name, value)
        
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

