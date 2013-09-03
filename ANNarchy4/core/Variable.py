import numpy as np

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
            
        other keys will be rejected.
        """
        self.eq = None
        self.init = None
        self.min = None
        self.max = None
        
        for key in keyValueArgs:

            if key == 'eq':
                self.eq = keyValueArgs[key]
            elif key=='init':
                self.init = keyValueArgs[key]
            elif key=='min':
                self.min = keyValueArgs[key]
            elif key=='max':
                self.max = keyValueArgs[key]
            else:
                print 'unknown key: '+key
         
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
        return object.__setattr__(self, name, value)
                
class Attribute(object):
    """
    Descriptor object, needed to extend Population and 
    Projection classes with new variables
    """
    def __init__(self, variable):
        """
        Initialise Attribute object.
        
        * variable:     variable name as string
        """
        self.variable = variable
        
    def __get__(self, instance):
        val = getattr(instance.cyInstance, self.variable)
        if isinstance(val, np.ndarray):
            return val.reshape(instance.geometry)
        else:
            return val
        
    def __set__(self, instance, value):
        setattr(instance.cyInstance, self.variable, value)

    def __delete__(self, instance):
        pass

class ViewAttribute(object):
    """
    Descriptor object, needed to extend PopulationViews 
    with new variables
    """
    def __init__(self, variable, ranks):
        """
        Initialise Attribute object.
        
        * *variable*: variable name as string
        * *ranks*: participation neurons
        """
        self._variable = variable
        self._ranks = ranks
        
    def __get__(self, instance):
        if self._variable in instance.population.variables:
            return getattr(instance.population.cyInstance, self._variable)[self._ranks]
        else:
            return None
        
    def __set__(self, instance, value):
        if self._variable in instance.population.variables:
            instance.set({self._variable: value})
        else:
            print 'Error: the variable', self._variable, 'does not exist in population', instance.population.name+'.'
            print instance.population.variables
            
    def __delete__(self, instance):
        pass
