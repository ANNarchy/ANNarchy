from ANNarchy4.core.Descriptor import Descriptor, Attribute
from ANNarchy4.core.Random import RandomDistribution
import numpy as np

class PopulationView(Descriptor):
    """ Container representing a subset of neurons of a Population."""
    
    def __init__(self, population, ranks):
        """
        Create a view of a subset of neurons within the same population.
        
        Parameter:
        
            * *population*: population object
            * *ranks: list or numpy array containing the ranks of the selected neurons.
        """
        self.population = population
        self.ranks = ranks
        self.size = len(self.ranks)
        
        for var in self.population.variables + self.population.parameters:
            setattr(self, var, Attribute(var))
        
    def get(self, name):
        """
        Returns current variable/parameter value.
        
        Parameter:
        
            * *name*: name of the parameter/variable.
        """
        if name in self.population.variables:
            all_val = getattr(self.population, name).reshape(self.population.size) #directly access the one-dimensional array
            return all_val[self.ranks] 
        elif name in self.population.parameters:
            return self.population.get_parameter(name)
        else:
            print "Error: population does not have a parameter/variable called", value + "."
        
    def set(self, value):
        """ Updates neuron variable/parameter definition.
        
        Parameters:
        
            * *value*: dictionary of parameters/variables to be updated for the corresponding subset of neurons. It can be a single value or a list/1D array of the same size as the PopulationView.
            
                .. code-block:: python
                
                    >>> subpop = pop[0:5]
                    >>> subpop.set( {'tau' : 20, 'rate'= np.random.rand(subpop.size) } )
                    
        .. warning::
        
            If you modify the value of a parameter, this will be the case for ALL neurons of the population, not only the subset.
        """
        for val_key in value.keys():
            if hasattr(self.population, val_key):
                # Check the value
                if isinstance(value[val_key], np.ndarray): # np.array
                    if value[val_key].ndim >1 or len(value[val_key]) != self.size:
                        print 'Error: you can only provide a 1D list/array of the same size as the PopulationView', self.size
                        return None
                    if val_key in self.population.parameters:
                        print 'Error: you can only provide a single value for parameters.'
                        return None
                    # Assign the value
                    for rk in self.ranks:
                        setattr(self.population.neuron(rk), value[val_key][rk])
                elif isinstance(value[val_key], list): # list
                    if value[val_key].ndim >1 or len(value[val_key]) != self.size:
                        print 'Error: you can only provide a 1D list/array of the same size as the PopulationView', self.size
                        return None
                    if val_key in self.population.parameters:
                        print 'Error: you can only provide a single value for parameters.'
                        return None
                    # Assign the value
                    for rk in self.ranks:
                        setattr(self.population.neuron(rk), value[val_key][rk])   
                elif isinstance(value[val_key], RandomDistribution): # random distribution
                    for rk in self.ranks:
                        setattr(self.population.neuron(rk), float(value[val_key].getValue()))
                else: # single value
                    for rk in self.ranks:
                        setattr(self.population.neuron(rk), value[val_key])
            else:
                print "Error: population does not contain value: '"+val_key+"'"
                return None
                
    def __add__(self, other):
        """Allows to join two PopulationViews if they have the same population."""
        if other.population == self.population:
            return PopulationView(self.population, list(set(self.ranks + other.ranks)))
        else:
            print 'Error: can only add two PopulationViews of the same population.'
            return None
                
    def __repr__(self):
        """Defines the printing behaviour."""
        string ="PopulationView of " + str(self.population.name) + '\n'
        string += '  Ranks: ' +  str(self.ranks)
        string += '\n'
        for rk in self.ranks:
            string += '* ' + str(self.population.neuron(rk)) + '\n'
        return string
