from ANNarchy4.core.Variable import Descriptor, ViewAttribute

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
            exec("self."+var+"= ViewAttribute('"+var+"', self.ranks)")
        
    def get(self, value):
        """
        Get current variable/parameter value
        
        Parameter:
        
            * *value*: value name as string
        """
        if value in self.population.variables:
            all_val = getattr(self.population, value) #directly access the one-dimensional array
            return all_val[self.ranks] 
        elif value in self.population.parameters:
            return self.population.get_parameter(value)
        else:
            print "Error: population does not contain value: '"+value+"'"
        
    def set(self, value):
        """ Updates neuron variable/parameter definition
        
        Parameters:
        
            * *value*: value need to be update
            
                .. code-block:: python
                
                    set( {'tau' : 20, 'rate'= np.random.rand((8,8)) } )
        """
        for val_key in value.keys():
            if hasattr(self.population, val_key):
                val = getattr(self.population, val_key)
                if isinstance(value[val_key], int) or isinstance(value[val_key], float): # one for all
                    val[self.ranks[:]] = value[val_key]
                elif self.ranks.size == value[val_key].size: # distinct
                    val[self.ranks[:]] = value[val_key][:]
                else:
                    print 'Error: mismatch between amount of neurons in population view and given data.'
                    return
                
                setattr(self.population, val_key, val)
            else:
                print "Error: population does not contain value: '"+val_key+"'"
                return
                
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
