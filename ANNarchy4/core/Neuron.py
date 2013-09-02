from Master import Master

class Neuron(Master):
    """
    Definition of an ANNarchy neuron.
    """
    
    def __init__(self, debug=False, order=[], **keyValueArgs ):
        """ The user describes the initialization of variables / parameters as key-value pair 'variable'='value'. 
        Neuron variables are described as Variable object consisting of 'variable'='"update rule as string"' and 'init'='initialization value'.
        
        Parameters:
        
            * *keyValueArgs*: dictionary contain the variable / parameter declarations as key-value pairs. For example:

                .. code-block:: python
        
                    tau = 5.0, 

                initializes a parameter ``tau`` with the value 5.0 

                .. code-block:: python
        
                    rate = Variable( init=0.0, rate="tau * drate / dt + rate = sum(exc)" )

                and variable ``rate`` bases on his excitory inputs.

            * *order*: execution order of update rules. For exampple:
                
                .. code-block:: python
                
                    order = ['mp', 'rate'] 
              
                leads to execution of the mp update and then the rate will updated.
                
            * *debug*: prints all defined variables/parameters to standard out (default = False)
        """
        Master.__init__(self, debug, order, keyValueArgs)
        
        
class IndividualNeuron(object):
    """Neuron object returned by the Population.neuron(rank) method.
    
    This only a wrapper around the Population data. It has the same attributes (parameter and variable) as the original population.
    """
    def __init__(self, pop, rank):
        self.__dict__['pop']  = pop
        self.__dict__['rank']  = rank
        self.__dict__['__members__'] = pop.parameters + pop.variables
        self.__dict__['__methods__'] = []
        
    def __getattr__(self, name):
        if name in self.pop.variables:
            return eval('self.pop.cyInstance._get_single_'+name+'(self.rank)')
        elif name in self.pop.parameters:
            return self.pop.__getattribute__(name)
        print 'Error: population has no attribute called', name
        print 'Parameters:', self.pop.parameters
        print 'Variables:', self.pop.variables 
                       
    def __setattr__(self, name, val):
        if hasattr(getattr(self.__class__, name, None), '__set__'):
            return object.__setattr__(self, name, val)
        if name in self.pop.variables:
            eval('self.pop.cyInstance._set_single_'+name+'(self.rank, val)')
        elif name in self.pop.parameters:
            print 'Warning: parameters are population-wide, this will affect all other neurons.'
            self.pop.__setattr__(name, val)
            
    def __repr__(self):
        desc = 'Neuron of the population ' + self.pop.name + ' with rank ' + str(self.rank) + ' (coordinates ' + str(self.pop.coordinates_from_rank(self.rank)) + ').\n'
        desc += 'Parameters:\n'
        for param in self.pop.parameters:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + ';'
        desc += '\nVariables:\n'
        for param in self.pop.variables:
            desc += '  ' + param + ' = ' + str(self.__getattr__(param)) + ';'
        return desc
