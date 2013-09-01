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
