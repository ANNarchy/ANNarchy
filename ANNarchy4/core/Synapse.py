from Master import Master
import Global

from ANNarchy4 import parser

class Synapse(Master):
    """
    Definition of an ANNarchy synapse.
    """

    def __init__(self, debug=False, order=[], **keyValueArgs):
        """ The user describes the initialization of variables / parameters as key-value pair 'variable'='value'. 
        Synapse variables are described as Variable object consisting of 'variable'='"update rule as string"' and 'init'='initialzation value'.
        
        Parameters:
        
            * *keyValueArgs*: dictionary contain the variable / parameter declarations as key-value pairs. For example:

                .. code-block:: python
        
                    tau = 5.0, 

                initializes a parameter ``tau`` with the value 5.0 

                .. code-block:: python
        
                    value = Variable( init=0.0, rate="tau * drate / dt + value = pre.rate * 0.1" )

                and a simple update of the synaptic weight.

            * *order*: execution order of update rules.
                            
            * *debug*: prints all defined variables/parameters to standard out (default = False)
        """

        Master.__init__(self, debug, order, keyValueArgs)

        self.parser = parser.SynapseAnalyser(self.variables)
        self.parsed_variables, self.global_operations = self.parser.parse()
