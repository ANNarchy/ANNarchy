
from Master import Master

class Neuron(Master):
    """
    Definition of an ANNarchy neuron.
    """
    
    def __init__(self, debug=False, order=[], **keyValueArgs ):
        """
        Description of an ANNarchy neuron. The user describes the initialization of variables as tuple 'variable'='value'. 
        The updating of neuron variables is described as tuple 'variable'='"update rule as string"'.
        
        Reserved key words: 
        * 'debug': enables some useful debug printouts
        * 'order': if dependencies insists between variables, the user can describe an execution order
        """
        if debug:
            print '\n\tNeuron class\n'
        
        Master.__init__(self, debug, order, keyValueArgs)
