from Variable import Variable
from SpikeVariable import SpikeVariable

from Master2 import Master2

import pprint

class RateNeuron(Master2):
    """
    Python definition of a mean rate coded neuron in ANNarchy4. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters, equations, extra_values={}, functions=None):
        """ 
        The user describes the initialization of variables / parameters. Neuron parameters are described as Variable object consisting of key - value pairs 
        <name> = <initialization value>. The update rule executed in each simulation step is described as equation.
        
        *Parameters*:
        
            * *parameters*: stored as *key-value pairs*. For example:

                .. code-block:: python
        
                    parameters = \"\"\"
                        tau = 10, 
                    \"\"\"

                initializes a parameter ``tau`` with the value 10. Please note, that you may specify several constraints for a parameter:
            
                * *population* : 
                
                * *min*:
                
                * *max*:

            * *equations*: simply as a string contain the equations
            
                .. code-block:: python
        
                    equations = \"\"\"
                        tau * drate / dt + rate = sum(exc)
                    \"\"\"

                spcifies a variable ``rate`` bases on his excitory inputs.

        """        
        Master2.__init__(self)
        
        self._convert(parameters, equations, extra_values) 

    def __str__(self):
        """
        Customized print.
        """
        return pprint.pformat( self._variables, depth=4 ) 
        
class SpikeNeuron(Master2):
    """
    Python definition of a mean rate coded neuron in ANNarchy4. This object is intended to encapsulate neuronal equations and further used in population class.
    """    
    def __init__(self, parameters="", equations="", spike="", reset="", extra_values={}, functions=None ):
        """ 
        The user describes the initialization of variables / parameters. Neuron parameters are described as Variable object consisting of key - value pairs 
        <name> = <initialization value>. The update rule executed in each simulation step is described as equation.
        
        *Parameters*:
        
            * *parameters*: stored as *key-value pairs*. For example:

                .. code-block:: python
        
                    parameters = \"\"\"
                        a = 0.2
                        b = 2
                        c = -65 
                    \"\"\"

                initializes a parameter ``tau`` with the value 10. Please note, that you may specify several constraints for a parameter:
            
                * *population* : 
                
                * *min*:
                
                * *max*:

            * *equations*: simply as a string contain the equations
            
                .. code-block:: python
        
                    equations = \"\"\"
                        dv/dt = 0.04 * v * v + 5*v + 140 -u + I
                    \"\"\"

                spcifies a variable ``rate`` bases on his excitory inputs.
                
            * *spike*: denotes the conditions when a spike should be emited.

                .. code-block:: python
        
                    spike = \"\"\"
                        v > treshold
                    \"\"\"

            * *reset*: denotes the equations executed after a spike

                .. code-block:: python
        
                    reset = \"\"\"
                        u = u + d
                        v = c
                    \"\"\"
        """        
        Master2.__init__(self)
        
        self._convert(parameters, equations, extra_values)
        
        spike_eq = self._prepare_string(spike)[0]
        reset_eq = self._prepare_string(reset)
        
        lside, rside = spike_eq.split(':')
        var_name = lside.replace(' ','')
        var_eq = rside.replace(' ','')
        
        old_var = self._variables[var_name]['var']
        new_var = SpikeVariable(
                            init = old_var.init,
                            eq = old_var.eq,
                            min = old_var.min,
                            max = old_var.max,
                            threshold = rside,
                            reset = reset_eq
                        )
        
        self._variables[var_name]['var'] = new_var 
        
    def __str__(self):
        return pprint.pformat( self._variables, depth=4 )
