class SpikeVariable(object):
    """
    Variable representation in ANNarchy.
    """
    def __init__(self,  **keyValueArgs):
        """
        Set of key-value pairs defining a spiking variable. These variable 
        only occurs once a neuron.
        
        Possible keys are:
        
            * *eq*: variable equation
            * *threshold*: if the variable exceeds threshold the neuron emits a spike
            * *init*: initial value. After emitting a spike the variable value is set back to this.
            * *min*: minimal value for this variable
            * *max*: maximal value for this variable    
            * *type*: data type, available are: float, int, bool (default is float)        
            
        other keys will be rejected.
        """
        self.eq = None
        self.threshold = None
        self.reset = None
        self.init = None
        self.min = None
        self.max = None
        self.type = None
        
        for key in keyValueArgs:

            if key == 'eq':
                self.eq = keyValueArgs[key]
            elif key=='threshold':
                self.threshold = keyValueArgs[key]
            elif key=='reset':
                self.reset = keyValueArgs[key]
            elif key=='init':
                self.init = keyValueArgs[key]
            elif key=='type':
                self.type = keyValueArgs[key]
            else:
                print('unknown key: '+key)
         
        #
        # for later operations it's not positive if no
        # default is set
        if self.init == None:
            self.init = self.type(0.0)
            
        if type(self.init) != self.type:
            if self.type != bool and type(self.init) != bool:
                self.init = float(self.init)
                self.type = float
         