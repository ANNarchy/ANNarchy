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
            elif key=='min':
                self.min = keyValueArgs[key]
            elif key=='max':
                self.max = keyValueArgs[key]
            else:
                print 'unknown key: '+key
                  
    #
    # some customization stuff, maybe needed later.
    def __str__(self):
        itemDir = self.__dict__
        str = '['
        for i in itemDir:
            str += '{0} : {1}, '.format(i, itemDir[i])
        str+= ']'

        return str
     
    def __repr__(self):
        return "<%s instance at %li> %s" % (self.__class__.__name__, id(self), self.__str__())  
