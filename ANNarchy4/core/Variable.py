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
         