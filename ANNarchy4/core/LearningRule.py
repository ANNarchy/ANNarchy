from Master import Master

class LearningRule(Master):
    """
    Definition of an ANNarchy learning rule.
    """
        
    def __init__(self, debug=False, order=[], **keyValueArgs):
        if debug==True:
            print '\n\tLearningRule class\n'
        
        Master.__init__(self, debug, order, keyValueArgs)