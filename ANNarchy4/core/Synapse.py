from Master import Master
from ANNarchy4 import parser

class Synapse(Master):
    def __init__(self, debug=False, order=[], **keyValueArgs):
        if debug==True:
            print '\n\tSynapse class\n'

        Master.__init__(self, debug, order, keyValueArgs)

        self.parser = parser.SynapseAnalyser(self.variables)
        self.parsedVariables = self.parser.parse()

        