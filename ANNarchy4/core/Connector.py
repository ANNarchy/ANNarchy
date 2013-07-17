from Random import *

#
# simple data storage
class Connector:
    def __init__(self, type, weights, delays=RandomDistribution('constant', [0])):
        self.weights = weights
        self.delays = delays
        self.type = type        

    def genCPPCall(self):
        if self.type == 'All2All':
            bool = 'true' #if self.allowSelfConnections else 'false'
            return '&(All2AllConnector('+ bool +', new ' + self.weights.genCPP() +'))'

        elif self.type == 'One2One':
            return '&(One2OneConnector(new ' + self.weights.genCPP() +'))'

        else:
            print 'ERROR: called unregistered connector.'
            return 'UnregisteredConnector'

    def instantiateConnector(self,type):
        import ANNarchyCython
        conn = None

        if self.type == 'All2All':
            conn = ANNarchyCython.All2AllConnector()
        elif self.type == 'One2One':
            conn = ANNarchyCython.One2OneConnector()
        elif self.type == 'DoG':
            conn = ANNarchyCython.DoGConnector()
        else:
            print 'Called unregistered connector.'

        return conn

