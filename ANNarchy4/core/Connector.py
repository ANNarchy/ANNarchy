"""
Connector.py
"""
from Random import RandomDistribution

class Connector(object):
    """
    The connector class manages all information and operations related to connection patterns.
    """

    def __init__(self, conn_type, weights, delays=RandomDistribution('constant', [0]), **parameters):
        """
        Initialize a connection object.

        Parameters:
                
            * *conn_type*: name of connection class (One2One, All2All, DoG, ...)
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.
        """
        self.weights = weights
        self.delays = delays
        self.conn_type = conn_type        
        self.parameters = parameters

    def init_connector(self, proj_type):
        """
        Returns a connector object (instance of python extension class). Will be called after compilation of ANNarchy py extension library.
        
        proj_type   ID of projection class (zero for standard projection)
        """
        import ANNarchyCython
        self.cyInstance = eval('ANNarchyCython.'+self.conn_type+'(proj_type)')

        print 'ANNarchyCython.'+self.conn_type+'(proj_type)'
        
    def cpp_call(self):
        """
        Generate connector initialization in ANNarchy.h. 
        HINT: only used if cpp_stand_alone=True provided to compile()
        """
        if self.conn_type == 'All2All':
            if 'allow_self_connections' in self.parameters.keys():
                bool_value = 'true' if self.parameters['allow_self_connections']==True else 'false'
            else:
                print 'All2All: no value for allow_self_connections. Default = False was set.'
                bool_value = 'false' # raise an error
                
            return '&(All2AllConnector('+ bool_value +', new ' + self.weights.genCPP() +'))'

        elif self.conn_type == 'One2One':
            return '&(One2OneConnector(new ' + self.weights.genCPP() +'))'

        else:
            print 'ERROR: no c++ equivalent registered.'
            return 'UnregisteredConnector'

