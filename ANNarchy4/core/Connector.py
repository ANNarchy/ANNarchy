"""
Connector.py
"""
from Random import RandomDistribution

class Connector:
    """
    Manage all information and operations related to connection patterns.
    """
    def __init__(self, conn_type, weights, delays=RandomDistribution('constant', [0]), **keyValueArgs):
        self.weights = weights
        self.delays = delays
        self.conn_type = conn_type        
        self.parameters = keyValueArgs

    def cpp_call(self):
        """
        Generate connector initializatuion in ANNarchy.h. 
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

    def init_connector(self, proj_type):
        """
        * Returns a connector object (instance of python extension class). 
        *
        *   proj_type   ID of projection class (zero for standard projection)
        """
        import ANNarchyCython
        conn = None

        if self.conn_type == 'All2All':
            conn = ANNarchyCython.All2AllConnector(proj_type)
        elif self.conn_type == 'One2One':
            conn = ANNarchyCython.One2OneConnector(proj_type)
        elif self.conn_type == 'DoG':
            conn = ANNarchyCython.DoGConnector(proj_type)
        else:
            print 'Called unregistered connector.'

        return conn

