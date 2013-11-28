"""
    
    Connector.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""
from Random import Constant
from Dendrite import Dendrite

class Connector(object):
    """
    The connector class manages all information and operations related to connection patterns.
    """

    def __init__(self, conn_type, weights, delays=0, **parameters):
        """
        Initialize a connection object.

        Parameters:
                
            * *conn_type*: name of connection class (One2One, All2All, DoG, ...)
            * *weights*: synaptic weights for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of weight values.
            * *delays*: synaptic delay for all synapses of one projection. Could be either a RandomDistribution object or an array with the corresponding amount of delay values.
            * *parameters*: any key-value pairs, except the previous ones, given to this function are interpreted as parameters for the connection pattern.
        """
        if isinstance(weights, float) or isinstance(weights, int):
            self.weights = Constant( float(weights) )
        else:
            self.weights = weights
        
        if isinstance(delays, float) or isinstance(delays, int):
            self.delays = Constant( int(delays) )
        else:
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
        
    def connect(self):
        self.init_connector(self.proj.generator.proj_class['ID'])
                      
        tmp = self.cyInstance.connect(self.proj.pre,
                                          self.proj.post,
                                          self.proj.post.generator.targets.index(self.proj.target),
                                          self.weights,
                                          self.delays,
                                          self.parameters
                                          )
        dendrites = []
        post_ranks = []
        for i in xrange(len(tmp)):
            dendrites.append(Dendrite(self.proj, tmp[i].post_rank, tmp[i]))
            post_ranks.append(tmp[i].post_rank)

        return dendrites, post_ranks
        
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

