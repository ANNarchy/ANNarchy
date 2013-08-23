"""
Projection.py
"""

import numpy as np
import traceback

import Global
from Variable import Descriptor, Attribute

from ANNarchy4 import generator

class Projection(object):
    """
    Python class representing the projection between two populations.
    """

    def __init__(self, pre, post, target, connector, synapse=None):
        """
        Constructor of a Projection object.

        Parameters:
                
        * pre: pre synaptic layer (either name or Population object)
        * post: post synaptic layer (either name or Population object)
        * target: connection type
        * connector: connection pattern object
        * synapse: synapse object
        """
        
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object 
        if isinstance(pre, str):
            for pop in Global.populations_:
                if pop.name == pre:
                    self.pre = pop
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for pop in Global.populations_:
                if pop.name == post:
                    self.post = pop
        else:
            self.post = post
            
        self.post.generator.add_target(target)
        self.target = target
        self.connector = connector
        self.synapse = synapse
        self.generator = generator.Projection(self, synapse)
        Global.projections_.append(self)
        
    def local_projection_by_coordinates(self, w, h, d=0):
        """
        Returns the local projection of a postsynaptic neuron according to its coordinate
        """
        return self.local_proj[self.post.rank_from_coordinates(w, h, d)]
            
    def local_projection_by_rank(self, rank):
        """
        Returns the local projection of a postsynaptic neuron according to its rank
        """
        return self.local_proj[rank]

    @property
    def local_projections(self):
        """
        List of all local projections.
        """
        return self.local_proj        

    def connect(self):
        self.connector.init_connector(self.generator.proj_class['ID'])          
        tmp = self.connector.cyInstance.connect(self.pre,
                                          self.post,
                                          self.connector.weights,
                                          self.post.generator.targets.index(self.target),
                                          self.connector.parameters
                                          )
        
        self.local_proj = []
        for i in xrange(len(tmp)):
            self.local_proj.append(LocalProjection(tmp[i], self))

    def gather_data(self, variable):
        blank_col=np.zeros((self.pre.height, 1))
        blank_row=np.zeros((1,self.post.width*self.pre.width+self.post.width +1))
        m_ges = None
        i=0
        
        for y in xrange(self.post.height):
            m_row = None
            
            for x in xrange(self.post.width):
                m = self.local_proj[i].get_variable(variable)
                
                if m_row == None:
                    m_row = np.ma.concatenate( [ blank_col, m.reshape(self.pre.height, self.pre.width) ], axis = 1 )
                else:
                    m_row = np.ma.concatenate( [ m_row, m.reshape(self.pre.height, self.pre.width) ], axis = 1 )
                m_row = np.ma.concatenate( [ m_row , blank_col], axis = 1 )
                
                i += 1
            
            if m_ges == None:
                m_ges = np.ma.concatenate( [ blank_row, m_row ] )
            else:
                m_ges = np.ma.concatenate( [ m_ges, m_row ] )
            m_ges = np.ma.concatenate( [ m_ges, blank_row ] )
        
        return m_ges
        
class LocalProjection(Descriptor):
    """
    A local projection encapsulates all synapses at neuron level.
    """
    def __init__(self, cyInstance, proj):
        self.cyInstance = cyInstance
        self.proj = proj
        self.pre = proj.pre
        
        #
        # base variables
        self.value = Attribute('value')
        self.rank = Attribute('rank')
        self.delay = Attribute('delay')
    
    @property
    def size(self):
        """
        Number of synapses.
        """
        return self.cyInstance.get_size()
        
    @property
    def target(self):
        """
        Connection type id.
        """
        return self.cyInstance.get_target()
        
    def get_variable(self, variable):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameters:
        
        * variable:    a string representing the variable's name.
        """
        if hasattr(self, variable):
            var = eval('self.'+variable)

            m = np.zeros(self.pre.width * self.pre.height)
            m[self.rank[:]] = var[:]

            return self._reshape_vector(var)
        else:
            print 'Error: variable',variable,'does not exist in this projection.'
            print traceback.print_stack()

    def get_parameter(self, parameter):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameters:
        
        * parameter:    a string representing the parameter's name.
        """
        if hasattr(self, parameter):
            return eval('self.'+parameter)
        else:
            print 'Error: parameter',parameter,'does not exist in this projection.'
            print traceback.print_stack()

    def add_synapse(self, rank, value, delay=0):
        """
        Adds a synapse to the local projection.
        
        Parameters:
        
        * rank:     rank of the presynaptic neuron
        * value:    synaptic weight
        * delay:    delay of the synapse
        """
        self.cyInstance.add_synapse(rank, value, delay)
    
    def remove_synapse(self, rank):
        """
        Removes the synapse.
        
        Parameters:
        
        * rank:     rank of the presynaptic neuron
        """
        self.cyInstance.remove_synapse(rank)

    def _reshape_vector(self, vector):
        """
        Transfers a list or a 1D np.array (indiced with ranks) into the correct 1D, 2D, 3D np.array
        """
        vec = np.array(vector) # if list transform to vec
        try:
            if self.pre.dimension == 1:
                return vec
            elif self.pre.dimension == 2:
                return vec.reshape(self.pre.height, self.pre.width)
            elif self.pre.dimension == 3:
                return vec.reshape(self.pre.depth, self.pre.height, self.pre.width)
        except ValueError:
            print 'Mismatch between pop: (',self.pre.width,',',self.pre.height,',',self.pre.depth,')'
            print 'and data vector (',type(vector),': (',vec.size,')'
            print traceback.print_stack()            
