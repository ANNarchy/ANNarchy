"""
Projection.py
"""
import numpy as np

import Global
from Dendrite import Dendrite
from ANNarchy4 import generator
from ANNarchy4.core.Random import RandomDistribution

class Projection(object):
    """
    Python class representing the projection between two populations.
    """

    def __init__(self, pre, post, target, connector, synapse=None):
        """
        Constructor of a Projection object.

        Parameters:
                
            * *pre*: pre synaptic layer (either name or Population object)
            * *post*: post synaptic layer (either name or Population object)
            * *target*: connection type
            * *connector*: connection pattern object
            * *synapse*: synapse object
        """
        
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object 
        if isinstance(pre, str):
            for pop in Global._populations:
                if pop.name == pre:
                    self.pre = pop
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for pop in Global._populations:
                if pop.name == post:
                    self.post = pop
        else:
            self.post = post
            
        self.post.generator.add_target(target)
        self.target = target
        self.connector = connector
        self.synapse = synapse
        self._dendrites = []
        self._post_ranks = []

        self.name = 'Projection'+str(len(Global._projections))
        self.generator = generator.Projection(self, self.synapse)
        Global._projections.append(self)
        
                
    def dendrite(self, pos):
        """
        Returns the dendrite of a postsynaptic neuron according to its rank.

        Parameters:

            * *pos*: could be either rank or coordinate of the requested postsynaptic neuron
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.post.rank_from_coordinates(pos)
        if rank in self._post_ranks:
            return self._dendrites[rank]
        else:
            print 'Error: neuron of rank', str(rank), 'has no synapse in this projection.'
            return None
    
    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns dendrite of the given position in the postsynaptic population. 
        
        If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        return self.dendrite(args[0])
        
    def __iter__(self):
        " Returns iteratively each dendrite in the population in ascending rank order."
        for n in range(self.size):
            yield self._dendrites(n)  
        
    @property
    def size(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return len(self._dendrites)
        
    def __len__(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return len(self._dendrites)
        
    @property
    def dendrites(self):
        """
        List of dendrites corresponding to this projection.
        """
        return self._dendrites

    def _parsed_variables(self):
        """
        Returns parsed variables in case of an attached synapse.
        """
        if self.synapse:
            return self.generator.parsed_variables
        else:
            return []

    def connect(self):
        
        self.connector.init_connector(self.generator.proj_class['ID'])
                      
        tmp = self.connector.cyInstance.connect(self.pre,
                                          self.post,
                                          self.connector.weights,
                                          self.post.generator.targets.index(self.target),
                                          self.connector.parameters
                                          )
        
        for i in xrange(len(tmp)):
            self._dendrites.append(Dendrite(self, tmp[i].post_rank, tmp[i]))
            self._post_ranks.append(tmp[i].post_rank)

    def gather_data(self, variable):
        blank_col=np.zeros((self.pre.geometry[1], 1))
        blank_row=np.zeros((1,self.post.geometry[0]*self.pre.geometry[0]+self.post.geometry[0] +1))
        
        m_ges = None
        i=0
        
        for y in xrange(self.post.geometry[1]):
            m_row = None
            
            for x in xrange(self.post.geometry[0]):
                m = self._dendrites[i].get_variable(variable)
                
                if m_row == None:
                    m_row = np.ma.concatenate( [ blank_col, m.reshape(self.pre.geometry[1], self.pre.geometry[0]) ], axis = 1 )
                else:
                    m_row = np.ma.concatenate( [ m_row, m.reshape(self.pre.geometry[1], self.pre.geometry[0]) ], axis = 1 )
                m_row = np.ma.concatenate( [ m_row , blank_col], axis = 1 )
                
                i += 1
            
            if m_ges == None:
                m_ges = np.ma.concatenate( [ blank_row, m_row ] )
            else:
                m_ges = np.ma.concatenate( [ m_ges, m_row ] )
            m_ges = np.ma.concatenate( [ m_ges, blank_row ] )
        
        return m_ges
