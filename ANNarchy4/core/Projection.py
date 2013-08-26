"""
Projection.py
"""

import numpy as np
import traceback

import Global
from Dendrite import Dendrite

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
        self._dendrites = []
        Global.projections_.append(self)
                
    def dendrite_by_coordinates(self, w, h, d=0):
        """
        Returns the local projection of a postsynaptic neuron according to its coordinate
        """
        return self._dendrites[self.post.rank_from_coordinates(w, h, d)]
            
    def dendrite_by_rank(self, rank):
        """
        Returns the local projection of a postsynaptic neuron according to its rank
        """
        return self._dendrites[rank]

    @property
    def dendrites(self):
        """
        List of all local projections.
        """
        return self._dendrites

    def connect(self):
        self.connector.init_connector(self.generator.proj_class['ID'])          
        tmp = self.connector.cyInstance.connect(self.pre,
                                          self.post,
                                          self.connector.weights,
                                          self.post.generator.targets.index(self.target),
                                          self.connector.parameters
                                          )
        
        for i in xrange(len(tmp)):
            self._dendrites.append(Dendrite(tmp[i], self))

    def gather_data(self, variable):
        blank_col=np.zeros((self.pre.height, 1))
        blank_row=np.zeros((1,self.post.width*self.pre.width+self.post.width +1))
        
        m_ges = None
        i=0
        
        for y in xrange(self.post.height):
            m_row = None
            
            for x in xrange(self.post.width):
                m = self._dendrites[i].get_variable(variable)
                
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
