"""
Projection class.
"""

import numpy as np

import Global
from Variable import Descriptor, Attribute

from ANNarchy4 import generator



class Projection(object):
    """
    Definition of a projection.
    """
    def __init__(self, pre, post, target, connector, synapse=None):
        """
        * pre: pre synaptic layer (either string or object)
        * post: post synaptic layer (either string or object)
        * target: connection type
        * connector: connection pattern
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
            
    def get_local(self, id):
        return self.local_proj[id]
        
    def gather_data(self, variable):
        blank_col=np.zeros((self.pre.height(), 1))
        blank_row=np.zeros((1,self.post.width()*self.pre.width()+self.post.width() +1))
        m_ges = None
        i=0
        
        for y in xrange(self.post.height()):
            m_row = None
            
            for x in xrange(self.post.width()):
                m = np.zeros(self.pre.width() * self.pre.height())
                if i < len(self.local_proj):
                    m[self.local_proj[i].rank[:]] = self.local_proj[i].value[:]

                # apply pre layer geometry
                if m_row == None:
                    m_row = np.ma.concatenate( [ blank_col, m.reshape(self.pre.height(), self.pre.width()) ], axis = 1 )
                else:
                    m_row = np.ma.concatenate( [ m_row, m.reshape(self.pre.height(), self.pre.width()) ], axis = 1 )
                m_row = np.ma.concatenate( [ m_row , blank_col], axis = 1 )
                
                i += 1
            
            if m_ges == None:
                m_ges = np.ma.concatenate( [ blank_row, m_row ] )
            else:
                m_ges = np.ma.concatenate( [ m_ges, m_row ] )
            m_ges = np.ma.concatenate( [ m_ges, blank_row ] )
        
        return m_ges
        
class LocalProjection(Descriptor):

    def __init__(self, cyInstance, proj):
        self.cyInstance = cyInstance
        self.proj = proj

        #
        # minimum set at par/var
        self.value = Attribute('value')
        self.rank = Attribute('rank')
        self.delay = Attribute('delay')
        
        #pre_def = ['psp', 'value', 'rank', 'delay']
        #for var in self.proj.generator.parsed_synapse_variables:
        #    if var['name'] in pre_def:
        #        continue
             
        #    if '_rand_' in var['name']:
        #        continue
                
        #    cmd = 'proj.'+var['name']+' = Attribute(\''+var['name']+'\')'
            #print cmd
        #    exec(cmd)

