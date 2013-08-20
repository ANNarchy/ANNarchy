from ANNarchy4.core.Random import *
from libcpp.vector cimport vector
from libc.stdlib cimport malloc
cimport numpy as np

#
# c++ class
cdef extern from "../build/Projection.h":
        cdef cppclass Projection:
                Projection(int preLayer, int postLayer, int postNeuronRank, int target)

                vector[int] getRank()

                vector[float] getValue()

                void initValues(vector[int] rank, vector[float] value)
                
                int getSynapseCount()

#
# c++ class
cdef extern from "../build/ANNarchy.h":
        cdef cppclass createProjInstance:
                createProjInstance()
                
                Projection* getInstanceOf(int id, int pre, int post, int postNeuronRank, int target)
                
#
# wrapper to c++ class, contains connection data of one neuron
cdef class LocalProjection:

        cdef Projection* cInstance

        def __cinit__(self, proj_type, preID, postID, rank, target):
            self.cInstance = createProjInstance().getInstanceOf(proj_type, preID, postID, rank, target)

        def init(self, ranks, values):
            self.cInstance.initValues(ranks, values)

        def size(self):
            return self.cInstance.getSynapseCount()
            
        property value:
            def __get__(self):
                return np.array(self.cInstance.getValue())

            def __set__(self, value):
                print 'currently not implemented'

        property rank:
            def __get__(self):
                return np.array(self.cInstance.getRank())
                
#
# bundles all connections of all neurons within population
cdef class pyProjection:
        cdef local
        cdef int target
        cdef pre
        cdef post
        
        def __init__(self, proj_type, pre, post, nbNeurons, target):
            self.local = []
            self.target= target
            self.pre = pre
            self.post = post

            for n in xrange(nbNeurons):
                self.local.append(LocalProjection(proj_type, pre.id, post.id, n, target))
            
        def values_all(self):
            m_ges = None
            i = 0
            
            #
            # order all matrices in same geometry as postsynaptic population            
            for y in xrange(self.post.height()):
                m_row = None
                
                for x in xrange(self.post.width()):
                    m = np.zeros(self.pre.width()*self.pre.height())
                    if i < len(self.local):
                        m[self.local[i].rank[:]] = self.local[i].value[:]
                       
                    # same geometry as pre layer
                    if m_row == None:
                        m_row = m.reshape(self.pre.width(),self.pre.height())
                    else:
                        m_row = np.ma.concatenate([m_row, m.reshape(self.pre.width(),self.pre.height())])    
                    
                    # next neuron
                    i+=1
                
                if m_ges == None:
                    m_ges = m_row
                else: 
                    m_ges = np.ma.concatenate((m_ges, m_row), axis=1)
             
            return m_ges.T

        def values(self, ranks):
            m_ges = None
            i = 0
            
            #
            # order all matrices in same geometry as postsynaptic population            
            for y in xrange(self.post.height()):
                m_row = None
                
                for x in xrange(self.post.width()):
                    m = np.zeros(self.pre.width()*self.pre.height())
                    if i < len(self.local):
                        m[self.local[i].rank[:]] = self.local[i].value[:]
                       
                    # same geometry as pre layer
                    if m_row == None:
                        m_row = m.reshape(self.pre.width(),self.pre.height())
                    else:
                        m_row = np.ma.concatenate([m_row, m.reshape(self.pre.width(),self.pre.height())])    
                    
                    # next neuron
                    i+=1
                
                if m_ges == None:
                    m_ges = m_row
                else: 
                    m_ges = np.ma.concatenate((m_ges, m_row), axis=1)
             
            return m_ges.T
            
        def size(self, id):
            return self.local[id].size()
        
        def init_local(self, id, r, v):
            self.local[id].init(r,v)
            
        def get_local(self, id):
            return self.local[id]

