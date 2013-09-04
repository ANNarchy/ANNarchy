# cython: embedsignature=True

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

        void setRank(vector[int] rank)

        vector[int] getDelay()

        void setDelay(vector[int] rank)

        vector[float] getValue()

        void setValue(vector[float] value)
        
        float getDt()
        
        float getTau()
        
        void setTau(float tau)
        
        void initValues(vector[int] rank, vector[float] value)
        
        int getSynapseCount()
        
        int addSynapse(int rank, float value, int delay)

        int removeSynapse(int rank)
        
        int getTarget() 
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
    cdef post_rank
    
    def __cinit__(self, proj_type, preID, postID, rank, target):
        self.post_rank = rank
        self.cInstance = createProjInstance().getInstanceOf(proj_type, preID, postID, rank, target)
        
    def init(self, ranks, values):
        self.cInstance.initValues(ranks, values)

    def add_synapse(self, rank, value, delay=0):
        err = self.cInstance.addSynapse(rank, value, delay)
        if err == -1:
            print 'Synapse already exist.'
    
    def remove_synapse(self, rank):
        err = self.cInstance.removeSynapse(rank)
        if err == -1:
            print 'Synapse not exist.'
        
    def get_target(self):
        return self.cInstance.getTarget()
    
    property size:
        def __get__(self):
            return self.cInstance.getSynapseCount()
        def __set__(self, value):
            print 'The dendrite size is a read-only value.'
    
    property post_rank:
        """
        Returns the rank of the neuron the dendrite belongs to.
        """
        def __get__(self):
            return self.post_rank
        def __set__(self, value):
            print 'The post_rank is a read-only value.'
        
    property dt:
        def __get__(self):
            return self.cInstance.getDt()

        def __set__(self, value):
            print 'The discretization step is only modifiable globally.'

    property tau:
        def __get__(self):
            return self.cInstance.getTau()

        def __set__(self, value):
            self.cInstance.setTau(value)
        
    property value:
        def __get__(self):
            return np.array(self.cInstance.getValue())

        def __set__(self, value):
            if isinstance(value, np.ndarray)==True:
                if value.ndim==1:
                    self.cInstance.setValue(value)
                else:
                    self.cInstance.setValue(value.reshape(self.size))
            else:
                self.cInstance.setValue(np.ones(self.size)*value)

    property rank: # pre synaptic rank
        def __get__(self):
            return np.array(self.cInstance.getRank())

        def __set__(self, rank):
            self.cInstance.setRank(rank)
