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
            self.cInstance.setValue(value)

    property rank:
        def __get__(self):
            return np.array(self.cInstance.getRank())

        def __set__(self, rank):
            self.cInstance.setRank(rank)

