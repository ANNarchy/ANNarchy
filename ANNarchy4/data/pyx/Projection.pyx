from ANNarchy4.core.Random import *
from libcpp.vector cimport vector
from libc.stdlib cimport malloc

#
# c++ class
cdef extern from "../build/Projection.h":
        cdef cppclass Projection:
                Projection(int preLayer, int postLayer, int postNeuronRank)

                vector[int] getRank()

                vector[float] getValue()

                void initValues(vector[int] rank, vector[float] value)

#
# wrapper to c++ class, contains connection data of one neuron
cdef class LocalProjection:

        cdef Projection* cInstance

        def __cinit__(self, preID, postID, rank):
             self.cInstance = new Projection(preID, postID, rank)

        def init(self, ranks, values):
             self.cInstance.initValues(ranks, values)

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

        def __init__(self, pre, post, nbNeurons):
            self.local = []

            for n in xrange(nbNeurons):
                self.local.append(LocalProjection(pre, post, n))

        def getLocal(self, id):
            return self.local[id]

