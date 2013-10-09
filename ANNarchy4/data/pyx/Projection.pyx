# cython: embedsignature=True

"""

    Projection.pyx
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Ülo Dinkelbach <helge.dinkelbach@gmail.com>

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
        
        void initValues(vector[int] rank, vector[float] value, vector[int] delay)
        
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
        
    def init(self, ranks, values, delays):
        self.cInstance.initValues(ranks, values, delays)

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

    property delay:
        def __get__(self):
            return np.array(self.cInstance.getDelay())

        def __set__(self, value):
            if isinstance(value, np.ndarray)==True:
                if value.ndim==1:
                    self.cInstance.setDelay(value)
                else:
                    self.cInstance.setDelay(value.reshape(self.size))
            else:
                self.cInstance.setDelay(np.ones(self.size)*value)

    property rank: # pre synaptic rank
        def __get__(self):
            return np.array(self.cInstance.getRank())

        def __set__(self, rank):
            self.cInstance.setRank(rank)
