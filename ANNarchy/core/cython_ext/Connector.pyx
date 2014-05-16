# distutils: language = c++

from libcpp.vector cimport vector
import random

import ANNarchy
from ANNarchy.core.Random import RandomDistribution

cdef class CSR:

    def __init__(self):
        self.data = {}

    cdef add (self, int rk, vector[int] r, vector[float] w, vector[int] d):
        cdef list val
        val = []
        val.append(r)
        val.append(w)
        val.append(d)
        self.data[rk] = val

    def keys(self):
        return self.data.keys()

    cpdef get_data(self):
        return self.data

def all_to_all(int pre_size, int post_size, weights, delays, allow_self_connections):
    """ Cython implementation of the all-to-all pattern."""

    cdef CSR synapses
    cdef float dt
    cdef int post, pre, size_pre
    cdef list tmp
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Create the projection data as CSR
    projection = CSR()

    # Determine the size of the the pre-field 
    # (the passed argument allow_self_connections already knows if it is the same population or not)
    if allow_self_connections:
        size_pre = pre_size
    else:
        size_pre = pre_size -1 

    for post in xrange(post_size):
        # List of pre ranks
        tmp = [i for i in xrange(pre_size)]
        if not allow_self_connections:
            tmp.remove(post)
        r = tmp
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](size_pre, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_pre)
        # Delays
        if isinstance(delays, float):
            d = vector[int](size_pre, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](size_pre, delays)
        elif isinstance(weights, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(size_pre) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection



def fixed_probability(int pre_size, int post_size, float probability, weights, delays, allow_self_connections):
    """ Cython implementation of the all-to-all pattern."""

    cdef CSR synapses
    cdef float dt
    cdef int post, pre, size_pre
    cdef list tmp
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Create the projection data as CSR
    projection = CSR()

    for post in xrange(post_size):
        # List of pre ranks
        tmp = []
        for i in xrange(pre_size):
            if not allow_self_connections and (i==post):
                continue
            if random.random() < probability:
                tmp .append(i) 
        r = tmp
        size_pre = len(tmp)
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](size_pre, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_pre)
        # Delays
        if isinstance(delays, float):
            d = vector[int](size_pre, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](size_pre, delays)
        elif isinstance(weights, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(size_pre) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection