# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool

cdef class CSR:
    """
    Container for the ranks, weights and delays of a projection.
    """
    # Data
    cdef public vector[int] post_rank
    cdef public vector[vector[int]] pre_rank
    cdef public vector[vector[double]] w
    cdef public vector[vector[int]] delay

    # Attributes
    cpdef public int max_delay
    cpdef public int uniform_delay
    cdef public int size
    cdef public int nb_synapses
    cdef public double dt

    # Insert methods
    cpdef add(self, int rk, r, w, d)
    cpdef push_back(self, int rk, vector[int] r, vector[double] w, vector[double] d)

    # Access methods
    cpdef int get_max_delay(self)
    cpdef int get_uniform_delay(self)

    # Method to clean a CSR object
    cpdef validate(self)
