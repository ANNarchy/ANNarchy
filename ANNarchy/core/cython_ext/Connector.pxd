# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool

cdef class CSR:
    """
    Container for the ranks, weights and values of a projection.
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

    # Insert methods
    cdef push_back(self, int rk, vector[int] r, vector[double] w, vector[int] d)

    # Access methods
    cpdef int get_max_delay(self)
