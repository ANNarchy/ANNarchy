# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp cimport bool

cdef class CSR:
    """
    Container for the ranks, weights and values of a projection.
    """
    # Data
    cdef vector[int] post_ranks
    cdef map[int, vector[int]] ranks
    cdef map[int, vector[double]] weights
    cdef map[int, vector[int]] delays

    # Attributes
    cpdef int max_delay
    cpdef float dt
    cdef public int size, nb_synapses

    # Insert methods
    cdef push_back (self, int rk, vector[int] r, vector[double] w, vector[int] d)

    # Access methods
    cpdef int get_max_delay(self)
    cpdef list get_post_ranks(self)
    cpdef bool uniform_delay(self)
