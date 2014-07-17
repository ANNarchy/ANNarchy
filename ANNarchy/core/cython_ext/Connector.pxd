# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

cdef class CSR:
    """
    Container for the ranks, weights and values of a projection.
    """
    # Data
    cpdef list post_ranks
    cdef map[int, vector[int]] ranks
    cdef map[int, vector[double]] weights
    cdef map[int, vector[int]] delay

    # Attributes
    cpdef int max_delay
    cpdef float dt
    cdef public int size, nb_synapses

    # Insert methods
    cdef push_back (self, int rk, vector[int] r, vector[double] w, vector[int] d)

    # Access methods
    cpdef set_delay(self, int rk, vector[int] d)
    cpdef get_delay(self)
    cpdef get_data(self)
    cpdef get_max_delay(self)
