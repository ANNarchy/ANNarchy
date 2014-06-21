# distutils: language = c++
from libcpp.vector cimport vector

cdef class CSR:
    cpdef dict data
    cpdef dict delay
    cpdef int max_delay
    cpdef float dt
    cdef public int size, nb_synapses
    cdef push_back (self, int rk, vector[int] r, vector[float] w, vector[int] d)
    cpdef set_delay(self, int rk, vector[int] d)
    cpdef get_delay(self)
    cpdef get_data(self)
    cpdef get_max_delay(self)
