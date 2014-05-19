# distutils: language = c++

from libcpp.vector cimport vector

cdef class CSR:
    cpdef dict data
    cdef push_back (self, int rk, vector[int] r, vector[float] w, vector[int] d)
    cpdef get_data(self)

