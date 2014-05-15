# distutils: language = c++

from libcpp.vector cimport vector

cdef class CSR:
    cpdef dict post

    cpdef add (self, int rk, vector[int] r, vector[float] w, vector[int] d)

