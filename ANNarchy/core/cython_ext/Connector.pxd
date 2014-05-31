# distutils: language = c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

cdef class CSR:
    cpdef dict data
    cdef push_back (self, int rk, vector[int] r, vector[float] w, vector[int] d)
    cpdef get_data(self)

