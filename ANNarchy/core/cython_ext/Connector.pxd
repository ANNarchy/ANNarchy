# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool
from sympy.mpmath.matrices.matrices import _matrix

cdef class LIL:
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

    # Method to clean a LIL object
    cpdef validate(self)

cdef extern from "CSRMatrix.hpp":
    cdef cppclass CSRMatrix:
        CSRMatrix(const unsigned int, const unsigned int)

        void push_back(int, vector[int], vector[double], vector[int])
        vector[int] row_begin()
        vector[int] column_indices()
        vector[double] values()

        int num_elements()

cdef class CSR:
    """
    Container for the ranks, weights and delays of a projection.
    """
    cdef CSRMatrix* _matrix

    # Insert methods
    cpdef add(self, int pre_rank, post_rank, w, d)
    cpdef push_back(self, int pre_rank, vector[int] post_ranks, vector[double] w, vector[double] d)
    
    # pre-defined pattern
    cpdef all_to_all(self, pre, post, weights, delays, allow_self_connections)