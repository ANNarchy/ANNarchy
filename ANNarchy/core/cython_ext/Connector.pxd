# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool
from sympy.mpmath.matrices.matrices import _matrix

cdef class LILConnectivity:
    """
    Container for the ranks, weights and delays of a projection.
    """
    # Data
    cdef public vector[int] post_rank
    cdef public vector[vector[int]] pre_rank
    cdef public vector[vector[double]] w
    cdef public vector[vector[int]] delay

    # Attributes
    cdef public int max_delay
    cdef public int uniform_delay
    cdef public int size
    cdef public int nb_synapses
    cdef public double dt

    # Insert methods
    cdef add(self, int rk, r, w, d)
    cdef push_back(self, int rk, vector[int] r, vector[double] w, vector[double] d)

    # Access methods
    cdef int get_max_delay(self)
    cdef int get_uniform_delay(self)

    # Method to clean a LIL object
    cdef validate(self)

    # pre-defined pattern
    cdef all_to_all(self, pre, post, weights, delays, allow_self_connections)
    cdef one_to_one(self, pre, post, weights, delays)
    cdef fixed_probability(self, pre, post, probability, weights, delays, allow_self_connections)
    cdef fixed_number_pre(self, pre, post, int number, weights, delays, allow_self_connections)
    cdef fixed_number_post(self, pre, post, int number, weights, delays, allow_self_connections)
    cdef gaussian(self, pre_pop, post_pop, float amp, float sigma, delays, limit, allow_self_connections)
    cdef dog(self, pre_pop, post_pop, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections)
