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
    cdef public bool requires_sorting
    cdef public int last_added_idx

    # Insert methods
    cpdef add(self, int rk, r, w, d)
    cpdef push_back(self, int rk, vector[int] r, vector[double] w, vector[double] d)

    # Access methods
    cpdef int get_max_delay(self)
    cpdef int get_uniform_delay(self)

    # Matrix characteristics (auto-tuning)
    cpdef compute_average_row_length(self)
    cpdef compute_average_col_idx_gap(self)

    # Method to validate a LIL object
    cpdef validate(self)

    # pre-defined pattern
    cpdef all_to_all(self, pre, post, weights, delays, allow_self_connections)
    cpdef one_to_one(self, pre, post, weights, delays)
    cpdef fixed_probability(self, pre, post, probability, weights, delays, allow_self_connections)
    cpdef fixed_number_pre(self, pre, post, int number, weights, delays, allow_self_connections)
    cpdef fixed_number_post(self, pre, post, int number, weights, delays, allow_self_connections)
    cpdef gaussian(self, pre_pop, post_pop, float amp, float sigma, delays, limit, allow_self_connections)
    cpdef dog(self, pre_pop, post_pop, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections)
