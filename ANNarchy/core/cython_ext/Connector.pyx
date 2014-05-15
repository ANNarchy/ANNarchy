# distutils: language = c++

from libcpp.vector cimport vector

cdef class CSR:

    def __init__(self):
        self.post = {}

    cpdef add (self, int rk, vector[int] r, vector[float] w, vector[int] d):
        cdef list val
        val = []
        val.append(r)
        val.append(w)
        val.append(d)
        self.post[rk] = val

    def keys(self):
        return self.post.keys()

    def data(self):
        return self.post

def all_to_all(int pre_size, int post_size, float weights, int delays):

    cdef CSR synapses
    cdef int post, pre
    cdef list tmp
    cdef vector[int] r, d
    cdef vector[float] w

    synapses = CSR()

    for post in xrange(post_size):
        tmp = [i for i in xrange(pre_size)]
        r = tmp
        w = vector[float](pre_size, weights)
        d = vector[int](pre_size, delays)
        synapses.add(post, r, w, d)

    return synapses

