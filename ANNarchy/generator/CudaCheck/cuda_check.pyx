# cython: embedsignature=True

cdef extern from "cuda_check.h":

    int get_major_version(int)

    int get_minor_version(int)

def get_cuda_version():
    major = get_major_version(0)
    minor = get_minor_version(0)
    return (major, minor)
