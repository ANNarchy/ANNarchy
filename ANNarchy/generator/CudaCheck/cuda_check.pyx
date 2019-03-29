# distutils: language = c++
# cython: embedsignature=True
# cython: language_level=2

cdef extern from "cuda_check.h":

    int get_major_version(int)
    int get_minor_version(int)
    int get_runtime_version()
    int get_max_threads_per_block(int)
    int get_warp_size(int)
    int num_devices()

def gpu_count():
    return num_devices()

def get_cuda_version():
    major = get_major_version(0)
    minor = get_minor_version(0)
    return (major, minor)

def runtime_version():
    return get_runtime_version()

def max_threads_per_block(device):
    return get_max_threads_per_block(device)

def warp_size(device):
    return get_warp_size(device)
