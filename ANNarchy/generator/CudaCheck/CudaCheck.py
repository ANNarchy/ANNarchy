import os
import ANNarchy.core.Global as Global

class CudaCheck(object):
    """
    A simple module for handling device parameter checking, needed for code generation
    """
    def __init__(self):
        """
        Initialization stuff
        """
        pass

    def gpu_count(self):
        try:
            from .cuda_check import gpu_count
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return gpu_count()

    def version(self):
        """
        Returns cuda compatibility as tuple(major,minor)
        """
        try:
            from .cuda_check import get_cuda_version
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return get_cuda_version()

    def version_str(self):
        """
        Returns cuda compatibility as string, usable for -gencode as argument.
        """
        try:
            from .cuda_check import get_cuda_version
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        cu_version = get_cuda_version()
        return str(cu_version[0])+str(cu_version[1])

    def runtime_version(self):
        try:
            from .cuda_check import runtime_version
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return runtime_version()

    def max_threads_per_block(self, device=0):
        try:
            from .cuda_check import max_threads_per_block
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return max_threads_per_block(device)

    def warp_size(self, device=0):
        try:
            from .cuda_check import warp_size
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return warp_size(device)
