import os
from ANNarchy.core import Global

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
        import cuda_check
        return cuda_check.gpu_count()

    def version(self):
        """
        Returns cuda compatibility as tuple(major,minor)
        """
        import cuda_check
        return cuda_check.get_cuda_version()

    def version_str(self):
        """
        Returns cuda compatibility as string, usable for -gencode as argument.
        """
        import cuda_check
        cu_version = cuda_check.get_cuda_version()
        return str(cu_version[0])+str(cu_version[1])

    def runtime_version(self):
        import cuda_check
        return cuda_check.runtime_version()

    def max_threads_per_block(self, device=0):
        import cuda_check
        return cuda_check.max_threads_per_block(device)

    def warp_size(self, device=0):
        import cuda_check
        return cuda_check.warp_size(device)
