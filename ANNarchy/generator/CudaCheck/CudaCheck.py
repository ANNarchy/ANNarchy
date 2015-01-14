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

    def version(self):
        """
        
        """
        import cuda_check
        return cuda_check.get_cuda_version()
