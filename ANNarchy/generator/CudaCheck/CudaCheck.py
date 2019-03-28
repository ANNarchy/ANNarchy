import os, imp
import ANNarchy.core.Global as Global
from ANNarchy import __path__ as ann_path

class CudaCheck(object):
    """
    A simple module for handling device parameter checking, needed for code generation
    """
    def __init__(self):
        """
        Initialization.

        The constructor tries to load the cuda_check module which wraps the CudaDeviceProperties
        with some accessor methods.

        Hint:

        The first access to any of the following functions could take a moment as the CUDA interface
        is then initialized.
        """
        fp, pathname, description = imp.find_module("cuda_check", [ann_path[0]+"/generator/CudaCheck/"])
        self.cy_cc = imp.load_module("cuda_check", fp, pathname, description)

    def gpu_count(self):
        try:
            result = self.cy_cc.gpu_count()
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return result

    def version(self):
        """
        Returns cuda compatibility as tuple(major,minor)
        """
        try:
            result = self.cy_cc.get_cuda_version()
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return result

    def version_str(self):
        """
        Returns cuda compatibility as string, usable for -gencode as argument.
        """
        try:
            cu_version = self.cy_cc.get_cuda_version()
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return str(cu_version[0])+str(cu_version[1])

    def runtime_version(self):
        try:
            result = self.cy_cc.runtime_version()
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return result

    def max_threads_per_block(self, device=0):
        try:
            result = self.cy_cc.max_threads_per_block(device)
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return result

    def warp_size(self, device=0):
        try:
            result = self.cy_cc.warp_size(device)
        except Exception as e:
            Global._print(e)
            Global._error('CUDA is not correctly installed on your system.')
        return result

