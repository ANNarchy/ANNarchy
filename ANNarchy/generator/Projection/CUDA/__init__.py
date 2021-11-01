"""
The CUDA package does contain all code templates required for the code generation in ANNarchy 
targeting NVIDIA graphic cards.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * COO:   coordinate
        * BSR:   blocked sparse row
        * CSR:   compressed sparse row
        * ELL:   ELLPACK
        * ELLR:  ELLPACK with some GPU-specific optimizations
        * HYB:   a hybrid format using ELLPACK and Coordinate
        * Dense: a full matrix representation    
"""
from . import COO as COO_CUDA
from . import BSR as BSR_CUDA
from . import CSR as CSR_CUDA
from . import ELL as ELL_CUDA
from . import ELLR as ELLR_CUDA
from . import HYB as HYB_CUDA
from . import Dense as Dense_CUDA

__all__ = ["BaseTemplates", "COO_CUDA", "BSR_CUDA", "CSR_CUDA", "ELL_CUDA", "ELLR_CUDA", "HYB_CUDA", "Dense_CUDA"]