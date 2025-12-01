"""
The CUDA package does contain all code templates required for the code generation in ANNarchy 
targeting NVIDIA graphic cards.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * COO:          coordinate
        * BSR:          blocked sparse row
        * CSR:          compressed sparse row
        * CSR_Scalar:   a specialized CSR implementation
        * CSR_Vector:   a specialized CSR implementation
        * ELL:          ELLPACK
        * ELLR:         ELLPACK with some GPU-specific optimizations
        * HYB:          a hybrid format using ELLPACK and Coordinate
        * Dense:        a full matrix representation

    there are some special purpose implementations:

        * CSR_T:        csrc (transposed)
"""
from . import BaseTemplates

from . import COO as COO_CUDA
from . import BSR as BSR_CUDA
from . import CSR as CSR_CUDA
from . import CSR_T as CSR_T_CUDA
from . import CSR_Scalar as CSR_SCALAR_CUDA
from . import CSR_Vector as CSR_VECTOR_CUDA
from . import ELL as ELL_CUDA
from . import ELLR as ELLR_CUDA
from . import SELL as SELL_CUDA
from . import HYB as HYB_CUDA
from . import Dense as Dense_CUDA
from . import Dense_T as Dense_T_CUDA

__all__ = ["BaseTemplates", "COO_CUDA", "BSR_CUDA", "CSR_CUDA", "CSR_T_CUDA", "CSR_SCALAR_CUDA", "CSR_VECTOR_CUDA", "ELL_CUDA", "ELLR_CUDA", "SELL_CUDA", "HYB_CUDA", "Dense_CUDA", "Dense_T_CUDA"]