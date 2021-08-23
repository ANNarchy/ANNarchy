"""
The OpenMP package does contain all code templates required for the openMP
code generation in ANNarchy.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * LIL: list-in-list
        * COO: coordinate
        * CSR: compressed sparse row
        * ELL: ELLPACK/ITPACK
        * ELL-R: ELLPACK format with row-length array 

    there are some special purpose implementations:

        * CSR_T: compressed sparse row (transposed)
        * LIL_P: a partitioned LIL representation
"""
from . import LIL as LIL_OpenMP
from . import LIL_P as LIL_Sliced_OpenMP
from . import COO as COO_OpenMP
from . import CSR as CSR_OpenMP
from . import CSR_T as CSR_T_OpenMP
from . import CSR_T_P as CSR_T_Sliced_OpenMP
from . import ELL as ELL_OpenMP
from . import ELLR as ELLR_OpenMP

__all__ = ["BaseTemplates", "LIL_OpenMP", "LIL_Sliced_OpenMP", "COO_OpenMP", "CSR_OpenMP", "CSR_T_OpenMP", "CSR_T_Sliced_OpenMP", "ELL_OpenMP", "ELLR_OpenMP"]