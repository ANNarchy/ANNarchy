"""
The OpenMP package does contain all code templates required for the openMP
code generation in ANNarchy.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * LIL: list-in-list
        * BSR: blocked sparse row
        * COO: coordinate
        * CSR: compressed sparse row
        * ELL: ELLPACK/ITPACK
        * ELLR: ELLPACK format with row-length array
        * SELL: sliced ELLPACK format
        * Dense: a full matrix representation

    there are some special purpose implementations:

        * CSR_T: compressed sparse row (transposed)
        * Dense_T: dense (transposed)
        * LIL_P: a partitioned LIL representation
"""
from . import LIL as LIL_OpenMP
from . import LIL_P as LIL_Sliced_OpenMP
from . import COO as COO_OpenMP
from . import DIA as DIA_OpenMP
from . import BSR as BSR_OpenMP
from . import CSR as CSR_OpenMP
from . import CSR_P as CSR_Sliced_OpenMP
from . import CSR_T as CSR_T_OpenMP
from . import CSR_T_P as CSR_T_Sliced_OpenMP
from . import ELL as ELL_OpenMP
from . import ELLR as ELLR_OpenMP
from . import SELL as SELL_OpenMP
from . import Dense as Dense_OpenMP
from . import Dense_T as Dense_T_OpenMP

__all__ = [
    "BaseTemplates",
    "LIL_OpenMP", "LIL_Sliced_OpenMP",
    "BSR_OpenMP",
    "COO_OpenMP",
    "DIA_OpenMP",
    "CSR_OpenMP", "CSR_Sliced_OpenMP", "CSR_T_OpenMP", "CSR_T_Sliced_OpenMP",
    "ELL_OpenMP",
    "ELLR_OpenMP",
    "SELL_OpenMP",
    "Dense_OpenMP", "Dense_T_OpenMP"
]
