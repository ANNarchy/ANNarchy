"""
The SingleThread package does contain all code templates required for the single-thread
code generation in ANNarchy.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * LIL: list-in-list
        * COO: coordinate
        * DIA: diagonal format
        * BSR: blocked compressed row
        * CSR: compressed sparse row
        * ELL: ELLPACK/ITPACK
        * ELLR: ELLPACK with row-length array
        * SELLR: sliced ELLPACK
        * HYB: hybrid format comprising of ELLPACK and coordinate
        * Dense: a full matrix representation

    there are some special purpose implementations:

        * _T suffix: a transposed implementation
        * _PV suffix: a specialized implementation for PopulationViews

"""
from . import LIL as LIL_SingleThread
from . import COO as COO_SingleThread
from . import DIA as DIA_SingleThread
from . import BSR as BSR_SingleThread
from . import CSR as CSR_SingleThread
from . import CSR_T as CSR_T_SingleThread
from . import ELL as ELL_SingleThread
from . import ELLR as ELLR_SingleThread
from . import SELL as SELL_SingleThread
from . import HYB as HYB_SingleThread
from . import Dense as Dense_SingleThread
from . import Dense_T as Dense_T_SingleThread
from . import Dense_PV as Dense_PV_SingleThread

__all__ = [
    "BaseTemplates",
    "LIL_SingleThread",
    "COO_SingleThread",
    "DIA_SingleThread",
    "BSR_SingleThread",
    "CSR_SingleThread", "CSR_T_SingleThread",
    "ELL_SingleThread", "ELLR_SingleThread", "SELL_SingleThread",
    "HYB_SingleThread",
    "Dense_SingleThread", "Dense_PV_SingleThread", "Dense_T_SingleThread"
]
