"""
The SingleThread package does contain all code templates required for the single-thread
code generation in ANNarchy.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * LIL: list-in-list
        * COO: coordinate
        * CSR: compressed sparse row
        * ELL: ELLPACK/ITPACK
        * ELL-R: ELLPACK with row-length array
        * HYB: hybrid format comprising of ELLPACK and coordinate

    there are some special purpose implementations:

        * CSR_T: compressed sparse row (transposed)

"""
from . import LIL as LIL_SingleThread
from . import COO as COO_SingleThread
from . import CSR as CSR_SingleThread
from . import CSR_T as CSR_T_SingleThread
from . import ELL as ELL_SingleThread
from . import ELLR as ELLR_SingleThread
from . import HYB as HYB_SingleThread

__all__ = ["BaseTemplates", "LIL_SingleThread", "COO_SingleThread", "CSR_SingleThread", "CSR_T_SingleThread", "ELL_SingleThread", "ELLR_SingleThread", "HYB_SingleThread"]