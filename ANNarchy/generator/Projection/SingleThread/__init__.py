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
        * CSR_T: compressed sparse row (transposed)
        * ELL: ELLPACK/ITPACK
        * HYB: hybrid format comprising of ELLPACK and coordinate
"""
from . import LIL as LIL_SingleThread
from . import COO as COO_SingleThread
from . import CSR as CSR_SingleThread
from . import CSR_T as CSR_T_SingleThread
from . import ELL as ELL_SingleThread
from . import HYB as HYB_SingleThread

__all__ = ["BaseTemplates", "LIL_SingleThread", "COO_SingleThread", "CSR_SingleThread", "CSR_T_SingleThread", "ELL_SingleThread", "HYB_SingleThread"]