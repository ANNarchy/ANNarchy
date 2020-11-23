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
__all__ = ["BaseTemplates", "LIL_SingleThread", "COO_SingleThread", "CSR_SingleThread", "CSR_T_SingleThread", "ELL_SingleThread", "HYB_SingleThread"]