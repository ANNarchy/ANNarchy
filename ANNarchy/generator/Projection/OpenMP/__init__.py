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
        * CSR_T: compressed sparse row (transposed)
        * ELL: ELLPACK/ITPACK
"""
__all__ = ["BaseTemplates", "LIL_OpenMP", "COO_OpenMP", "CSR_OpenMP", "CSR_T_OpenMP", "ELL_OpenMP"]