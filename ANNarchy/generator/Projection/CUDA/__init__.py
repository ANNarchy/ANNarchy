"""
The CUDA package does contain all code templates required for the code generation in ANNarchy 
targeting NVIDIA graphic cards.

BaseTemplates:

    defines the basic defintions common to all sparse matrix formates, e. g. projection header

[FORMAT]_SingleThread:

    defines the format specific defintions for the currently available formats:

        * LIL: list-in-list
        * COO: coordinate
        * CSR: compressed sparse row
        * ELL: ELLPACK with some GPU-specific optimizations
"""
__all__ = ["BaseTemplates", "COO_CUDA", "CSR_CUDA", "ELLR_CUDA"]