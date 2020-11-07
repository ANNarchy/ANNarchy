"""
The files of this module define the implementation of the synaptic connectivity using
several sparse matrix formats. To apply such a format, a *ProjStruct* needs to inherit
this template class.

To define a new format the template need to define the following methods:

* Constructor:      which receives two arguments the number of rows and columns of
                    the matrix.
* init_from_lil:    during instantiation of the *ProjStruct* object

Exported:

* SparseMatrixDefinitionsCPU: CPU templates for CSR, CSRC, CSRC_T and LIL, LILInv
* SparseMatrixDefinitionsGPU: GPU templates (TODO)
* HelperFunctions:            template code required by the previous fields, e. g. pairsort or
                              timing code.

Code of conduct:

* all variables related to the connectivity should not be exposed to the python wrapper
* get functions return always a LIL structure
* set function receive always a LIL structure
"""
from .SpMTemplatesCPU import SparseMatrixDefinitionsCPU
from .SpMTemplatesGPU import SparseMatrixDefinitionsGPU

__all__ = ['SparseMatrixDefinitionsCPU', 'SparseMatrixDefinitionsGPU']