#===============================================================================
#
#     SpMTemplatesCPU.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Julien Vitay <julien.vitay@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
"""
This code will be print into "sparse_matrix.hpp". The C++ compiler will copy and
paste all the below listed *.hpp into one large file. Which is then available to
the other generated classes. 

This approach leads to some constraints/effects:

* no pragma guards are needed in the seperate files. 
* all implementations can access the rng definition (inited by ANNarchy::init())
* no negative effect on compile time: as these files contain only template classes,
  they will only compiled if they are actually used, i. e. inherited by a ProjStruct.
* dependencies between classes are simply solved by order. I would not recommend the usage
  of forward declarations

Adding a new connectivity format:

This can be done simply by adding a new template class, which should at least receive an index
type (IT) as parameter. The class need to provide next to a constructor (actually optional) 
an *init_from_lil* method as shown below.

template<typename IT = unsigned int>
class NewConnectivity {
protected:
    // place here your variables to store connectivity

public:
    NewConnectivity(const unsigned int num_rows, const unsigned int num_columns) {

    }

    void init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    }
};

HINT:

Technically this definition could be also moved to ANNarchy/include. I leave this definition
here, to make extensions by the code generator possible.

TODO:

replace the *std::mt19937 rng* by *std::mt19937 std::vector[rng]* for parallel RNGs.
"""
SparseMatrixDefinitionsCPU = """#pragma once
#include "helper_functions.hpp"
#include "ANNarchy.h"

// List of List
#include "LILMatrix.hpp"
#include "LILInvMatrix.hpp"
#ifndef SKIP_OMP_DEFS
  #include "ParallelLIL.hpp"
#endif

// compressed sparse row
#include "CSRMatrix.hpp"
#include "CSRCMatrix.hpp"
#include "CSRCMatrixT.hpp"
#ifndef SKIP_OMP_DEFS
#include "CSRCMatrixTOMP.hpp"
#endif

// allow the user defined definition aka
// "old-style" connectivity definition
#include "Specific.hpp"
"""