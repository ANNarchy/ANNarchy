#===============================================================================
#
#     SpMTemplatesGPU.py
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

# HD (9th Oct 2020): the include of <random> appeared to be necessary otherwise
#                    nvcc didn't found std::mt19937. The same applies for assert().
SparseMatrixDefinitionsGPU = """#pragma once
#include <random>
#include <cassert>
#include "sparse_matrix.hpp"

#include "CSRMatrixCUDA.hpp"
#include "CSRCMatrixCUDA.hpp"
#include "COOMatrixCUDA.hpp"
#include "ELLRMatrixCUDA.hpp"
#include "HYBMatrixCUDA.hpp"
"""