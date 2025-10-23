/*
 *    test_MatrixCommon.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2025  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

// Google's testbench (will be copied locally by CMake)
#include <gtest/gtest.h>

// required imports for the ANNarchy template implementations
#include <random>
#include <fstream>

// ANNarchy helper functions
#include "helper_functions.hpp"

// ANNarchy Template Class imports
#include "BSRMatrix.hpp"
#include "BSRMatrixBitmask.hpp"
#include "BSRInvMatrix.hpp"
#include "COOMatrix.hpp"
#include "CSRMatrix.hpp"
#include "CSRCMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DenseMatrixBitmask.hpp"
#include "ELLMatrix.hpp"
#include "ELLRMatrix.hpp"
#include "LILMatrix.hpp"
#include "LILInvMatrix.hpp"
#include "SELLMatrix.hpp"
// Special case: receives one of the above classes
// and splits across the row-dimension
#include "PartitionedMatrix.hpp"

// Just a short-hand to easier identify different cases
// in Testclass::generate_expected_variable
enum Pattern {
    INIT = 0,
    SINGLE_POS,
    SINGLE_ROW,
    UPDATE_ALL
};

// Matrix type configuration
template<typename MatrixT, typename StorageT>
struct MatrixTestConfig {
    using MatrixClassType = MatrixT;    // Matrix class definition
    using MatrixValueType = StorageT;   // Which container type for non-zeros
};

// Generate a list of tested implementations
// which will be passed to TYPED_TEST_SUITE
using MatrixImplementations = ::testing::Types<
    // BSR-variants
    MatrixTestConfig<BSRMatrix<int, int, char, true>, std::vector<double>>,
    MatrixTestConfig<BSRMatrixBitmask<int, int, char, true>, std::vector<double>>,
    // COO-Variants
    MatrixTestConfig<COOMatrix<int, int>, std::vector<double>>,
    // CSR-variants
    MatrixTestConfig<CSRMatrix<int, int>, std::vector<double>>,
    MatrixTestConfig<CSRCMatrix<int, int>, std::vector<double>>,
    // Dense-variants
    MatrixTestConfig<DenseMatrix<int, int, char, true>, std::vector<double>>,
    MatrixTestConfig<DenseMatrix<int, int, char, false>, std::vector<double>>,
    MatrixTestConfig<DenseMatrixBitmask<int, int, char, true>, std::vector<double>>,
    // ELLPACK Variations
    MatrixTestConfig<ELLMatrix<int, int, true>, std::vector<double>>,
    MatrixTestConfig<ELLMatrix<int, int, false>, std::vector<double>>,
    MatrixTestConfig<ELLRMatrix<int, int, true>, std::vector<double>>,
    MatrixTestConfig<ELLRMatrix<int, int, false>, std::vector<double>>,
    MatrixTestConfig<SELLMatrix<int, int, true>, std::vector<double>>,
    // LIL-variants / Interface of ANNarchy
    MatrixTestConfig<LILMatrix<int, int>, std::vector<std::vector<double>>>,
    MatrixTestConfig<LILInvMatrix<int, int>, std::vector<std::vector<double>>>
>;

// PartitionedMatrix variants
using PartitionedMatrixImplementations = ::testing::Types<
    MatrixTestConfig<LILMatrix<int, int>, std::vector<std::vector<double>>>
>;