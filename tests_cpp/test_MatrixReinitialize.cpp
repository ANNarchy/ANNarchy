/*
 *    test_MatrixReinitialize.cpp
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
#include "test_MatrixCommon.hpp"

template<typename Config>
class MatrixReinitialize: public ::testing::Test {
protected:
    // The matrix configuration which should be tested
    using MatrixType = typename Config::MatrixClassType;
    using StorageType = typename Config::MatrixValueType;

    // Instance of the matrix
    MatrixType* mat_;

    // LIL representation used as "reference"
    std::vector<int> post_ranks_;
    std::vector<std::vector<int>> pre_ranks_;
    std::vector<std::vector<double>> values_;

    // run before each test ...    
    void SetUp() override {

        /*
            We will use this matrix pattern as test example:

                0,  0,  0,  0,  1,  0,  0,  0,
                0,  2,  3,  0,  0,  0,  0,  0,
                0,  0,  0,  4,  5,  6,  7,  0,
                0,  0,  0,  0,  0,  0,  0,  8,
                9,  0,  0,  0, 10,  0,  0,  0,
                0, 11,  0,  0,  0,  0,  0, 12

         */

        // In ANNarchy matrices are initialized in most case from a LIL
        // structure, we need to construct this here from hand:

        post_ranks_.push_back(0);       // 1st row
        pre_ranks_.push_back(std::vector<int>({4}));
        values_.push_back(std::vector<double>({1.0}));

        post_ranks_.push_back(1);       // 2nd row
        pre_ranks_.push_back(std::vector<int>({1,2}));
        values_.push_back(std::vector<double>({2.0, 3.0}));

        post_ranks_.push_back(2);       // 3rd row
        pre_ranks_.push_back(std::vector<int>({3,4,5,6}));
        values_.push_back(std::vector<double>({4.0, 5.0, 6.0, 7.0}));

        post_ranks_.push_back(3);       // 4th row
        pre_ranks_.push_back(std::vector<int>({7}));
        values_.push_back(std::vector<double>({8.0}));

        post_ranks_.push_back(4);       // 5th row
        pre_ranks_.push_back(std::vector<int>({0,4}));
        values_.push_back(std::vector<double>({9.0, 10.0}));
        
        post_ranks_.push_back(5);       // 6th row 
        pre_ranks_.push_back(std::vector<int>({1,7}));
        values_.push_back(std::vector<double>({11.0, 12.0}));

        //
        // Initialize the Matrix container
        if constexpr(std::is_same_v<MatrixType, BSRMatrix<int, int, char, true>> ||
                     std::is_same_v<MatrixType, BSRMatrixBitmask<int, int, char, true>>) {
            // BSR needs a additional block size
            mat_ = new MatrixType(6, 8, 2);
        } else if constexpr(std::is_same_v<MatrixType, SELLMatrix<int, int, true>>) {
            // SELL needs slice size S, here I choose 3 this means two blocks will be created.
            mat_ = new MatrixType(6, 8, 3);
        } else {
            // for most formats: num_rows, num_columns
            mat_ = new MatrixType(6,8);
        }

        mat_->init_matrix_from_lil(post_ranks_, pre_ranks_);
    }

    // Run after each test ...    
    void TearDown() override {
        delete mat_;
    }
};

//
// Add the SpMV implementations, see MatrixCommon.hpp for a list of formats.
TYPED_TEST_SUITE(MatrixReinitialize, MatrixImplementations);

TYPED_TEST(MatrixReinitialize, VerifyInitialState) {
    // Dense sizes
    EXPECT_EQ(this->mat_->num_rows(), 6);
    EXPECT_EQ(this->mat_->num_columns(), 8);

    // How many non-zeros
    int nnz = 0;
    for (auto it = this->pre_ranks_.cbegin(); it != this->pre_ranks_.cend(); ++it)
        nnz += it->size();

    EXPECT_EQ(this->mat_->nb_synapses(), nnz);
}

TYPED_TEST(MatrixReinitialize, DenseSizeAccessors) {
    // Call clear
    this->mat_->clear();

    // dense sizes should not change
    EXPECT_EQ(this->mat_->num_rows(), 6);
    EXPECT_EQ(this->mat_->num_columns(), 8);
}

TYPED_TEST(MatrixReinitialize, PostRanksAccessor) {
    // Call clear
    this->mat_->clear();

    // read-out row indices
    auto post_ranks = this->mat_->get_post_rank();

    // The vector should be empty
    EXPECT_TRUE(post_ranks.empty());
}

TYPED_TEST(MatrixReinitialize, GetNumberOfDendrites) {
    // Call clear
    this->mat_->clear();

    // No rows allocated.
    EXPECT_EQ(this->mat_->nb_dendrites(), 0);
}