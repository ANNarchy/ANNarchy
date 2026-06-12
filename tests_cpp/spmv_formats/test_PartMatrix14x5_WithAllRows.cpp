/*
 *    test_PartMatrix14x5_WithAllRows.cpp
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
class Matrix14x5_WithAllRows: public ::testing::Test {
protected:
    // The matrix configuration which should be tested
    using PartMatrixType = typename Config::MatrixClassType;
    using PartStorageType = typename Config::MatrixValueType;

    // Instance of the matrix
    PartitionedMatrix<PartMatrixType, int, int>* mat_;

    // LIL representation used as "reference"
    std::vector<int> post_ranks_;
    std::vector<std::vector<int>> pre_ranks_;
    std::vector<std::vector<double>> values_;

    // run before each test ...    
    void SetUp() override {

        /*
            We will use this matrix pattern as test example:

                0,  0,  0,  0,  1,
                0,  0,  0,  2,  3,
                0,  0,  0,  4,  0,
                5,  6,  7,  0,  0,
                8,  0,  0,  0,  0,
                9,  0,  0,  0, 10,
                0,  0,  0, 11,  0,
               12,  0,  0,  0,  0,
               13, 14,  0,  0,  0,
                0, 15,  0,  0,  0,
                0,  0, 16, 17, 18,
                0,  0,  0,  0, 19,
               20,  0,  0,  0, 21,
                0, 22,  0,  0,  0,

         */

        // In ANNarchy matrices are initialized in most case from a LIL
        // structure, we need to construct this here from hand:

        post_ranks_.push_back(0);       // 1st row
        pre_ranks_.push_back(std::vector<int>({4}));
        values_.push_back(std::vector<double>({1.0}));

        post_ranks_.push_back(1);       // 2nd row
        pre_ranks_.push_back(std::vector<int>({3, 4}));
        values_.push_back(std::vector<double>({2.0, 3.0}));

        post_ranks_.push_back(2);       // 3rd row
        pre_ranks_.push_back(std::vector<int>({3}));
        values_.push_back(std::vector<double>({4.0}));

        post_ranks_.push_back(3);       // 4th row
        pre_ranks_.push_back(std::vector<int>({0, 1, 2}));
        values_.push_back(std::vector<double>({5.0, 6.0, 7.0}));

        post_ranks_.push_back(4);       // 5th row
        pre_ranks_.push_back(std::vector<int>({0}));
        values_.push_back(std::vector<double>({8.0}));
        
        post_ranks_.push_back(5);       // 6th row 
        pre_ranks_.push_back(std::vector<int>({0, 4}));
        values_.push_back(std::vector<double>({9.0, 10.0}));

        post_ranks_.push_back(6);       // 7th row 
        pre_ranks_.push_back(std::vector<int>({3}));
        values_.push_back(std::vector<double>({11.0}));

        post_ranks_.push_back(7);       //  8th row 
        pre_ranks_.push_back(std::vector<int>({0}));
        values_.push_back(std::vector<double>({12.0}));

        post_ranks_.push_back(8);       //  9th row 
        pre_ranks_.push_back(std::vector<int>({0, 1}));
        values_.push_back(std::vector<double>({13.0, 14.0}));

        post_ranks_.push_back(9);       // 10th row 
        pre_ranks_.push_back(std::vector<int>({1}));
        values_.push_back(std::vector<double>({15.0}));

        post_ranks_.push_back(10);      // 11th row
        pre_ranks_.push_back(std::vector<int>({2, 3, 4}));
        values_.push_back(std::vector<double>({16.0, 17.0, 18.0}));

        post_ranks_.push_back(11);      // 12th row 
        pre_ranks_.push_back(std::vector<int>({4}));
        values_.push_back(std::vector<double>({19.0}));

        post_ranks_.push_back(12);      // 13th row 
        pre_ranks_.push_back(std::vector<int>({0, 4}));
        values_.push_back(std::vector<double>({20.0, 21.0}));

        post_ranks_.push_back(13);      // 14th row 
        pre_ranks_.push_back(std::vector<int>({1}));
        values_.push_back(std::vector<double>({22.0}));

        // Initialize the Matrix container.
        mat_ = new PartitionedMatrix<PartMatrixType, int, int>(14, 5);

        // This will results in three partitions: 5x5, 5x5, and 4x5
        mat_->init_matrix_from_lil(post_ranks_, pre_ranks_, 3);
    }

    // Run after each test ...    
    void TearDown() override {
        delete mat_;
    }


};

//
// Add the SpMV implementations, see MatrixCommon.hpp for a list of formats.
TYPED_TEST_SUITE(Matrix14x5_WithAllRows, PartitionedMatrixImplementations);

/********************************************************************************************/
/*  Represent the non-zeros aka connectivity                                                */
/*  which includes the ANNarchy wrapper accessors                                           */
/********************************************************************************************/

TYPED_TEST(Matrix14x5_WithAllRows, DenseSizeAccessors) {
    // dense sizes   
    EXPECT_EQ(this->mat_->num_rows(), 14);
    EXPECT_EQ(this->mat_->num_columns(), 5);
}

TYPED_TEST(Matrix14x5_WithAllRows, PostRanksAccessor) {
    // read-out
    auto post_ranks = this->mat_->get_post_rank();

    // Top-level vectors equal in size?
    EXPECT_EQ(this->post_ranks_.size(), post_ranks.size());

    // Compare the content of the vectors
    auto post_it1 = this->post_ranks_.cbegin();
    auto post_it2 = post_ranks.cbegin();

    for(; post_it1 != this->post_ranks_.cend(); ++post_it1, ++post_it2) {
        EXPECT_EQ(*post_it1, *post_it2);
    }
}

TYPED_TEST(Matrix14x5_WithAllRows, PreRanksAccessor) {
    // read-out column indices, placed row-wise in vectors
    auto pre_ranks = this->mat_->get_pre_ranks();

    // Top-level vectors equal in size?
    EXPECT_EQ(this->pre_ranks_.size(), pre_ranks.size());

    // Compare for each sub-vector if they are equal in size?
    auto pre_it1 = this->pre_ranks_.cbegin();
    auto pre_it2 = pre_ranks.cbegin();

    for(; pre_it1 != this->pre_ranks_.cend(); ++pre_it1, ++pre_it2) {
        EXPECT_EQ(pre_it1->size(), pre_it2->size());
    }

    // Last, we compare the column indices, row-by-row
    pre_it1 = this->pre_ranks_.cbegin();
    pre_it2 = pre_ranks.cbegin();
    for(; pre_it1 != this->pre_ranks_.cend(); ++pre_it1, ++pre_it2) {
        auto pre_sub_it1 = pre_it1->cbegin();
        auto pre_sub_it2 = pre_it2->cbegin();

        for(; pre_sub_it1 != pre_it1->cend(); ++pre_sub_it1, ++pre_sub_it2) {
            EXPECT_EQ(*pre_sub_it1, *pre_sub_it2);
        }
    }
}

TYPED_TEST(Matrix14x5_WithAllRows, GetNumberOfDendrites) {
    // Compare number of rows with non-zeros
    EXPECT_EQ(this->mat_->nb_dendrites(), this->pre_ranks_.size());
}

TYPED_TEST(Matrix14x5_WithAllRows, GetNumberOfSynapses) {
    // How many non-zeros
    int nnz = 0;
    for (auto it = this->pre_ranks_.cbegin(); it != this->pre_ranks_.cend(); ++it)
        nnz += it->size();

    EXPECT_EQ(this->mat_->nb_synapses(), nnz);
}

TYPED_TEST(Matrix14x5_WithAllRows, GetDendriteSize) {
    // compare row-lengths (note need to access with LIL index)
    int i = 0;
    for (auto it = this->pre_ranks_.cbegin(); it != this->pre_ranks_.cend(); ++it, ++i)
        EXPECT_EQ(it->size(), this->mat_->dendrite_size(i));
}

TYPED_TEST(Matrix14x5_WithAllRows, GetDendriteRanks) {
    // compare the access to a specific row
    auto exp_pre_ranks = this->pre_ranks_[2];
    auto act_pre_ranks = this->mat_->get_dendrite_pre_rank(2);

    EXPECT_EQ(exp_pre_ranks.size(), act_pre_ranks.size());

    auto it1 = exp_pre_ranks.cbegin();
    auto it2 = act_pre_ranks.cbegin();

    for (; it1 != exp_pre_ranks.cend(); ++it1, ++it2) {
        EXPECT_EQ(*it1, *it2);
    }
}
