/*
 *    DenseMatrixOffsets.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                        Julien Vitay <julien.vitay@gmail.com>
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

#include "DenseMatrix.hpp"

/*
 *  @brief              Connectivity representation using a full matrix.
 *  @tparam     IT      data type to represent the ranks within the matrix. Generally unsigned data types should be chosen.
 *                      The data type determines the maximum size for the number of elements in a column respectively the number
 *                      of rows encoded in the matrix:
 * 
 *                      - unsigned char (1 byte):        [0 .. 255]
 *                      - unsigned short int (2 byte):   [0 .. 65.535]
 *                      - unsigned int (4 byte):         [0 .. 4.294.967.295]
 *
 *                      The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and ::num_columns_)
 * 
 *              ST      the second type should be used if the index type IT could overflow. For instance, the nb_synapses method should return ST as
 *                      the maximum value in case a full dense matrix would be IT times IT entries.
 *              MT      We need to store if a matrix value is set in a boolean mask. The size of each entry is determined by MT (we recommend char as its only 1 byte).
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major=true>
class DenseMatrixOffsets : public DenseMatrix<IT, ST, MT, row_major> {
protected:
    IT low_row_rank_;
    IT high_row_rank_;
    IT low_column_rank_;
    IT high_column_rank_;

    /*
     *  @brief      Decode the column indices for nonzeros in the matrix.
     */
    virtual std::vector<IT> decode_column_indices(IT row_idx) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixOffsets::decode_column_indices()" << std::endl;
    #endif
        auto indices = std::vector<IT>();
        ST idx;
        if (row_major) {
            for (IT c = 0; c < this->num_columns_; c++) {
                idx = row_idx * this->num_columns_ + c;
                if (this->mask_[idx])
                    indices.push_back(c);
            }
        } else {
            for (IT c = 0; c < this->num_columns_; c++) {
                idx = c * this->num_rows_ + row_idx;
                if (this->mask_[idx])
                    indices.push_back(c);
            }
        }

        return indices;
    }

public:

    /**
     * @brief       Construct a new dense matrix object.
     * @details     This function does not allocate the matrix.
     *
     * @param[in]   num_rows      number of rows in the matrix
     * @param[in]   num_columns   number of columns in the matrix
     */
    explicit DenseMatrixOffsets(const IT low_row_rank, const IT high_row_rank, const IT low_column_rank, const IT high_column_rank):
        DenseMatrix<IT, ST, MT, row_major>(high_row_rank - low_row_rank, high_column_rank - low_column_rank) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixOffsets::DenseMatrixOffsets()" << std::endl;
    #endif

        low_row_rank_ = low_row_rank;
        high_row_rank_ = high_row_rank;
        low_column_rank_ = low_column_rank;
        high_column_rank_ = high_column_rank;

        // we check if we can encode all possible values
        assert( (static_cast<long long>(this->num_rows_ * this->num_columns_) < static_cast<long long>(std::numeric_limits<ST>::max())) );
    }

    /**
     *  @details    get row indices
     *  @returns    a list of row indices for all rows comprising of at least one element
     */
    std::vector<IT> get_post_rank() {
        auto post_ranks = static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->get_post_rank();

        for (IT& rk : post_ranks)
            rk += this->low_row_rank_;

        return post_ranks;
    }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() {
        auto pre_ranks = static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->get_pre_ranks();

        for (auto it = pre_ranks.begin(); it != pre_ranks.end(); it++) {
            for (IT& rk : *it)
                rk += this->low_column_rank_;
        }

        return pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  row_idx     index of the selected row.
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(IT row_idx) {
        auto pre_ranks = static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->get_dendrite_pre_rank(row_idx - this->low_row_rank_);

        for (IT& rk : pre_ranks)
            rk += this->low_column_rank_;

        return pre_ranks;
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    simply sets the post_rank and pre_rank arrays without further sanity checking.
     *  @todo       Instead of duplicating the code, one might transform the post_ranks/pre_ranks array and then call the DenseMatrix::init_matrix_from_lil()
     */
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixOffsets::init_matrix_from_lil()" << std::endl;
    #endif

        // Sanity checks
        assert ( (post_ranks.size() == pre_ranks.size()) );
        assert ( (static_cast<unsigned long int>(post_ranks.size()) <= static_cast<unsigned long int>(std::numeric_limits<IT>::max())) );

        // Sanity check: enough memory?
        if (!check_free_memory(this->num_columns_ * this->num_rows_ * sizeof(MT)))
            return false;

        // Allocate mask
        this->mask_ = std::vector<MT>(this->num_rows_ * this->num_columns_, static_cast<MT>(false));

        // Iterate over LIL and update mask entries to *true* if nonzeros are existing.
        for (IT lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            IT row_idx = post_ranks[lil_idx] - low_row_rank_;

            for (auto inner_col_it = pre_ranks[lil_idx].cbegin(); inner_col_it != pre_ranks[lil_idx].cend(); inner_col_it++) {
                IT col_idx = (*inner_col_it) - low_column_rank_;
                if (row_major)
                    this->mask_[row_idx * this->num_columns_ + col_idx] = static_cast<MT>(true);
                else
                    this->mask_[col_idx * this->num_rows_ + row_idx] = static_cast<MT>(true);
            }
        }

        return true;
    }

    /**
     *  @brief      get a list of pre-synaptic neuron ranks and their efferent connections.
     *  @details    while the LILMatrix::nb_synapses and LILMatrix::nb_synapses_per_dendrite are row-centered this
     *              function contains the number of row entries for all columns with at least one row entry.
     *  @returns    a std::map with the pre-synaptic ranks as index and the number of nonzeros per column.
     */
    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();

        if (row_major) {
            for (IT i = 0; i < this->num_rows_; i++) {
                for (IT j = 0; j < this->num_columns_; j++) {
                    ST idx = i*this->num_columns_ + j;
                    if (this->mask_[idx]) num_efferents[j+this->low_column_rank_]++;
                }
            }
        } else {
            for(IT j = 0; j < this->num_columns_; j++) {
                for(IT i = 0; i < this->num_rows_; i++) {
                    ST idx = j*this->num_rows_ + i;
                    if (this->mask_[idx]) num_efferents[j+this->low_column_rank_]++;
                }
            }
        }

        return num_efferents;
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT row_idx, const std::vector<VT> values)
    {
        return static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->update_matrix_variable_row(variable, row_idx - this->low_row_rank_, values);
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface to retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  row_idx     index of the selected row.
     *  @returns    a vector containing all elements of the provided variable and row_idx
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT>& variable, const IT &row_idx) {
        return static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->get_matrix_variable_row(variable, row_idx - this->low_row_rank_);
    }

    /**
     *  @brief      print the some information on the nonzeros to console.
     *  @details    The print-out will contain among others number rows, number columns, number nonzeros.
     *              Please note, that type casts are required to print-out the numbers encoded if IT or ST
     *              is e.g. unsigned char.
     */
    void print_matrix_statistics() {
        std::cout << "  row offsets: " << static_cast<unsigned long>(this->low_row_rank_) << " - " << static_cast<unsigned long>(this->high_row_rank_) << std::endl;
        std::cout << "  column offsets: " << static_cast<unsigned long>(this->low_column_rank_) << " - " << static_cast<unsigned long>(this->high_column_rank_) << std::endl;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. Please note, that type casts are
     *              required to print-out the numbers encoded if IT or ST is e.g. unsigned char.
     */
    void print_data_representation() {
        // wrapper data
        std::cout << "DenseMatrix with reduced dimensions instance at " << this << std::endl;
        print_matrix_statistics();

        // main data field
        static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->print_data_representation();
    }
};
