/*
 *    BSRInvMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#include <vector>
#include <map>
#include <algorithm>

#include "BSRMatrix.hpp"

/**
 *	\brief		Implementation of a blocked compressed sparse row (BSR) format.
 *	\details	A blocked variant of the classic compressed sparse row matrix format. It is basically
 *              the idea of a compressed sparse row, but instead of single values we encode a dense block.
 *              The format has been described in detail, e.g. in:
 * 
 *              * Eberhardt & Hoemmen (2016): Optimization of block sparse matrix-vector multiplication on shared-memory parallel architectures
 *
 *	\tparam 	IT		    index data type
 *	\tparam 	VT		    value data type
 *  \tparam     MT          As a zero can represent a non-existing entry or a existing entry, we need an additional array which encodes if a position is
 *                          a non-zero entry in the matrix. In this implementation each value of the mask corresponds to one position. The size of each
 *                          entry is determined by MT (we recommend char as it consumes only 1 byte).
 *  \tparam     row_major   row_major   determines the matrix storage for the dense sub matrices. If
 *                          set to true, the matrix will be stored as row major, otherwise
 *                          in column major.
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major=true>
class BSRInvMatrix: public BSRMatrix<IT, ST, MT, row_major> {

protected:
    // The BSR has a CSR-like top level structure. Consequently, we can apply the idea of the CSRC (Brette & Goodman 2011).
    std::vector<IT>     block_column_pointer_;
    std::vector<IT>     block_row_index_;
    std::vector<IT>     block_inv_index_;

    bool inverse_connectivity_matrix() {
    #ifdef _DEBUG
        std::cout << "BSRInvMatrix::inverse_connectivity_matrix()" << std::endl;
    #endif

        // Iterate across the BSR and extract tile positions
        auto inv_view_col_idx = std::map<int, std::vector<IT>>();
        auto inv_view_tile_idx = std::map<int, std::vector<IT>>();

        for (IT block_row_idx = 0; block_row_idx < this->block_row_pointer_.size()-1; block_row_idx++ ) {
            for (IT blk_col_idx = this->block_row_pointer_[block_row_idx]; blk_col_idx < this->block_row_pointer_[block_row_idx+1]; blk_col_idx++) {
                inv_view_col_idx[this->block_column_index_[blk_col_idx]].push_back(block_row_idx);
                inv_view_tile_idx[this->block_column_index_[blk_col_idx]].push_back(blk_col_idx);
            }
        }

        // Store the inverse view as CSR-like
        IT nb_block_columns = IT(ceil(double(this->num_columns_) / double(this->tile_size_)));
        this->block_column_pointer_ = std::vector<IT>(nb_block_columns+1, 0);
        this->block_row_index_ = std::vector<IT>();

        IT offset = 0;
        for (IT blk_col_idx = 0; blk_col_idx < nb_block_columns; blk_col_idx++) {
            block_column_pointer_[blk_col_idx] = offset;
            
            // only process if the key exists
            if (inv_view_col_idx.count(blk_col_idx)) {
                offset += inv_view_col_idx[blk_col_idx].size();

                auto tmp = std::move(inv_view_col_idx[blk_col_idx]);
                auto tmp2 = std::move(inv_view_tile_idx[blk_col_idx]);
                pairsort<IT, IT>(tmp.data(), tmp2.data(), tmp.size());
                
                this->block_row_index_.insert(this->block_row_index_.end(), tmp.begin(), tmp.end());
                this->block_inv_index_.insert(this->block_inv_index_.end(), tmp2.begin(), tmp2.end());
            }
        }
        block_column_pointer_[nb_block_columns] = offset;

        // I'm not completely sure if its needed, but won't hurt
        this->block_row_index_.shrink_to_fit();
        this->block_inv_index_.shrink_to_fit();

        return true;
    }

public:
    /**
     *  \brief      Constructor of a BSRMatrix
     *  \details    which will throw an exception of the number of rows/columns is not divisable by tile_size
     */
    explicit BSRInvMatrix(const unsigned int num_rows, const unsigned int num_columns, const unsigned int tile_size):
        BSRMatrix<IT, ST, MT, row_major>(num_rows, num_columns, tile_size) {
    #ifdef _DEBUG
        std::cout << "BSRInvMatrix::BSRInvMatrix(this=" << this << ")" << std::endl;
    #endif
    }

    ~BSRInvMatrix() {
    #ifdef _DEBUG
        std::cout << "BSRInvMatrix::~BSRInvMatrix(this=" << this << ")" << std::endl;
    #endif
    }

    /**
     * \details     Clear the STL container
     */
    void clear() override {
    #ifdef _DEBUG
        std::cout << "BSRInvMatrix::clear()" << std::endl;
    #endif
        // clear forward view
        BSRMatrix<IT, ST, MT, row_major>::clear();

        // clear backward view        
        block_column_pointer_.clear();
        block_column_pointer_.shrink_to_fit();

        block_row_index_.clear();
        block_row_index_.shrink_to_fit();

        block_inv_index_.clear();
        block_inv_index_.shrink_to_fit();
    }

    //
    //  Accessors for the computation
    //

    //
    //  Initialization methods
    //

    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector<std::vector<IT>> column_indices) {
    #ifdef _DEBUG
        std::cout << "BSRInvMatrix::init_matrix_from_lil()" << std::endl;
    #endif

        // Construct forward view
        bool success = static_cast<BSRMatrix<IT, ST, MT, row_major>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // Construct inverse view
        success = inverse_connectivity_matrix();

        // Debug
    #ifdef _DEBUG
        this->print_data_representation();
    #endif
        return success;
    }

    //
    //  Accessors for the Python ANNarchy interface
    //

    /**
     *  @brief      get a list of pre-synaptic neuron ranks and their efferent connections.
     *  @details    while the LILMatrix::nb_synapses and LILMatrix::nb_synapses_per_dendrite are row-centered this
     *              function contains the number of row entries for all columns with at least one row entry.
     *  @returns    a std::map with the pre-synaptic ranks as index and the number of nonzeros per column.
     */
    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();

        return num_efferents;
    }

    //
    //  Other helpful functions
    //

    /**
     *  \brief      Returns size in bytes for connectivity.
     *  \details    Includes the backward and forward view.
     */
    size_t size_in_bytes() override {
        size_t size = 0;

        // forward view of BSR
        size += BSRMatrix<IT, ST, MT, row_major>::size_in_bytes();

        // inverse view
        size += block_column_pointer_.capacity() * sizeof(IT);
        size += block_row_index_.capacity() * sizeof(IT);
        size += block_inv_index_.capacity() * sizeof(IT);

        return size;
    }

    void print_data_representation(bool print_memory_footprint=true) {
        std::cout << "Forward view:" << std::endl;
        static_cast<BSRMatrix<IT, ST, MT, row_major>*>(this)->print_data_representation(false);

        std::cout << "Number of block columns: " << this->block_column_pointer_.size()-1 << std::endl;
        std::cout << "inv tile indices = [ ";
        for (IT block_column_idx = 0; block_column_idx < this->block_column_pointer_.size()-1; block_column_idx++ ) {
            std::cout << "[ ";
            for (IT blk_row_idx = block_column_pointer_[block_column_idx]; blk_row_idx < block_column_pointer_[block_column_idx+1]; blk_row_idx++) {
                std::cout << block_inv_index_[blk_row_idx] << " ";
            }
            std::cout << "] ";
        }
        std::cout << "]" << std::endl;
        std::cout << "block row indices = [ ";
        for (IT block_column_idx = 0; block_column_idx < this->block_column_pointer_.size()-1; block_column_idx++ ) {
            std::cout << "[ ";
            for (IT blk_row_idx = block_column_pointer_[block_column_idx]; blk_row_idx < block_column_pointer_[block_column_idx+1]; blk_row_idx++) {
                std::cout << block_row_index_[blk_row_idx] << " ";
            }
            std::cout << "] ";
        }
        std::cout << "]" << std::endl;

        if (print_memory_footprint)
            std::cout << "Requires " << (this->size_in_bytes() / 1024.0 / 1024) << "MB (~" << this->size_in_bytes() / this->nb_synapses() << " bytes per non-zero)" << std::endl;
    }
};
