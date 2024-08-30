/*
 *    CSRCMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020-21  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#include "CSRMatrix.hpp"

/**
 *  @brief      Implementation of the compressed sparse row and column (CSRC) matrix format.
 *  @details    A detailed explaination of the format can be found e. g. in 
 *              
 *                  Brette, R. and Goodman, D. F. M. (2011). Vectorized algorithms for spiking neural network
 *                  simulation. Neural computation, 23(6):1503–1535.
 *
 *              The major idea is to extend the forward view of the CSR format by an backward view to
 *              allow a column-oriented indexing which is required for the spike propagation.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class CSRCMatrix: public CSRMatrix<IT, ST> {
protected:
    // CSR inverse
    std::vector<ST> _col_ptr;       ///< the i-th and (i+1)-th entry marks the begin and end of a column.
    std::vector<IT> _row_idx;       ///< the row indices within a column
    std::vector<ST> _inv_idx;       ///< supposed for the access the data arrays (which are stored in forward view)

public:
    explicit CSRCMatrix(const IT num_rows, const IT num_columns) : CSRMatrix<IT, ST>(num_rows, num_columns) {
        _col_ptr = std::vector<ST>(this->num_columns_+1); 
    }

    /*
     *  \brief      Destructor
     *  \details    Destroys only components which belongs to backward view.
     */
    ~CSRCMatrix() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrix::~CSRCMatrix()" << std::endl;
    #endif
    }

    /**
     * \details     Clear the STL container
     */
    void clear() {
        // clear forward view
        static_cast<CSRMatrix<IT, ST>*>(this)->clear();

    #ifdef _DEBUG
        std::cout << "CSRCMatrix::clear()" << std::endl;
    #endif
        // clear backward view
        _col_ptr.clear();
        _col_ptr.shrink_to_fit();
        _row_idx.clear();
        _row_idx.shrink_to_fit();
        _inv_idx.clear();
        _inv_idx.shrink_to_fit();
    }

    //
    //  Attribute accessors
    //
    inline ST* col_ptr() {
        return _col_ptr.data();
    }

    inline IT* row_indices() {
        return _row_idx.data();
    }

    inline ST* inverse_indices() {
        return _inv_idx.data();
    }

    /**
     *  @brief      initialize from LIL representation.
     *  @see        LILMatrix::init_matrix_from_lil(), CSRMatrix::init_matrix_from_lil()
     */
    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrix::init_matrix_from_lil():" << std::endl;
    #endif
        // create forward view
        bool success = static_cast<CSRMatrix<IT, ST>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // compute backward view
        inverse_connectivity_matrix();

        return true;
    }

    /**
     *  @brief      initialize from a .csv file.
     *  @see        LILMatrix::init_matrix_from_csv()
     */
    template<typename VT, bool zero_based=true>
    std::vector<VT> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::init_matrix_from_csv()" << std::endl;
    #endif
        // create forward view
        auto values = static_cast<CSRMatrix<IT, ST>*>(this)->template init_matrix_from_csv<VT, zero_based>(filename, delimiter);

        // compute backward view
        inverse_connectivity_matrix();

        return values;
    }

    //
    //  ANNarchy connectivity patterns
    //
    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrix::fixed_number_pre_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " nnz per row: " << nnz_per_row << std::endl;
    #endif
        clear();

        // create forward view
        static_cast<CSRMatrix<IT, ST>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // compute backward view
        inverse_connectivity_matrix();

        return true;
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
        clear();
    #ifdef _DEBUG
        std::cout << "CSRCMatrix::fixed_probability_pattern():" << std::endl;
    #endif
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT, ST>(this->num_rows_, this->num_columns_);
        lil_mat->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // Generate CSRC_T from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;
    }

    void inverse_connectivity_matrix() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrix::inverse_connectivity_matrix()" << std::endl;
    #endif
        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        auto inv_post_rank = std::map< IT, std::vector< IT > >();
        auto inv_idx = std::map< IT, std::vector< ST > >();

        // iterate over post neurons, post_rank_it encodes the current rank
        for (IT i = 0; i < this->num_rows_; i++ ) {
            ST row_begin = this->row_begin_[i];
            ST row_end = this->row_begin_[i+1];

            // iterate over synapses, update both result containers
            for (ST syn_idx = row_begin; syn_idx < row_end; syn_idx++ ) {
                inv_post_rank[this->col_idx_[syn_idx]].push_back(i);
                inv_idx[this->col_idx_[syn_idx]].push_back(syn_idx);
            }
        }

        //
        // store as CSR
        //
        _col_ptr.clear();
        _row_idx.clear();
        _inv_idx.clear();
        ST curr_off = 0;

        // iterate over pre-neurons
        for ( IT i = 0; i < this->num_columns_; i++) {
            _col_ptr.push_back( curr_off );
            if ( !inv_post_rank[i].empty() ) {
                _row_idx.insert(_row_idx.end(), inv_post_rank[i].begin(), inv_post_rank[i].end());
                _inv_idx.insert(_inv_idx.end(), inv_idx[i].begin(), inv_idx[i].end());

                curr_off += inv_post_rank[i].size();
            }
        }
        _col_ptr.push_back(curr_off);

        // remove unneccessary allocated space
        _row_idx.shrink_to_fit();
        _inv_idx.shrink_to_fit();

        // sanity check
        if ( this->num_non_zeros_ != curr_off ) {
            std::cerr << "Something went wrong: (nb_synapes = " << this->num_non_zeros_ << ") != (curr_off = " << curr_off << ")" << std::endl;
        } else {
        #if defined(_DEBUG_CONN)
            print_data_representation(2, true);
        #elif defined(_DEBUG)
            print_data_representation(2, false);
        #endif
        }
    }

    /**
     *  @brief      get a list of pre-synaptic neuron ranks and their efferent connections.
     *  @details    while the LILMatrix::nb_synapses and LILMatrix::nb_synapses_per_dendrite are row-centered this
     *              function contains the number of row entries for all columns with at least one row entry.
     *  @returns    a std::map with the pre-synaptic ranks as index and the number of nonzeros per column.
     */
    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();

        for (IT i = 0; i < this->num_columns_; i++) {
            if ((_col_ptr[i+1] - _col_ptr[i]) == 0)
                continue;
            
            num_efferents[i] = static_cast<IT>(_col_ptr[i+1] - _col_ptr[i]);
        }

        return num_efferents;
    }

    /**
     *  \brief      Returns size in bytes for connectivity.
     *  \details    Includes the backward and forward view.
     */
    size_t size_in_bytes() {
        size_t size = 0;

        // forward view of CSR
        size += static_cast<CSRMatrix<IT, ST>*>(this)->size_in_bytes();

        // backward view
        size += sizeof(std::vector<ST>);
        size += _col_ptr.capacity() * sizeof(ST);
        size += sizeof(std::vector<IT>);
        size += _row_idx.capacity() * sizeof(IT);
        size += sizeof(std::vector<IT>);
        size += _inv_idx.capacity() * sizeof(ST);

        return size;
    }

    void print_data_representation(int indent_spaces=0, bool print_container=true) {
        int empty_columns = 0;
        for (IT r = 0; r < _col_ptr.size()-1; r++ ) {
            if (_col_ptr[r+1]-_col_ptr[r] == 0)
                empty_columns++;
        }

        if (print_container) {
            std::cout << std::string(indent_spaces, ' ') << "Forward view:" << std::endl;
            static_cast<CSRMatrix<IT, ST>*>(this)->print_data_representation(indent_spaces+2, print_container);

            std::cout << std::string(indent_spaces, ' ') << "Backward view:" << std::endl;
            std::cout << std::string(indent_spaces+2, ' ') << "#empty columns: " << empty_columns << std::endl;
            std::cout << std::string(indent_spaces+2, ' ') << "CSRCMatrix instance at " << this << std::endl;

            std::cout << std::string(indent_spaces+4, ' ') << "col_begin = [ ";
            for (IT c = 0; c < _col_ptr.size(); c++ ) {
                std::cout << static_cast<unsigned long>(_col_ptr[c]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+4, ' ') << "row_idx = [ ";
            for (auto i = 0; i < _row_idx.size(); i++ ) {
                std::cout << static_cast<unsigned long>(_row_idx[i]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+4, ' ') << "inv_idx = [ ";
            for (auto i = 0; i < _inv_idx.size(); i++ ) {
                std::cout << static_cast<unsigned long>(_inv_idx[i]) << " ";
            }
            std::cout << "]" << std::endl;
        }else {
            std::cout << std::string(indent_spaces, ' ') << "#empty columns: " << empty_columns << std::endl;
        }

    }
};
