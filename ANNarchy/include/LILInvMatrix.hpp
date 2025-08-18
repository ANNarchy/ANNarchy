/*
 *    LILInvMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
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

#include "LILMatrix.hpp"

/**
 *  @brief      Implementation of the *list-in-list* (LIL) sparse matrix format providing forward and backward indexing.
 *  @details    Inherits the LILMatrix class and extends the available variables which form the *forward* view by additional
 *              indexing from the column side. The backward view allows a column-based indexing of the connectivity matrix.
 *              But be aware, that there is a higher spreading of memory access. As the LILMatrix is inherited and ANNarchy
 *              user-interface is row-oriented, this class does not contain any get/set or initialize methods for the variables.
 *              The connector methods should call the LILMatrix counter parts and LILInvMatrix::inverse_connectivity_matrix()
 *              afterwards.
 *
 *              Let's consider the following example matrix
 *
 *                      | 0 1 0 |
 *                  A = | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 *
 *              has the LIL representation
 *
 *                  post_rank = [ 0 1 3 ]
 *
 *                  pre_rank = [ [ 1 ], [ 0, 2  ], [ 2 ] ]
 *
 *              The backward view would consists of std::map<> with the keys 0, 1, 2 as all columns contain at least one value. The attached vectors
 *              will contain the row indices:
 *
 *                  inv_pre_rank = [
 *                      0: [1],
 *                      1: [0],
 *                      2: [1, 3]
 *                  ]
 *
 *              The inv_post_rank will then contain the dense column indices additionally:
 *
 *                  inv_post_rank = [0, 1, 2]
 *
 *  @tparam     IT      data type to represent the ranks within the matrix. Please refer to LILMatrix for more details.
 *  @tparam     ST      the second type should be used if the index type IT could overflow. Please refer to LILMatrix for more details.
 *
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class LILInvMatrix: public LILMatrix<IT, ST> {
  public:
    /**
     *  @brief      Constructor
     */
    explicit LILInvMatrix(const IT num_rows, const IT num_columns)  : LILMatrix<IT, ST>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "Created LIL-type matrix with backward view " << this << " using dense dimension " << static_cast<long>(this->num_rows_) << "x" << static_cast<long>(this->num_columns_) << std::endl;
    #endif
    }

    /**
     *  @brief      Destructor
     */
    ~LILInvMatrix() {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::~LILInvMatrix(this=" << this << ")" << std::endl;
    #endif
    }

    /**
     *  @brief      Clear the containers.
     */
    void clear() override {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::clear(this=" << this << ")" << std::endl;
    #endif
        // clear forward view
        LILMatrix<IT, ST>::clear();

        // clear backward view
        inv_post_rank.clear();
        inv_post_rank.shrink_to_fit();

        for (auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        inv_pre_rank.clear();
    }

    /// don't copy the container instead returns a reference
    std::vector< IT >& get_inv_post_rank() { return inv_post_rank; }

    /// don't copy the container instead returns a reference
    const std::map< IT, std::vector< std::pair<IT, IT> > >& get_inv_pre_rank() const { return inv_pre_rank; }

    /**
     *  @see    LILMatrix::init_matrix_from_lil()
     */
    bool init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::init_matrix_from_lil():" << std::endl;
    #endif

        // create forward view
        bool success = static_cast<LILMatrix<IT, ST>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // compute backward view
        inverse_connectivity_matrix();

        // done
        return true;
    }

    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @details    this function creates also the variable array which is usually created/initialized afterwards.
     *  @tparam     VT          value type of the nonzero
     *  @tparam     zero_based  set to true if the contained data in csv has as minimum possible index 0. If
     *                          set to false, the read-in indices will be decremented by 1.
     *  @see        LILMatrix::init_matrix_from_csv()
     */
    template<typename VT, bool zero_based=true>
    std::vector<std::vector<VT>> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::init_matrix_from_csv()" << std::endl;
    #endif
        // create forward view
        auto values = static_cast<LILMatrix<IT, ST>*>(this)->template init_matrix_from_csv<VT, zero_based>(filename, delimiter);

        // compute backward view
        inverse_connectivity_matrix();

        return values;
    }

    /**
     *  @see LILMatrix::fixed_number_pre_pattern()
     */
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::fixed_number_pre_pattern():" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrix<IT, ST>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // compute backward view
        inverse_connectivity_matrix();
    }

    /**
     *  @see LILMatrix::fixed_probability_pattern()
     */
    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::fixed_probability_pattern():" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrix<IT, ST>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // compute backward view
        inverse_connectivity_matrix();
    }

    /**
     *  @brief      get a list of pre-synaptic neuron ranks and their efferent connections.
     *  @details    while the LILMatrix::nb_synapses and LILMatrix::nb_synapses_per_dendrite are row-centered this
     *              function contains the number of row entries for all columns with at least one row entry.
     *  @returns    a std::map with the pre-synaptic ranks as index and the number of nonzeros per column.
     */
    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();
        for (auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++) {
            num_efferents[it->first] = it->second.size();
        }

        return num_efferents;
    }

    // Returns size in bytes for connectivity
    size_t size_in_bytes() override {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::size_in_bytes(this=" << this << ")" << std::endl;
    #endif
        // constants
        size_t size = 2 * sizeof(unsigned int);

        // forward view
        size += LILMatrix<IT, ST>::size_in_bytes();

        // backward - column indices
        size += sizeof(std::vector<IT>);
        size += inv_post_rank.capacity() * sizeof(IT);

        // backward - inverted matrix
        size += sizeof(std::map< IT, std::vector< std::pair<IT, IT> > >);
        for ( auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++ ) {
            size += sizeof(IT); // key
            size += (it->second).capacity() * sizeof(IT); // value
        }

        return size;
    }

  protected:
    /**
     *  @brief      Create backward view
     *  @details    based on the forward connectivity stored in LILMatrix::post_rank and LILMatrix::pre_rank we
     *              compute backward view.
     */ 
    void inverse_connectivity_matrix() {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::inverse_connectivity_matrix():" << std::endl;
    #endif
        // std::map < dense column_index, < sparse row_idx, sparse col_idx > >
        inv_pre_rank =  std::map< IT, std::vector< std::pair<IT, IT> > > ();
        for (int i=0; i<this->pre_rank.size(); i++) {
            for (int j=0; j<this->pre_rank[i].size(); j++) {
                inv_pre_rank[this->pre_rank[i][j]].push_back(std::pair<IT, IT>(i,j));
            }
        }

        // store the dense column indices, please note that std::map has sorted indices
        inv_post_rank = std::vector< IT >();
        for (auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++) {
            inv_post_rank.push_back(it->first);
        }

        // remove unneccessary allocated bytes (caused by push_back)
        for (auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++) {
            // idx pairs are stored in avector
            it->second.shrink_to_fit();
        }
        inv_post_rank.shrink_to_fit();
    }

    void print_data_representation() {
        // Forward view
        LILMatrix<IT, ST>::print_data_representation();

        // Backward view
        std::cout << "LILInvMatrix instance at " << this <<  std::endl;
        std::cout << "inv_pre_rank <pre_index, vector<pair<post_index, pre_index>>:" << std::endl;
        for (auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++) {
            std::cout << it->first << ": ";
            for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++ ) {
                std::cout << "(" << it2->first << "," << it2->second << ")" << std::endl;
            }
            std::cout << std::endl;
        }
    }

  protected:
    // Backward view
    std::vector< IT > inv_post_rank ;
    std::map< IT, std::vector< std::pair<IT, IT> > > inv_pre_rank ;
};
