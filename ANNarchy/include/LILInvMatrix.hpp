/*
 *
 *    LILInvMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *    Julien Vitay <julien.vitay@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

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
 */
template<typename IT = unsigned int>
class LILInvMatrix: public LILMatrix<IT> {
public:
    // Backward view
    std::map< IT, std::vector< std::pair<IT, IT> > > inv_pre_rank ;
    std::vector< IT > inv_post_rank ;

    /*
     *  @brief      Create backward view
     *  @details    based on the forward connectivity stored in LILMatrix::post_rank and LILMatrix::pre_rank we
     *              compute backward view.
     */ 
    void inverse_connectivity_matrix() {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::inverse_connectivity_matrix():" << std::endl;
    #endif
        inv_pre_rank =  std::map< IT, std::vector< std::pair<IT, IT> > > ();
        for (int i=0; i<this->pre_rank.size(); i++) {
            for (int j=0; j<this->pre_rank[i].size(); j++) {
                inv_pre_rank[this->pre_rank[i][j]].push_back(std::pair<IT, IT>(i,j));
            }
        }
        inv_post_rank =  std::vector< IT > (this->num_rows_, -1);
        for (int i=0; i<this->post_rank.size(); i++) {
            inv_post_rank[this->post_rank[i]] = i;
        }
    }

public:
    LILInvMatrix(const IT num_rows, const IT num_columns)  : LILMatrix<IT>(num_rows, num_columns) {
    }

    /**
     *  @see LILMatrix::init_matrix_from_lil()
     */
    void init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::init_matrix_from_lil():" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrix<IT>*>(this)->init_matrix_from_lil(row_indices, column_indices);

        // compute backward view
        inverse_connectivity_matrix();
    }

    /**
     *  @see LILMatrix::fixed_number_pre_pattern()
     */
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::fixed_number_pre_pattern():" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrix<IT>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

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
        static_cast<LILMatrix<IT>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // compute backward view
        inverse_connectivity_matrix();
    }

    void print_data_representation() {
        // Forward view
        static_cast<LILMatrix<IT>*>(this)->print_data_representation();

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

    // Returns size in bytes for connectivity
    size_t size_in_bytes() {
        // constants
        size_t size = 2 * sizeof(unsigned int);

        // forward view
        size += static_cast<LILMatrix<IT>*>(this)->size_in_bytes();

        // backward
        size += inv_post_rank.capacity() * sizeof(IT);
        for ( auto it = inv_pre_rank.begin(); it != inv_pre_rank.end(); it++ ) {
            size += sizeof(IT); // key
            size += (it->second).capacity() * sizeof(IT); // value
        }

        return size;
    }
};