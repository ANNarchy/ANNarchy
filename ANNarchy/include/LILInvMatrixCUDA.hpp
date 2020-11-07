/*
 *
 *    LILInvMatrixCUDA.hpp
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
 *  @brief      Implements *list-in-list* (LIL) with forward and backward view on CUDA devices.
 *  @details    As for the CPU-side implementation, we inherit the LILMatrixCUDA class with its forward view
 *              and add the backward view. Du this inherition path, the LILInvMatrix::inverse_connectivity_matrix()
 *              is not available and need to be reimplemented in this class.
 */
template<typename IT = unsigned int>
class LILInvMatrixCUDA: public LILMatrixCUDA<IT> {
public:
    // Inverse connectivity, only on gpu
    int* gpu_col_ptr;
    int* gpu_row_idx;
    int* gpu_inv_idx;

    explicit LILInvMatrixCUDA(const unsigned int num_rows, const unsigned int num_columns)  : LILMatrixCUDA<IT>(num_rows, num_columns) {
    }

    void inverse_connectivity_matrix() {
        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        std::vector< std::vector< int > > pre_to_post_rank = std::vector< std::vector< int > >(this->num_columns_, std::vector<int>());
        std::vector< std::vector< int > > pre_to_post_idx = std::vector< std::vector< int > >(this->num_columns_, std::vector<int>());

        // some iterator definitions we need
        typename std::vector<std::vector<int> >::iterator pre_rank_out_it = this->pre_rank.begin();  // 1st level iterator
        typename std::vector<int>::iterator pre_rank_in_it;                                    // 2nd level iterator
        typename std::vector< int >::iterator post_rank_it = this->post_rank.begin();
        int post_lil_idx = 0;

        // iterate over post neurons, post_rank_it encodes the current rank
        for( ; pre_rank_out_it != this->pre_rank.end(); pre_rank_out_it++, post_rank_it++, post_lil_idx++ ) {
            int syn_idx = this->row_begin[post_lil_idx]; // start point of the flattened array, post-side
            // iterate over synapses, update both result containers
            for( pre_rank_in_it = pre_rank_out_it->begin(); pre_rank_in_it != pre_rank_out_it->end(); pre_rank_in_it++) {
                //std::cout << *pre_rank_in_it << "->" << *post_rank_it << ": " << syn_idx << std::endl;
                pre_to_post_rank[*pre_rank_in_it].push_back(*post_rank_it);
                pre_to_post_idx[*pre_rank_in_it].push_back(syn_idx);
                syn_idx++;
            }
        }

        std::vector<int> col_ptr = std::vector<int>( this->num_columns_, 0 );
        int curr_off = 0;
        for ( int i = 0; i < this->num_columns_; i++) {
            col_ptr[i] = curr_off;
            curr_off += pre_to_post_rank[i].size();
        }
        col_ptr.push_back(curr_off);

    #ifdef _DEBUG_CONN
        std::cout << "Pre to Post:" << std::endl;
        for ( int i = 0; i < this->num_columns_; i++ ) {
            std::cout << i << ": " << col_ptr[i] << " -> " << col_ptr[i+1] << std::endl;
        }
    #endif

        // backward view to GPU
        cudaMalloc((void**)&gpu_col_ptr, col_ptr.size()*sizeof(int));
        cudaMemcpy(gpu_col_ptr, col_ptr.data(), col_ptr.size()*sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> row_idx = this->flattenArray(pre_to_post_rank);
        cudaMalloc((void**)&gpu_row_idx, row_idx.size()*sizeof(int));
        cudaMemcpy(gpu_row_idx, row_idx.data(), row_idx.size()*sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> inv_idx = this->flattenArray(pre_to_post_idx);
        cudaMalloc((void**)&gpu_inv_idx, inv_idx.size()*sizeof(int));
        cudaMemcpy(gpu_inv_idx, inv_idx.data(), inv_idx.size()*sizeof(int), cudaMemcpyHostToDevice);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "LILInvMatrixCUDA::inverse_connectivity_matrix(): " << cudaGetErrorString(err) << std::endl;
        }
    }

    void init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::init_matrix_from_lil()" << std::endl;
    #endif
        // initialize forward view
        static_cast<LILMatrixCUDA<IT>*>(this)->init_matrix_from_lil(row_indices, column_indices);

        // compute backward view
        inverse_connectivity_matrix();
    }

    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::fixed_number_pre_pattern()" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrixCUDA<IT>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // compute backward view
        inverse_connectivity_matrix();
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILInvMatrix::fixed_probability_pattern()" << std::endl;
    #endif
        // create forward view
        static_cast<LILMatrixCUDA<IT>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // compute backward view
        inverse_connectivity_matrix();
    }

};
