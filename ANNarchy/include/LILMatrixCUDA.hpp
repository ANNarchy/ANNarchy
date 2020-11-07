/*
 *
 *    LILMatrixCUDA.cuh
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
  *     @brief      Implementation of the *list-in-list* (LIL) sparse matrix format for CUDA devices.
  *     @details    Inheriting LILMatrix, this class generates a host-side instances and transfers flattened vectors to CPU.
  *     @see        LILMatrix
  */
template<typename IT = unsigned int>
class LILMatrixCUDA: public LILMatrix<IT> {
protected:

    /**
     *  @brief      Transfer host-side LIL representation to GPU
     *  @details    The list-in-list container pre_rank is flattened.
     */
    void host_to_device_transfer() {
        //
        // LIL -> CSR on host
        num_non_zeros_ = 0;
        row_begin = std::vector<IT>(this->post_rank.size()+1, 0);
        auto col_idx = std::vector<IT>();
        auto pre_rank_it = this->pre_rank.cbegin();
        for (auto i = 0; i < this->post_rank.size(); i++, pre_rank_it++) {
            row_begin[i] = col_idx.size();
            col_idx.insert(col_idx.end(), pre_rank_it->cbegin(), pre_rank_it->cend());
        }
        row_begin[this->post_rank.size()] = col_idx.size();
        num_non_zeros_ = col_idx.size();

        assert( (num_non_zeros_ == static_cast<LILMatrix<IT>*>(this)->nb_synapses()) );

        //
        // Copy data
        cudaMalloc((void**)&gpu_post_rank, this->post_rank.size()*sizeof(IT));
        cudaMemcpy(gpu_post_rank, this->post_rank.data(), this->post_rank.size()*sizeof(IT), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&gpu_row_ptr, row_begin.size()*sizeof(IT));
        cudaMemcpy(gpu_row_ptr, row_begin.data(), row_begin.size()*sizeof(IT), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&gpu_pre_rank, col_idx.size()*sizeof(IT));
        cudaMemcpy(gpu_pre_rank, col_idx.data(), col_idx.size()*sizeof(IT), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::host_to_device_transfer: " << cudaGetErrorString(err) << std::endl;
        }
    }

public:
    IT* gpu_row_ptr;
    IT* gpu_post_rank;
    IT* gpu_pre_rank;
    unsigned int num_non_zeros_;

    // needed for CSR->LIL (otherwise we would need to copy it every time ...)
    std::vector<IT> row_begin;

    LILMatrixCUDA(const unsigned int num_rows, const unsigned int num_columns) : LILMatrix<IT>(num_rows, num_columns) {
        
    }

    void init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "LILMatrixCUDA::init_matrix_from_lil() " << std::endl;
    #endif
        // Init on host
        static_cast<LILMatrix<IT>*>(this)->init_matrix_from_lil(row_indices, column_indices);

        // Transfer to device
        host_to_device_transfer();
    }
    
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILMatrixCUDA::fixed_number_pre_pattern() " << std::endl;
    #endif
        // Build up LIL on host
        static_cast<LILMatrix<IT>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // Transfer to device
        host_to_device_transfer();
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Build up LIL on host
        static_cast<LILMatrix<IT>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // Transfer to device
        host_to_device_transfer();
    }

    //
    //  Variables
    //
    template <typename VT>
    VT* init_matrix_variable_gpu(const std::vector< std::vector<VT> > &host_variable) {
        VT* gpu_variable;
        auto flat_variable = flattenArray<VT>(host_variable);

        assert( (flat_variable.size() == num_non_zeros_) );
        
        cudaMalloc((void**)&gpu_variable, num_non_zeros_*sizeof(VT));
        cudaMemcpy(gpu_variable, flat_variable.data(), num_non_zeros_*sizeof(VT), cudaMemcpyHostToDevice);

        return gpu_variable;
    }
    
    template <typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, this->post_rank.size() * sizeof(VT));
        cudaMemcpy(gpu_variable, host_variable.data(), this->post_rank.size() * sizeof(VT), cudaMemcpyHostToDevice);

        return gpu_variable;
    }

    /*
    * (De-)Flattening of LIL structures
    */
    template<typename T>
    std::vector<T> flattenArray(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatVec = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        for ( it = in.begin(); it != in.end(); it++)
        {
            flatVec.insert(flatVec.end(), it->begin(), it->end());
        }

        return flatVec;
    }

    template<typename T>
    std::vector<std::vector<T> > deFlattenArray( std::vector<T> in )
    {
        std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();
        std::vector<int>::iterator it;

        int t=0;
        for ( int i = 0; i < this->post_rank.size(); i++)
        {
            int num_syn = row_begin[i+1]-row_begin[i];
            if ( num_syn > 0 ) {
                std::vector<T> tmp = std::vector<T>(in.begin()+t, in.begin()+t+num_syn);
                t += num_syn;

                deFlatVec.push_back(tmp);
            }
        }

        if ( t != in.size() )
            std::cerr << "DeFlattenArray(): something went wrong ..." << std::endl;

        return deFlatVec;
    }

    template<typename T>
    std::vector<T> deFlattenDendrite ( std::vector<T> in, int rank )
    {
        std::vector<T> deFlatVec = std::vector<T>();
        std::vector<int>::iterator it;

        if ( row_begin[rank] != row_begin[rank+1] ) {
            deFlatVec = std::vector<T>(in.begin()+row_begin[rank], in.begin()+row_begin[rank+1]);
        }

        return deFlatVec;
    }
};
