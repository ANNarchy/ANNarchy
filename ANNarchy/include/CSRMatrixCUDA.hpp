/*
 *
 *    CSRMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020-21  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
 *  @brief      Implementation of the *compressed sparse row* format on CUDA devices.
 */
template<typename IT = unsigned int>
class CSRMatrixCUDA: public CSRMatrix<IT> {

public:
    IT* gpu_row_ptr;
    IT* gpu_post_rank;
    IT* gpu_pre_rank;

    CSRMatrixCUDA<IT>(const IT num_rows, const IT num_columns) : CSRMatrix<IT>(num_rows, num_columns) {
    }

    void init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::init_matrix_from_lil() " << std::endl;
    #endif
        // host side
        static_cast<CSRMatrix<IT>*>(this)->init_matrix_from_lil(row_indices, column_indices);

        //
        // Copy data
        cudaMalloc((void**)&gpu_post_rank, this->post_ranks_.size()*sizeof(IT));
        cudaMemcpy(gpu_post_rank, this->post_ranks_.data(), this->post_ranks_.size()*sizeof(IT), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&gpu_row_ptr, this->row_begin_.size()*sizeof(IT));
        cudaMemcpy(gpu_row_ptr, this->row_begin_.data(), this->row_begin_.size()*sizeof(IT), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&gpu_pre_rank, this->col_idx_.size()*sizeof(IT));
        cudaMemcpy(gpu_pre_rank, this->col_idx_.data(), this->col_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_from_lil: " << cudaGetErrorString(err) << std::endl;
        }

    }

    //
    //  Variables
    //
    template <typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, this->num_non_zeros_*sizeof(VT));
        cudaMemcpy(gpu_variable, host_variable.data(), this->num_non_zeros_*sizeof(VT), cudaMemcpyHostToDevice);

        return gpu_variable;
    }

    template <typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, this->post_ranks_.size() * sizeof(VT));
        cudaMemcpy(gpu_variable, host_variable.data(), this->post_ranks_.size() * sizeof(VT), cudaMemcpyHostToDevice);

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
            int num_syn = this->row_begin[i+1]-this->row_begin[i];
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

        if ( this->row_begin_[rank] != this->row_begin_[rank+1] ) {
            deFlatVec = std::vector<T>(in.begin()+this->row_begin_[rank], in.begin()+this->row_begin_[rank+1]);
        }

        return deFlatVec;
    }
};