/*
 *    CSRCMatrixCUDA.hpp
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

#include "CSRCMatrix.hpp"

/**
 *  @brief      Implementation of the *compressed sparse row and column* format on CUDA devices.
 *  @details    For more details on the format please refer to CSRCMatrix.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class CSRCMatrixCUDA: public CSRCMatrix<IT, ST> {
protected:
    bool check_free_device_memory(size_t required) {
        size_t free, total;
        cudaMemGetInfo( &free, &total );
    #ifdef _DEBUG
        std::cout << "Allocate " << required << " and have " << free << "( " << (double(required)/double(total)) * 100.0 << " percent of total memory)" << std::endl;
    #endif
        return (required < free);
    }

    void free_device_memory() {
        // CSR forward view
        if (gpu_post_rank) {
            cudaFree(gpu_post_rank);
            gpu_post_rank = nullptr;
        }
        if (gpu_row_ptr) {
            cudaFree(gpu_row_ptr);
            gpu_row_ptr = nullptr;
        }
        if (gpu_pre_rank) {
            cudaFree(gpu_pre_rank);
            gpu_pre_rank = nullptr;
        }

        // backward view
        if (gpu_col_ptr) {
            cudaFree(gpu_col_ptr);
            gpu_col_ptr = nullptr;
        }
        if (gpu_row_idx) {
            cudaFree(gpu_row_idx);
            gpu_row_idx = nullptr;
        }
        if (gpu_inv_idx) {
            cudaFree(gpu_inv_idx);
            gpu_inv_idx = nullptr;
        }

        // check for errors
        auto free_err = cudaGetLastError();
        if (free_err != cudaSuccess) {
            std::cerr << "CSRCMatrixCUDA::free_device_memory: " << cudaGetErrorString(free_err) << std::endl;
        }
    }

    bool host_to_device_transfer() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDA::host_to_device()" << std::endl;
    #endif
        //
        //  Free (maybe) existing allocations
        free_device_memory();

        // Sanity check: can we allocate the data?
        size_t req_size = sizeof(IT)*this->post_ranks_.size() + sizeof(ST)*this->row_begin_.size() + sizeof(IT)*this->col_idx_.size() + this->_col_ptr.size()*sizeof(ST) + this->_row_idx.size()*sizeof(IT) + this->_inv_idx.size()*sizeof(IT);
        if (!check_free_device_memory(req_size))
            return false;

        //
        //  Allocate device memory
        cudaMalloc((void**)&gpu_post_rank, this->post_ranks_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_row_ptr, this->row_begin_.size()*sizeof(ST));
        cudaMalloc((void**)&gpu_pre_rank, this->col_idx_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_col_ptr, this->_col_ptr.size()*sizeof(ST));
        cudaMalloc((void**)&gpu_row_idx, this->_row_idx.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_inv_idx, this->_inv_idx.size()*sizeof(IT));
        auto malloc_err = cudaGetLastError();
        if (malloc_err != cudaSuccess) {
            std::cerr << "CSRCMatrixCUDA::init_matrix_from_lil - cudaMalloc: " << cudaGetErrorString(malloc_err) << std::endl;
            return false;
        }

        //
        // Copy data
        cudaMemcpy(gpu_post_rank, this->post_ranks_.data(), this->post_ranks_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_row_ptr, this->row_begin_.data(), this->row_begin_.size()*sizeof(ST), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_pre_rank, this->col_idx_.data(), this->col_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_col_ptr, this->_col_ptr.data(), this->_col_ptr.size()*sizeof(ST), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_row_idx, this->_row_idx.data(), this->_row_idx.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_inv_idx, this->_inv_idx.data(), this->_inv_idx.size()*sizeof(IT), cudaMemcpyHostToDevice);
        auto copy_err = cudaGetLastError();
        if (copy_err != cudaSuccess) {
            std::cerr << "CSRCMatrixCUDA::init_matrix_from_lil - cudaMemcpy: " << cudaGetErrorString(copy_err) << std::endl;
            return false;
        }

        return true;
    }

public:
    // CSR forward view
    IT* gpu_post_rank;
    ST* gpu_row_ptr;
    IT* gpu_pre_rank;

    // backward view
    ST* gpu_col_ptr;
    IT* gpu_row_idx;
    IT* gpu_inv_idx;

    explicit CSRCMatrixCUDA<IT, ST>(const IT num_rows, const IT num_columns) : CSRCMatrix<IT, ST>(num_rows, num_columns) {
        gpu_post_rank = nullptr;
        gpu_row_ptr = nullptr;
        gpu_pre_rank = nullptr;

        gpu_col_ptr = nullptr;
        gpu_row_idx = nullptr;
        gpu_inv_idx = nullptr;
    }

    /**
     *  @brief      Destructor
     */
    ~CSRCMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDA::~CSRCMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<CSRCMatrix<IT, ST>*>(this)->clear();

        // clear device
        free_device_memory();
    }

    bool init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDA::init_matrix_from_lil() " << std::endl;
    #endif

        // host side
        bool success = static_cast<CSRCMatrix<IT, ST>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // copy to gpu
        return host_to_device_transfer();
    }

    bool fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // host side
        bool success = static_cast<CSRCMatrix<IT, ST>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);
        if (!success)
            return false;

        // copy to gpu
        return host_to_device_transfer();
    }

    //
    //  Variables
    //
    template <typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
        size_t size_in_bytes = this->num_non_zeros_*sizeof(VT);
        if (!check_free_device_memory(size_in_bytes)) {
            std::cerr << "Failed to allocate the GPU variable. Please check the available memory ..." << std::endl;
            return nullptr;
        }

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, size_in_bytes);
        cudaMemcpy(gpu_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

        return gpu_variable;
    }

    template <typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
        size_t size_in_bytes = this->post_ranks_.size() * sizeof(VT);
        if (!check_free_device_memory(size_in_bytes)) {
            std::cerr << "Failed to allocate the GPU variable. Please check the available memory ..." << std::endl;
            return nullptr;
        }

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, size_in_bytes);
        cudaMemcpy(gpu_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

        return gpu_variable;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(VT* gpu_variable) {
        auto host_tmp = std::vector<std::vector<VT>>();
        if (gpu_variable == nullptr)
            return host_tmp;

        auto flat_data = std::vector<VT>(this->num_non_zeros_, 0.0);
        cudaMemcpy(flat_data.data(), gpu_variable, this->num_non_zeros_*sizeof(VT), cudaMemcpyDeviceToHost);

        for (auto post_rk = this->post_ranks_.cbegin(); post_rk != this->post_ranks_.cend(); post_rk++) {
            host_tmp.push_back(std::vector<VT>(flat_data.begin()+this->row_begin_[*post_rk], flat_data.begin()+this->row_begin_[*post_rk+1]));
        }
        return host_tmp;
    }
};