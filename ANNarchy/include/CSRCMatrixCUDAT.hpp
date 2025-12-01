/*
 *    CSRCMatrixCUDAT.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2022    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#include "CSRCMatrixT.hpp"

/**
 *  @brief      Implementation of the *compressed sparse row and column* format in transposed form on CUDA devices.
 *  @details    For more details on the format please refer to CSRCMatrixT.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class CSRCMatrixCUDAT: public CSRCMatrixT<IT, ST> {
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
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDAT::host_to_device()" << std::endl;
    #endif
        // CSR forward view
        if (gpu_post_rank) {
            cudaFree(gpu_post_rank);
            gpu_post_rank = nullptr;
        }
        if (gpu_row_ptr) {
            cudaFree(gpu_row_ptr);
            gpu_row_ptr = nullptr;
        }
        if (gpu_col_ptr) {
            cudaFree(gpu_col_ptr);
            gpu_col_ptr = nullptr;
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
            std::cerr << "CSRCMatrixCUDAT::free_device_memory: " << cudaGetErrorString(free_err) << std::endl;
        }
    }

    bool host_to_device_transfer() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDAT::host_to_device()" << std::endl;
    #endif
        //
        //  Free (maybe) existing allocations
        free_device_memory();

        // Sanity check: can we allocate the data?
        size_t req_size = sizeof(IT)*this->post_ranks_.size() + sizeof(ST)*this->row_ptr_.size() + sizeof(IT)*this->col_idx_.size() + this->col_ptr_.size()*sizeof(ST) + this->row_idx_.size()*sizeof(IT) + this->inv_idx_.size()*sizeof(IT);
        if (!check_free_device_memory(req_size))
            return false;

        //
        //  Allocate device memory
        cudaMalloc((void**)&gpu_post_rank, this->post_ranks_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_row_ptr, this->row_ptr_.size()*sizeof(ST));
        cudaMalloc((void**)&gpu_col_idx, this->col_idx_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_col_ptr, this->col_ptr_.size()*sizeof(ST));
        cudaMalloc((void**)&gpu_row_idx, this->row_idx_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_inv_idx, this->inv_idx_.size()*sizeof(IT));
        auto malloc_err = cudaGetLastError();
        if (malloc_err != cudaSuccess) {
            std::cerr << "CSRCMatrixCUDAT::init_matrix_from_lil - cudaMalloc: " << cudaGetErrorString(malloc_err) << std::endl;
            return false;
        }

        //
        // Copy data
        cudaMemcpy(gpu_post_rank, this->post_ranks_.data(), this->post_ranks_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_row_ptr, this->row_ptr_.data(), this->row_ptr_.size()*sizeof(ST), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_col_idx, this->col_idx_.data(), this->col_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_col_ptr, this->col_ptr_.data(), this->col_ptr_.size()*sizeof(ST), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_row_idx, this->row_idx_.data(), this->row_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_inv_idx, this->inv_idx_.data(), this->inv_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        auto copy_err = cudaGetLastError();
        if (copy_err != cudaSuccess) {
            std::cerr << "CSRCMatrixCUDAT::init_matrix_from_lil - cudaMemcpy: " << cudaGetErrorString(copy_err) << std::endl;
            return false;
        }

        return true;
    }

public:
    // CSR forward view
    IT* gpu_post_rank;
    ST* gpu_row_ptr;
    IT* gpu_col_idx;

    // backward view
    ST* gpu_col_ptr;
    IT* gpu_row_idx;
    IT* gpu_inv_idx;

    /**
     *  @brief      Constructor
     */
    explicit CSRCMatrixCUDAT<IT, ST>(const IT num_rows, const IT num_columns) : CSRCMatrixT<IT, ST>(num_rows, num_columns) {
        // CSR forward view
        gpu_post_rank = nullptr;
        gpu_row_ptr = nullptr;
        gpu_col_idx = nullptr;

        // backward view
        gpu_col_ptr = nullptr;
        gpu_row_idx = nullptr;
        gpu_inv_idx = nullptr;
    }

    /**
     *  @brief      Destructor
     */
    ~CSRCMatrixCUDAT() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDAT::~CSRCMatrixCUDAT()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() override {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDAT::clear()" << std::endl;
    #endif
        // clear host
        CSRCMatrixT<IT, ST>::clear();

        // clear device
        free_device_memory();
    }

    bool init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixCUDAT::init_matrix_from_lil() " << std::endl;
    #endif

        // host side
        bool success = static_cast<CSRCMatrixT<IT, ST>*>(this)->init_matrix_from_lil(row_indices, column_indices);
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
        if (gpu_variable == nullptr) {
            std::cerr << "CSRCMatrixT::get_device_matrix_variable_as_lil: device pointer has been invalid." << std::endl;
            return std::vector<std::vector<VT>>();
        }

        // transfer data from GPU
        auto flat_data = std::vector<VT>(this->num_non_zeros_, 0.0);
        cudaMemcpy(flat_data.data(), gpu_variable, this->num_non_zeros_*sizeof(VT), cudaMemcpyDeviceToHost);

        // convert to LIL using parent class implementation.
        auto host_tmp = (static_cast<CSRCMatrixT<IT, ST>*>(this))->get_matrix_variable_all(flat_data);
        return host_tmp;
    }
};