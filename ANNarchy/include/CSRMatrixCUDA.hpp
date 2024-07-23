/*
 *    CSRMatrixCUDA.hpp
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
 *  @brief      Implementation of the *compressed sparse row* format on CUDA devices.
 */
template<typename IT = unsigned int, typename ST = unsigned int>
class CSRMatrixCUDA: public CSRMatrix<IT, ST> {
protected:
    bool check_free_memory(size_t required) {
        size_t free, total;
        cudaMemGetInfo( &free, &total );
    #ifdef _DEBUG
        std::cout << "Allocate " << required << " and have " << free << "( " << (double(required)/double(total)) * 100.0 << " percent of total memory)" << std::endl;
    #endif
        return required < free;
    }

    void free_device_memory() {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::free_device_memory()" << std::endl;
    #endif
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

        auto free_err = cudaGetLastError();
        if (free_err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::free_device_memory: " << cudaGetErrorString(free_err) << std::endl;
        }
    }

    bool host_to_device() {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::host_to_device()" << std::endl;
    #endif
        //
        //  Free (maybe) existing allocations
        free_device_memory();

        // Sanity check: can we allocate the data?
        if (!check_free_memory(sizeof(IT)*this->post_ranks_.size() + sizeof(ST)*this->row_begin_.size() + sizeof(IT)*this->col_idx_.size()))
            return false;

        //
        //  Allocate device memory
        cudaMalloc((void**)&gpu_post_rank, this->post_ranks_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_row_ptr, this->row_begin_.size()*sizeof(ST));
        cudaMalloc((void**)&gpu_pre_rank, this->col_idx_.size()*sizeof(IT));
        auto malloc_err = cudaGetLastError();
        if (malloc_err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_from_lil - cudaMalloc: " << cudaGetErrorString(malloc_err) << std::endl;
            return false;
        }

        //
        //  Copy data
        cudaMemcpy(gpu_post_rank, this->post_ranks_.data(), this->post_ranks_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_row_ptr, this->row_begin_.data(), this->row_begin_.size()*sizeof(ST), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_pre_rank, this->col_idx_.data(), this->col_idx_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        auto copy_err = cudaGetLastError();
        if (copy_err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_from_lil - cudaMemcpy: " << cudaGetErrorString(copy_err) << std::endl;
            return false;
        }

        return true;
    }

public:
    ST* gpu_row_ptr;
    IT* gpu_post_rank;
    IT* gpu_pre_rank;

    CSRMatrixCUDA<IT, ST>(const IT num_rows, const IT num_columns) : CSRMatrix<IT, ST>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::CSRMatrixCUDA()" << std::endl;
    #endif
        gpu_row_ptr = nullptr;
        gpu_post_rank = nullptr;
        gpu_pre_rank = nullptr;
    }

    /**
     *  @brief      Destructor
     *  @details    responsible to delete the allocated GPU memory.
     */
    ~CSRMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::~CSRMatrixCUDA()" << std::endl;
    #endif
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<CSRMatrix<IT, ST>*>(this)->clear();

        // clear device
        free_device_memory();
    }

    bool init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::init_matrix_from_lil() " << std::endl;
    #endif

        // Initialization on host side
        bool success = static_cast<CSRMatrix<IT, ST>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // transfer to GPU
        return host_to_device();
    }

    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::fixed_number_pre_pattern()" << std::endl;
    #endif
        // Initialization on host side
        static_cast<CSRMatrix<IT, ST>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // transfer to GPU
        host_to_device();
    }

    void fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Initialization on host side
        static_cast<CSRMatrix<IT, ST>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // transfer to GPU
        host_to_device();
    }

    //
    //  Variables
    //
    template <typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif
        assert( (this->nb_synapses() == host_variable.size()) );
        size_t size_in_bytes = this->nb_synapses()*sizeof(VT);
        if (!check_free_memory(size_in_bytes)) {
            std::cerr << "Failed to allocate the GPU variable. Please check the available memory ..." << std::endl;
            return nullptr;
        }

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, size_in_bytes);
        cudaMemcpy(gpu_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_variable_gpu: " << cudaGetErrorString(err) << std::endl;
        }

        return gpu_variable;
    }

    template <typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::init_vector_variable_gpu()" << std::endl;
    #endif
        assert( (this->post_ranks_.size() == host_variable.size()) );
        size_t size_in_bytes = this->post_ranks_.size() * sizeof(VT);
        if(!check_free_memory(size_in_bytes)) {
            std::cerr << "Failed to allocate the GPU variable. Please check the available memory ..." << std::endl;
            return nullptr;
        }

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, size_in_bytes);
        cudaMemcpy(gpu_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_vector_variable_gpu: " << cudaGetErrorString(err) << std::endl;
        }

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

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        // standard compressed sparse row size
        size_t size = static_cast<CSRMatrix<IT, ST>*>(this)->size_in_bytes();

        // GPU pointer
        size += 2 * sizeof(IT*);
        size += sizeof(ST*);

        return size;
    }
};