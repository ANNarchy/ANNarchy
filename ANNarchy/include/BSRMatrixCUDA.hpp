/*
 *    BSRMatrixCUDA.hpp
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

#include "BSRMatrix.hpp"

/**
 *  \brief      Implementation of the blocked sparse row (BSR) format for GPUs
 *  \details    The format has been described in detail, e.g. in:
 * 
 *              * Vershoor and Jalba (2012): Analysis and performance estimation of the Conjugate Gradient method on multiple GPUs
 *              * Eberhardt & Hoemmen (2016): Optimization of block sparse matrix-vector multiplication on shared-memory parallel architectures
 *              * Benetia et al. 2018 (2018): BestSF: A Sparse Meta-Format for Optimizing SpMV on GPU
 *              * NVIDIA Corporation: https://docs.nvidia.com/cuda/cusparse/index.html
 * 
 *	\tparam 	IT		    index data type
 *	\tparam 	VT		    value data type

 *  \note       The current implementation forces *char* as mask type for CUDA devices. Also the storage scheme is forced to be column major.
 */
template<typename IT=unsigned int, typename ST=unsigned long int>
class BSRMatrixCUDA: public BSRMatrix<IT, ST, char, false> 
{
    IT* gpu_block_row_pointer_;
    IT* gpu_block_column_index_;
    char* gpu_tile_mask_;

    void free_device_memory() {
        if (gpu_block_row_pointer_) {
            cudaFree(gpu_block_row_pointer_);
            gpu_block_row_pointer_ = nullptr;
        }
        if (gpu_block_column_index_) {
            cudaFree(gpu_block_column_index_);
            gpu_block_column_index_ = nullptr;
        }
        if (gpu_tile_mask_) {
            cudaFree(gpu_tile_mask_);
            gpu_tile_mask_ = nullptr;
        }
    }

    bool transfer_to_device() {
        // Allocate 
        cudaMalloc((void**)&gpu_block_row_pointer_, this->block_row_pointer_.size() * sizeof(IT));
        cudaMalloc((void**)&gpu_block_column_index_, this->block_column_index_.size() * sizeof(IT));
        cudaMalloc((void**)&gpu_tile_mask_, this->tile_mask_.size()*sizeof(char));
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "BCSRMatrixCUDA::transfer_to_device():" << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // Transfer
        cudaMemcpy(gpu_block_row_pointer_, this->block_row_pointer_.data(), this->block_row_pointer_.size() * sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_block_column_index_, this->block_column_index_.data(), this->block_column_index_.size() * sizeof(IT), cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "BCSRMatrixCUDA::transfer_to_device():" << cudaGetErrorString(err) << std::endl;
            return false;
        }

        return true;
    }

public:
    BSRMatrixCUDA(const unsigned int num_rows, const unsigned int num_columns, const unsigned int block_size) :
        BSRMatrix<IT, ST, char, false>(num_rows, num_columns, block_size) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixCUDA::BSRMatrixCUDA()" << std::endl;
    #endif
            gpu_block_row_pointer_ = nullptr;
            gpu_block_column_index_ = nullptr;
            gpu_tile_mask_ = nullptr;
        }

    /**
     *  @brief      Destructor
     *  @details    Please note, the clear() method should be called in advance.
     */
    ~BSRMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "BSRMatrixCUDA::~BSRMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the BSR matrix
     *  @details    responsible to delete the allocated GPU memory.
     */
    void clear() override {
    #ifdef _DEBUG
        std::cout << "BSRMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        BSRMatrix<IT, ST, char, false>::clear();

        // clear device
        free_device_memory();
    }

    void load_from_file(std::string filename=std::string("mat.txt"), bool measure_time = true) {
        static_cast<BSRMatrix<IT, ST, char, false>*>(this)->load_from_file(filename, measure_time);

        transfer_to_device();
    }

    inline IT* gpu_block_row_pointer() {
        return gpu_block_row_pointer_;
    }

    inline IT* gpu_block_column_index() {
        return gpu_block_column_index_;
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        bool success = static_cast<BSRMatrix<IT, ST, char, false>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);
        if (!success)
            return false;

        size_t required = this->block_row_pointer_.size() * sizeof(IT) + this->block_column_index_.size() * sizeof(IT) + this->tile_mask_.size()*sizeof(char);
        if( !check_free_memory_cuda(required) )
            return false;

        return transfer_to_device();
    }

    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "BCSRMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif
        assert( this->tile_mask_.size() == host_variable.size() );

        size_t required = host_variable.size() * sizeof(VT);
        if (!check_free_memory_cuda(required))
            return nullptr;

        // Allocate and copy
        VT* gpu_variable;
        auto malloc_err = cudaMalloc((void**)&gpu_variable, required);
        if ( malloc_err != cudaSuccess ) {
            std::cout << "BCSRMatrixCUDA::init_matrix_variable_gpu():" << cudaGetErrorString(malloc_err) << std::endl;
            return nullptr;
        }

        auto transfer_err = cudaMemcpy(gpu_variable, host_variable.data(), required, cudaMemcpyHostToDevice);
        if ( transfer_err != cudaSuccess ) {
            std::cout << "BCSRMatrixCUDA::init_matrix_variable_gpu():" << cudaGetErrorString(transfer_err) << std::endl;
            cudaFree(gpu_variable);
            return nullptr;
        }

        return gpu_variable;
    }

    template<typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(const VT* gpu_variable) {
        std::vector<std::vector<VT>> tmp;

        std::cout << "Not implemented ..." << std::endl;
        return tmp;
    }
};
