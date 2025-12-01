/*
 *    COOMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#include "COOMatrix.hpp"

/**
 *  @brief      Implementation of the *coordinate* format on CUDA devices.
 */
template<typename IT = unsigned int, typename ST = unsigned long int, IT SEGMENT_SIZE=32>
class COOMatrixCUDA: public COOMatrix<IT, ST> {

  protected:
    IT* gpu_row_indices_;
    IT* gpu_column_indices_;
    ST* gpu_segments_;

    std::vector<ST> segments_;

    void free_device_memory() {
        if (gpu_row_indices_) {
            cudaFree(gpu_row_indices_);
            gpu_row_indices_ = nullptr;
        }
        if (gpu_column_indices_) {
            cudaFree(gpu_column_indices_);
            gpu_column_indices_ = nullptr;
        }
        if (gpu_segments_) {
            cudaFree(gpu_segments_);
            gpu_segments_ = nullptr;
        }

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "COOMatrixCUDA::free_device_memory(): " << cudaGetErrorString(err) << std::endl;
    }

    bool host_to_device_transfer() {

        if(!check_free_memory_cuda(this->row_indices_.size()*sizeof(IT) + this->column_indices_.size()*sizeof(IT) + this->segments_.size() * sizeof(ST)))
            return true;

        cudaMalloc((void**)&gpu_row_indices_, this->row_indices_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_column_indices_, this->column_indices_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_segments_, this->segments_.size()*sizeof(ST));

        cudaMemcpy(gpu_row_indices_, this->row_indices_.data(), this->row_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_column_indices_, this->column_indices_.data(), this->column_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_segments_, this->segments_.data(), this->segments_.size()*sizeof(ST), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "COOMatrixCUDA::host_to_device_transfer: " << cudaGetErrorString(err) << std::endl;
            return false;
        } else {
            return true;
        }
    }

    /**
     *  @brief      Split the matrix in groups of SEGMENT_SIZE rows.
     *  @details    The method creates a helper array, which groups the nonzero values in groups. 
     *              The idea is, that we know which fraction of the large array contain to a fixed group of rows.
     */
    void compute_segments() {
        // Compute how the row indices are distributed
        auto tmp_segments = std::vector<ST>( ceil(double(this->num_rows_) / double(SEGMENT_SIZE)), 0 );
        for (ST i = 0; i < this->row_indices_.size(); i++) {
            IT chunk_idx = (this->row_indices_[i]/SEGMENT_SIZE);
            tmp_segments[chunk_idx]++;
        }

        // Determine the segment borders
        segments_ = std::vector<ST>(1,0);
        for (auto it = tmp_segments.begin(); it != tmp_segments.end(); it++) {
            segments_.push_back(segments_.back()+*it);
        }
    #ifdef _DEBUG
        std::cout << "Using " << segments_.size()-1 << " segments of size = " << SEGMENT_SIZE << std::endl;
        for (auto i = 0; i < segments_.size()-1; i++) {
            std::cout << "chunk[" << i << "]:" << segments_[i] << "-" << segments_[i+1] << " ( " << segments_[i+1] - segments_[i] << " entries)" << std::endl;
        }
        /*
        for (auto i = 0; i < segments_.size()-1; i++) {
            std::cout << "chunk[" << i*32 << "-" << (i+1)*32  << "] = ";
            for (auto j = segments_[i]; j < segments_[i+1]; j++ ) {
                std::cout << this->row_indices_[j] << " ";
            }
            std::cout << std::endl;
        }
        */
    #endif
    }

  public:
    explicit COOMatrixCUDA<IT, ST, SEGMENT_SIZE>(const IT num_rows, const IT num_columns) : COOMatrix<IT, ST>(num_rows, num_columns) {
        gpu_row_indices_ = nullptr;
        gpu_column_indices_ = nullptr;
        gpu_segments_ = nullptr;
    }

    COOMatrixCUDA<IT, ST, SEGMENT_SIZE>( COOMatrix<IT, ST>* other ) : COOMatrix<IT, ST>( other ) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::copy constructor"<< std::endl;
    #endif
        compute_segments();

        host_to_device_transfer();
    }

    /**
     *  @brief      Destructor
     */
    ~COOMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::~COOMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() override {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        COOMatrix<IT, ST>::clear();
        
        this->segments_.clear();
        this->segments_.shrink_to_fit();

        // clear device
        free_device_memory();
    }


    inline IT* gpu_row_indices() {
        return gpu_row_indices_;
    }

    inline IT* gpu_column_indices() {
        return gpu_column_indices_;
    }

    IT number_of_segments() {
        return this->segments_.size() - 1;
    }

    ST* gpu_segments() {
        return this->gpu_segments_;
    }

    IT segment_size() {
        return SEGMENT_SIZE;
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        bool success = static_cast<COOMatrix<IT, ST>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);
        if (!success)
            return false;

        compute_segments();

        return host_to_device_transfer();
    }

    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif

        assert( (this->row_indices_.size() == host_variable.size()) );

        // Sanity check
        size_t required = this->row_indices_.size()*sizeof(VT);
        check_free_memory_cuda(required);

        // Allocate and copy
        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, required);
        cudaMemcpy(gpu_variable, host_variable.data(), required, cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "COOMatrixCUDA::init_matrix_variable_gpu: " << cudaGetErrorString(err) << std::endl;
        }
        return gpu_variable;
    }

    template<typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(const VT* gpu_variable) {
        std::vector<std::vector<VT>> tmp;

        std::cout << "Not implemented ..." << std::endl;
        return tmp;
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() override {
        // standard coordinate size
        size_t size = COOMatrix<IT, ST>::size_in_bytes();

        // segments host container
        size += sizeof(std::vector<ST>);
        size += sizeof(this->segments_.capacity()*sizeof(ST));

        // GPU pointer
        size += 2 * sizeof(IT*);
        size += sizeof(ST*);

        return size;
    }
};