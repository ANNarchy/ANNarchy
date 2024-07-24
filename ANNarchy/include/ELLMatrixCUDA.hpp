/*
 *    ELLMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020-2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

#include "ELLMatrix.hpp"

/**
 *  @brief      An implementation of the ELLPACK format intended for the usage on GPUs.
 *  @details    This implementation is intended as part of the hybrid ELLPACK/Coordinate
 *              format. For single use, we intended the usage of the modified
 *              ELLPACK-R format.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class ELLMatrixCUDA: public ELLMatrix<IT, ST, false> {
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
        if (gpu_post_ranks_) {
            cudaFree(gpu_post_ranks_);
            gpu_post_ranks_ = nullptr;
        }
        if (gpu_col_idx_) {
            cudaFree(gpu_col_idx_);
            gpu_col_idx_ = nullptr;
        }

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "ELLMatrixCUDA::free_device_memory(): " << cudaGetErrorString(err) << std::endl;
    }

    bool host_to_device_transfer() {
        // Sanity check: can we allocate the data?
        if (!check_free_memory(sizeof(IT)*this->post_ranks_.size() + sizeof(IT)*this->col_idx_.size()))
            return false;

        // Allocate the data arrays
        cudaMalloc((void**)& gpu_post_ranks_, sizeof(IT)*this->post_ranks_.size());
        cudaMalloc((void**)& gpu_col_idx_, sizeof(IT)*this->col_idx_.size());

        // Copy the data arrays
        cudaMemcpy(gpu_post_ranks_, this->post_ranks_.data(), sizeof(IT)*this->post_ranks_.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_col_idx_, this->col_idx_.data(), sizeof(IT)*this->col_idx_.size(), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "ELLMatrixCUDA::host_to_device_transfer(): " << cudaGetErrorString(err) << std::endl;
            return false;
        }else{
            return true;
        }
    }

public:
    IT* gpu_post_ranks_;
    IT* gpu_col_idx_;

    /**
     *  Default constructor
     */
    explicit ELLMatrixCUDA<IT, ST>(const IT num_rows, const IT num_columns) : ELLMatrix<IT, ST, false>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::ELLMatrixCUDA()" << std::endl;
    #endif
        gpu_post_ranks_ = nullptr;
        gpu_col_idx_ = nullptr;
    }

    /**
     *  Initialize host side with other ELLPACK-R instance (host side)
     */
    ELLMatrixCUDA<IT, ST>( ELLMatrix<IT, ST, false>* other ) : ELLMatrix<IT, ST, false>( other ) {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::copy constructor"<< std::endl;    
    #endif
        host_to_device_transfer();
    }

    /**
     *  @brief      Destructor
     */
    ~ELLMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::~ELLMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<ELLMatrix<IT, ST, false>*>(this)->clear();

        // clear device
        free_device_memory();
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
        assert( (post_ranks.size() == pre_ranks.size()) );
        assert( (post_ranks.size() > 0) );

    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        // Initialize on host
        bool success = static_cast<ELLMatrix<IT, ST, false>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);
        if(!success)
            return false;

        // Initialize on device and transfer data
        return host_to_device_transfer();
    }

    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::fixed_number_pre_pattern()" << std::endl;
    #endif
        // Initialization on host side
        static_cast<ELLMatrix<IT, ST, false>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // transfer to GPU
        host_to_device_transfer();
    }

    void fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Initialization on host side
        static_cast<ELLMatrix<IT, ST, false>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // transfer to GPU
        host_to_device_transfer();
    }

    //
    //  Init variables
    //
    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cerr << "ELLMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif
        size_t size_in_bytes = host_variable.size() * sizeof(VT);
        // sanity check
        check_free_memory(size_in_bytes);

        // Allocate
        VT* new_variable;
        cudaMalloc((void**)& new_variable, size_in_bytes);

        // Copy
        cudaMemcpy(new_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "ELLMatrixCUDA::init_matrix_variable_gpu<>(): " << cudaGetErrorString(err) << std::endl;
    #endif
        return new_variable;
    }

    template<typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
        size_t size_in_bytes = host_variable.size() * sizeof(VT);
        // sanity check
        check_free_memory(size_in_bytes);

        // Allocate
        VT* new_variable;
        cudaMalloc((void**)& new_variable, host_variable.size() * sizeof(VT));

        // Copy
        cudaMemcpy(new_variable, host_variable.data(), host_variable.size() * sizeof(VT), cudaMemcpyHostToDevice);
        return new_variable;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(VT* gpu_variable) {
        auto tmp = std::vector<std::vector<VT>>();
        return tmp;
    }
    
    IT* get_device_col_idx() {
        return gpu_col_idx_;
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        // standard ELLPACK size
        size_t size = static_cast<ELLMatrix<IT, ST, false>*>(this)->size_in_bytes();

        // GPU pointer
        size += 2 * sizeof(IT*);

        return size;
    }

    /**
     *  \brief      overloaded std::ostream operator<<
     *  \details    for the object itself
     *  \param[IN]  os      ostream instance
     *  \param[IN]  matrix  object instance
     *  \return     manipulated ostream instance
     */
     friend std::ostream& operator<< (std::ostream& os, const ELLMatrixCUDA<IT>& matrix) {
        os << "num_rows_: " << matrix.num_rows_ << std::endl;
        os << "maxnzr_: " << matrix.maxnzr_ << std::endl;
        
        os << "col_idx_:" << std::endl;
        os << "[ ";
        for(int s = 0; s < matrix.col_idx_.size(); s++) {
            os << matrix.col_idx_[s] << " ";
        }
        os << "]" << std::endl;
        os << "values_:" << std::endl;
        for(int r = 0; r < matrix.num_rows_; r++) {
            os << "[ ";
            for(int s = 0; s < matrix.maxnzr_; s++) {
                os << matrix.values_[s * matrix.num_rows_ + r] << " ";
            }
            os << "]" << std::endl;
        }
        return os;
    }

    /**
     *  \brief      overloaded std::ostream operator<<
     *  \details    for the reference to an object
     *  \param[IN]  os      ostream instance
     *  \param[IN]  matrix  object reference
     *  \return     manipulated ostream instance
     */
    friend std::ostream& operator<< (std::ostream& os, ELLMatrixCUDA<IT>* matrix) {
        return os << *matrix;
    }
};