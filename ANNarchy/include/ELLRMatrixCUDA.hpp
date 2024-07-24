/*
 *    ELLRMatrixCUDA.hpp
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

#include "ELLRMatrix.hpp"

/**
 *  @brief      A variation of the ELLPACK format intended for the usage on GPUs.
 *  @details    There are two major changes: first the dense matrix is stored as column-major
 *              representation and secondly a row-length array is introduced. Both together
 *              should improve memory access pattern and reduce branching.
 *
 *              A detailed description can be found in:
 * 
 *                  Vasquez et al. (2009) The sparse matrix vector product on GPUs.
 *                  Vasquez et al. (2011) A new approach for sparse matrix vector product on NVIDIA GPUs
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class ELLRMatrixCUDA: public ELLRMatrix<IT, ST, false> {
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
        if (gpu_rl_) {
            cudaFree(gpu_rl_);
            gpu_rl_ = nullptr;
        }

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "ELLRMatrixCUDA::free_device_memory(): " << cudaGetErrorString(err) << std::endl;
    }

    bool host_to_device_transfer() {
        //
        //  Free (maybe) existing allocations
        free_device_memory();

        // Sanity check: can we allocate the data?
        if(!check_free_memory(sizeof(IT)*this->post_ranks_.size() + sizeof(IT)*this->col_idx_.size() + sizeof(IT)*this->rl_.size()))
            return false;

        // Allocate the data arrays
        cudaMalloc((void**)& gpu_post_ranks_, sizeof(IT)*this->post_ranks_.size());
        cudaMalloc((void**)& gpu_col_idx_, sizeof(IT)*this->col_idx_.size());
        cudaMalloc((void**)& gpu_rl_, sizeof(IT)*this->rl_.size());

        // Copy the data arrays
        cudaMemcpy(gpu_post_ranks_, this->post_ranks_.data(), sizeof(IT)*this->post_ranks_.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_col_idx_, this->col_idx_.data(), sizeof(IT)*this->col_idx_.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_rl_, this->rl_.data(), sizeof(IT)*this->rl_.size(), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "ELLRMatrixCUDA::host_to_device_transfer(): " << cudaGetErrorString(err) << std::endl;
            return false;
        } else {
            return true;
        }
    }

public:
    IT* gpu_post_ranks_;
    IT* gpu_col_idx_;
    IT* gpu_rl_;

    /**
     *  Default constructor
     */
    explicit ELLRMatrixCUDA<IT, ST>(const IT num_rows, const IT num_columns) : ELLRMatrix<IT, ST, false>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::ELLRMatrixCUDA()" << std::endl;
    #endif
        gpu_post_ranks_ = nullptr;
        gpu_col_idx_ = nullptr;
        gpu_rl_ = nullptr;
    }

    /**
     *  Initialize host side with other ELLPACK-R instance (host side)
     */
    ELLRMatrixCUDA<IT, ST>( ELLRMatrix<IT, ST, false>* other ) : ELLRMatrix<IT, ST, false>( other ) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::copy constructor"<< std::endl;    
    #endif
        host_to_device_transfer();
    }

    /**
     *  @brief      Destructor
     */
    ~ELLRMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::~ELLRMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<ELLRMatrix<IT, ST, false>*>(this)->clear();

        // clear device
        free_device_memory();
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        // Sanity check
        assert( (post_ranks.size() == pre_ranks.size()) );
        assert( (post_ranks.size() > 0) );

        // Initialize on host
        bool success = static_cast<ELLRMatrix<IT, ST, false>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);
        if (!success) {
            std::cerr << "ELLRMatrixCUDA::init_matrix_from_lil(): host side construction failed." << std::endl;
            return false;
        }

        // Transfer to device
        return host_to_device_transfer();
    }

    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::fixed_number_pre_pattern()" << std::endl;
    #endif
        // Initialization on host side
        bool success = static_cast<ELLRMatrix<IT, ST, false>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);
        if (!success) {
            std::cerr << "ELLRMatrixCUDA::fixed_number_pre_pattern(): host side construction failed." << std::endl;
            return false;
        }

        // transfer to GPU
        return host_to_device_transfer();
    }

    bool fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Initialization on host side
        bool success = static_cast<ELLRMatrix<IT, ST, false>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);
        if (!success) {
            std::cerr << "ELLRMatrixCUDA::fixed_probability_pattern(): host side construction failed." << std::endl;
            return false;
        }

        // transfer to GPU
        return host_to_device_transfer();
    }

    //
    //  Init variables
    //
    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
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
            std::cerr << "ELLRMatrixCUDA::init_matrix_variable_gpu<>(): " << cudaGetErrorString(err) << std::endl;
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

    IT* get_device_rl() {
        return gpu_rl_;
    }

    /**
     *  \brief      overloaded std::ostream operator<<
     *  \details    for the object itself
     *  \param[IN]  os      ostream instance
     *  \param[IN]  matrix  object instance
     *  \return     manipulated ostream instance
     */
     friend std::ostream& operator<< (std::ostream& os, const ELLRMatrixCUDA<IT>& matrix) {
        os << "num_rows_: " << matrix.num_rows_ << std::endl;
        os << "maxnzr_: " << matrix.maxnzr_ << std::endl;
        
        os << "col_idx_:" << std::endl;
        for(int r = 0; r < matrix.num_rows_; r++) {
            os << "[ ";
            for(int s = 0; s < matrix.maxnzr_; s++) {
                os << matrix.col_idx_[s * matrix.num_rows_ + r] << " ";
            }
            os << "]" << std::endl;
        }
        os << "one array - col_idx_:" << std::endl;
        os << "[ ";
        for(int s = 0; s < matrix.col_idx_.size(); s++) {
            os << matrix.col_idx_[s] << " ";
        }
        os << "]" << std::endl;
        os << "rl:" << std::endl;
        os << "[ ";
        for(int s = 0; s < matrix.rl_.size(); s++) {
            os << matrix.rl_[s] << " ";
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
    friend std::ostream& operator<< (std::ostream& os, ELLRMatrixCUDA<IT>* matrix) {
        return os << *matrix;
    }
};