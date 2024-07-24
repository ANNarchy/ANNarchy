/*
 *    SELLMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *	  Copyright (C) 2021-22  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                  2021-22  Qi Tang <kevin2014tq@gmail.com>
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
#include "SELLMatrix.hpp"

template<typename IT = unsigned int, typename ST = unsigned long int>
class SELLMatrixCUDA : public SELLMatrix<IT, ST, false> {
  protected:
    bool check_free_memory(size_t required) {
          size_t free, total;
          cudaMemGetInfo(&free, &total);

    #ifdef _DEBUG
          std::cout << "Allocate " << required << " and have " << free << "( " << (double(required) / double(total)) * 100.0 << " percent of total memory)" << std::endl;
    #endif
        return (required < free);
    }

    void free_device_memory() {
        if(d_row_ptr) {
            cudaFree(d_row_ptr);
            d_row_ptr = nullptr;
        }
        if (d_col_idx) {
            cudaFree(d_col_idx);
            d_col_idx = nullptr;
        }

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "SELLMatrixCUDA::free_device_memory(): " << cudaGetErrorString(err) << std::endl;
    }

    bool host_to_device_transfer() {
        //compute memory require
        size_t row_ptr_size = this->row_ptr_.size() * sizeof(ST);
        size_t col_idx_size = this->col_idx_.size() * sizeof(IT);

        // Sanity check: can we allocate the data?
        check_free_memory(row_ptr_size + col_idx_size);

        // Allocate the data arrays
        cudaMalloc((void**)&d_row_ptr, row_ptr_size);
        cudaMalloc((void**)&d_col_idx, col_idx_size);

        // Copy the data arrays
        cudaMemcpy(d_row_ptr, this->row_ptr_.data(), row_ptr_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, this->col_idx_.data(), col_idx_size, cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "ELLMatrixCUDA::host_to_device_transfer(): " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        return true;
    }

  public:
    ST* d_row_ptr;
    IT* d_col_idx;


    /**
     *  Default constructor
     */
    explicit SELLMatrixCUDA<IT,ST>(const IT num_rows, const IT num_columns, const IT block_size) : SELLMatrix<IT,ST, false>(num_rows, num_columns, block_size) {
    #ifdef _DEBUG
        std::cout << "SELLMatrixCUDA::SELLMatrixCUDA()" << std::endl;
    #endif
        d_row_ptr = nullptr;
        d_col_idx = nullptr;
    }

    /**
     *  Initialize host side with other sliced ELLPACK instance (host side)
     */
    /*SELLMatrixCUDA<IT,ST>(SELLMatrix<IT, ST, false>* other) : SELLMatrix<IT,ST, false>(other) {
    #ifdef _DEBUG
            std::cout << "SELLMatrixCUDA::copy constructor" << std::endl;
    #endif
        host_to_device_transfer();
    }*/

    /**
     *  @brief      Destructor
     */
    ~SELLMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "SELLMatrixCUDA::~SELLMatrixCUDA()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the matrix
     *  @details    should be called before destructor.
     */
    void clear() {
    #ifdef _DEBUG
            std::cout << "SELLMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<SELLMatrix<IT, ST, false>*>(this)->clear();

        // clear device
        free_device_memory();
    }


    /*
    *   init matrix from lil format  
    */
    bool init_matrix_from_lil(std::vector<IT>& post_ranks, std::vector< std::vector<IT> >& pre_ranks) {
        assert((post_ranks.size() == pre_ranks.size()));
        assert((post_ranks.size() > 0));

    #ifdef _DEBUG
            std::cout << "SELLMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        // Initialize on host
        static_cast<SELLMatrix<IT, ST, false>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);

        // transfer to device
        return host_to_device_transfer();
    }

    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
            std::cout << "SELLMatrixCUDA::fixed_number_pre_pattern()" << std::endl;
    #endif
        // Initialization on host side
        static_cast<SELLMatrix<IT, ST, false>*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // transfer to GPU
        host_to_device_transfer();

        return true;
    }

    /*
    void fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
            std::cout << "SELLMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Initialization on host side
        static_cast<SELLMatrix<IT, false>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // transfer to GPU
        host_to_device_transfer();
    }  */


    //
    //  Init variables
    //
    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT>& host_variable) {
    #ifdef _DEBUG
            std::cerr << "SELLMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif
        size_t size_in_bytes = host_variable.size() * sizeof(VT);
        // sanity check
        if (!check_free_memory(size_in_bytes))
            return nullptr;

        // Allocate
        VT* new_variable;
        cudaMalloc((void**)&new_variable, size_in_bytes);
    #ifdef _DEBUG
        auto malloc_err = cudaGetLastError();
        if (malloc_err != cudaSuccess)
            std::cerr << "SELLMatrixCUDA::init_matrix_variable_gpu<>() - allocate: " << cudaGetErrorString(malloc_err) << std::endl;
    #endif

        // Copy
        cudaMemcpy(new_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

    #ifdef _DEBUG
        auto memcpy_err = cudaGetLastError();
        if (memcpy_err != cudaSuccess)
            std::cerr << "SELLMatrixCUDA::init_matrix_variable_gpu<>() - memcpy: " << cudaGetErrorString(memcpy_err) << std::endl;
    #endif
        return new_variable;
    }

    // Init vector variable
    template<typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT>& host_variable) {
        size_t size_in_bytes = host_variable.size() * sizeof(VT);
        // sanity check
        check_free_memory(size_in_bytes);

        // Allocate
        VT* new_variable;
        cudaMalloc((void**)&new_variable, host_variable.size() * sizeof(VT));

        // Copy
        cudaMemcpy(new_variable, host_variable.data(), host_variable.size() * sizeof(VT), cudaMemcpyHostToDevice);
        return new_variable;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    /*template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(VT* gpu_variable) {
        auto tmp = std::vector<std::vector<VT>>();
        return tmp;
    }*/

    template<typename VT>
    void get_vector_variable_from_gpu(std::vector<VT>& host_variable, VT* d_variable) {
        size_t size_in_bytes = host_variable.size() * sizeof(VT);

        // Copy
        cudaMemcpy(host_variable.data(), d_variable, size_in_bytes, cudaMemcpyDeviceToHost);
    }

    IT* get_device_col_idx() {
        return d_col_idx;
    }

    /**
     *  \brief      overloaded std::ostream operator<<
     *  \details    for the object itself
     *  \param[IN]  os      ostream instance
     *  \param[IN]  matrix  object instance
     *  \return     manipulated ostream instance
     */
    friend std::ostream& operator<< (std::ostream& os, const SELLMatrixCUDA<IT>& matrix) {
        os << "num_rows_: " << matrix.num_rows_ << std::endl;
        os << "blocksize_: " << matrix.blocksize_ << std::endl;
        os << "num_blocks_: " << matrix.num_blocks_ << std::endl;
        os << "num_non_zeros_: " << matrix.num_non_zeros_ << std::endl;

        os << "col_idx_:" << std::endl;
        os << "[ ";
        for (int s = 0; s < matrix.col_idx_.size(); s++) {
            os << matrix.col_idx_[s] << " ";
        }
        os << "]" << std::endl;

        os << "post_ranks_:" << std::endl;
        os << "[ ";
        for (int s = 0; s < matrix.post_ranks_.size(); s++) {
            os << matrix.post_ranks_[s] << " ";
        }
        os << "]" << std::endl;

        os << "rowptr_:" << std::endl;
        os << "[ ";
        for (int s = 0; s < matrix.rowptr_.size(); s++) {
            os << matrix.rowptr_[s] << " ";
        }
        os << "]" << std::endl;

        return os;
    }

    /**
     *  \brief      overloaded std::ostream operator<<
     *  \details    for the reference to an object
     *  \param[IN]  os      ostream instance
     *  \param[IN]  matrix  object reference
     *  \return     manipulated ostream instance
     */
    friend std::ostream& operator<< (std::ostream& os, SELLMatrixCUDA<IT>* matrix) {
        return os << *matrix;
    }

};
