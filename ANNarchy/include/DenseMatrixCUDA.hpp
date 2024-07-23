/*
 *    DenseMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                        Julien Vitay <julien.vitay@gmail.com>
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

#include "DenseMatrix.hpp"

/*
 *  @brief              Connectivity representation using a full matrix.
 *  @details            Contrary to all other classes in this template library this matrix format is not a sparse matrix.
 *  @tparam     IT      data type to represent the ranks within the matrix. Generally unsigned data types should be chosen.
 *                      The data type determines the maximum size for the number of elements in a column respectively the number
 *                      of rows encoded in the matrix:
 * 
 *                      - unsigned char (1 byte):        [0 .. 255]
 *                      - unsigned short int (2 byte):   [0 .. 65.535]
 *                      - unsigned int (4 byte):         [0 .. 4.294.967.295]
 *
 *                      The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and ::num_columns_)
 * 
 *              ST      the second type should be used if the index type IT could overflow. For instance, the nb_synapses method should return ST as
 *                      the maximum value in case a full dense matrix would be IT times IT entries.
 *              MT      We need to store if a matrix value is set in a bitmask. The size of each entry is determined by MT (we recommend char as its only 1 byte).
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major = false>
class DenseMatrixCUDA : public DenseMatrix<IT, ST, MT, row_major> {
protected:
    MT* gpu_mask_;

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
        std::cout << "DenseMatrixCUDA::free_device_memory()" << std::endl;
    #endif
        if (gpu_mask_) {
            cudaFree(gpu_mask_);
            gpu_mask_ = nullptr;
        }
        auto free_err = cudaGetLastError();
        if (free_err != cudaSuccess) {
            std::cerr << "DenseMatrixCUDA::free_device_memory: " << cudaGetErrorString(free_err) << std::endl;
        }
    }

    bool host_to_device() {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::host_to_device()" << std::endl;
    #endif
        //
        //  Allocate device memory
        cudaMalloc((void**)&gpu_mask_, this->mask_.size()*sizeof(MT));
        auto malloc_err = cudaGetLastError();
        if (malloc_err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_from_lil - cudaMalloc: " << cudaGetErrorString(malloc_err) << std::endl;
            return false;
        }

        //
        //  Copy data
        cudaMemcpy(gpu_mask_, this->mask_.data(), this->mask_.size()*sizeof(MT), cudaMemcpyHostToDevice);
        auto copy_err = cudaGetLastError();
        if (copy_err != cudaSuccess) {
            std::cerr << "CSRMatrixCUDA::init_matrix_from_lil - cudaMemcpy: " << cudaGetErrorString(copy_err) << std::endl;
            return false;
        }

        return true;
    }

public:

    explicit DenseMatrixCUDA(const IT num_rows, const IT num_columns): DenseMatrix<IT, ST, MT, row_major>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::DenseMatrixCUDA()" << std::endl;
    #endif
        gpu_mask_ = nullptr;
    }

    /**
     *  @brief      Destructor
     *  @details    calls the DenseMatrix::clear method. Is not declared as virtual as inheriting classes in our
     *              framework should never be destroyed by the base pointer.
     */
    ~DenseMatrixCUDA() {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::~DenseMatrixCUDA()" << std::endl;
    #endif
        clear();
    }

    inline MT* device_mask() {
        return this->gpu_mask_;
    }

    /**
     *  @brief      Clear the dense matrix.
     *  @details    Clears the connectivity data stored in the *post_rank* and *pre_rank* STL containers and free 
     *              the allocated memory. **Important**: allocated variables are not effected by this!
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::clear()" << std::endl;
    #endif
    }

    bool init_matrix_from_lil(std::vector<IT> &row_indices, std::vector< std::vector<IT> > &column_indices) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::init_matrix_from_lil() " << std::endl;
    #endif

        // Initialization on host side
        bool success = static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->init_matrix_from_lil(row_indices, column_indices);
        if (!success)
            return false;

        // transfer to GPU
        return host_to_device();
    }

    void fixed_probability_pattern(std::vector<int> post_ranks, std::vector<int> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrixCUDA::fixed_probability_pattern() " << std::endl;
    #endif
        // Initialization on host side
        static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // transfer to GPU
        host_to_device();
    }

    //
    //  Variables
    //
    template <typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::init_matrix_variable_gpu()" << std::endl;
    #endif
        size_t num_dense_elem = this->num_rows_ * this->num_columns_;
        assert( (num_dense_elem == host_variable.size()) );

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, num_dense_elem*sizeof(VT));
        cudaMemcpy(gpu_variable, host_variable.data(), num_dense_elem*sizeof(VT), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "DenseMatrixCUDA::init_matrix_variable_gpu: " << cudaGetErrorString(err) << std::endl;
        }

        return gpu_variable;
    }

    template <typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixCUDA::init_vector_variable_gpu()" << std::endl;
    #endif
        assert( (this->num_rows_ == host_variable.size()) );
        size_t size_in_bytes = this->num_rows_ * sizeof(VT);
        check_free_memory(size_in_bytes);

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, size_in_bytes);
        cudaMemcpy(gpu_variable, host_variable.data(), size_in_bytes, cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "DenseMatrixCUDA::init_vector_variable_gpu: " << cudaGetErrorString(err) << std::endl;
        }

        return gpu_variable;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(VT* gpu_variable) {
        auto host_tmp = std::vector<std::vector<VT>>();

        auto flat_data = std::vector<VT>(this->num_rows_*this->num_columns_, 0.0);
        cudaMemcpy(flat_data.data(), gpu_variable, this->num_rows_*this->num_columns_*sizeof(VT), cudaMemcpyDeviceToHost);

        return this->get_matrix_variable_all(flat_data);
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        size_t size = 2 * sizeof(IT);               // scalar values

        size += static_cast<DenseMatrix<IT, ST, MT, row_major>*>(this)->size_in_bytes();

        return size;
    }
};
