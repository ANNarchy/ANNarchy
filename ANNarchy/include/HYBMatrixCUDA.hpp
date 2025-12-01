/*
 *    HYBMatrixCUDA.hpp
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

#include "HYBMatrix.hpp"
#include "COOMatrixCUDA.hpp"
#include "ELLMatrixCUDA.hpp"

/**
 *  As the hybrid format is a combination of two formats, one can not simply
 *  express a variable as one single container.
 */
template<typename VT>
struct hyb_local_gpu {
    VT* ell;
    VT* coo;

    hyb_local_gpu() {
        ell = nullptr;
        coo = nullptr;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "hyb_local_gpu::clear()" << std::endl;
    #endif
        cudaFree(ell);
        cudaFree(coo);
        ell = nullptr;
        coo = nullptr;
    }

    ~hyb_local_gpu() {
    }
};

/**
 *  @brief      Implementation of the hybrid (HYB) sparse matrix format for GPUs using CUDA.
 *  @details    This class extends the HYBMatrix class by the necessary codes for the usage on CUDA-capable
 *              devices. Please note, that the default second parameter of HYBMatrix, row- or column-major
 *              is set to false, as a row-major encoded ELLPACK matrix would make no sense on CUDA devices.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class HYBMatrixCUDA: public HYBMatrix<IT, ST, false>
{
  protected:
    ELLMatrixCUDA<IT, ST> *ell_matrix_gpu;
    COOMatrixCUDA<IT, ST> *coo_matrix_gpu;

  public:
    explicit HYBMatrixCUDA(const IT num_rows, const IT num_columns): HYBMatrix<IT, ST, false>(num_rows, num_columns) {
    #ifdef _DEBUG
        std::cout << "HYBMatrixCUDA::HYBMatrixCUDA()" << std::endl;
    #endif
    }

    void clear() override{
    #ifdef _DEBUG
        std::cout << "HYBMatrixCUDA::clear()" << std::endl;
    #endif
        // call clear of partial matrices
        ell_matrix_gpu->clear();
        coo_matrix_gpu->clear();
    }

    ELLMatrixCUDA<IT, ST>* get_ell() {
        return ell_matrix_gpu;
    }

    COOMatrixCUDA<IT, ST>* get_coo() {
        return coo_matrix_gpu;
    }

    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices, unsigned int ell_size=std::numeric_limits<unsigned int>::max()) {
    #ifdef _DEBUG
        std::cout << "HYBMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif
        // Create matrix on host-side
        bool success = static_cast<HYBMatrix<IT, ST, false>*>(this)->init_matrix_from_lil(row_indices, column_indices, ell_size);
        if (!success)
            return false;

        // store sizes for verification
        auto ell_nb_synapses = static_cast<HYBMatrix<IT, ST, false>*>(this)->get_ell_instance()->nb_synapses();
        auto coo_nb_synapses = static_cast<HYBMatrix<IT, ST, false>*>(this)->get_coo_instance()->nb_synapses();

        // Initialize GPU side
        ell_matrix_gpu = new ELLMatrixCUDA<IT, ST>(static_cast<HYBMatrix<IT, ST, false>*>(this)->get_ell_instance());
        coo_matrix_gpu = new COOMatrixCUDA<IT, ST>(static_cast<HYBMatrix<IT, ST, false>*>(this)->get_coo_instance());
        
        // Re-assign host side pointer: they will first destroy the already existing instances and then set the
        // new pointers
        this->replace_pointer( static_cast<ELLMatrix<IT, ST, false>*>(ell_matrix_gpu), static_cast<COOMatrix<IT, ST>*>(coo_matrix_gpu) );

        // verify
        assert( (ell_nb_synapses == static_cast<HYBMatrix<IT, ST, false>*>(this)->get_ell_instance()->nb_synapses()) );
        assert( (coo_nb_synapses == static_cast<HYBMatrix<IT, ST, false>*>(this)->get_coo_instance()->nb_synapses()) );

        return true;
    }

    template<typename VT>
    hyb_local_gpu<VT>* init_matrix_variable_gpu(const hyb_local<VT>* host_variable) {
        auto new_variable = new hyb_local_gpu<VT>();

        new_variable->ell = ell_matrix_gpu->init_matrix_variable_gpu(host_variable->ell);
        new_variable->coo = coo_matrix_gpu->init_matrix_variable_gpu(host_variable->coo);
        
        return new_variable;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(hyb_local_gpu<VT> gpu_variable) {
        auto tmp = std::vector<std::vector<VT>>();
        return tmp;
    }

    size_t size_in_bytes() override {
        size_t size = 0;

        size += static_cast<ELLMatrixCUDA<IT, ST>*>(ell_matrix_gpu)->size_in_bytes();
        size += static_cast<COOMatrixCUDA<IT, ST>*>(coo_matrix_gpu)->size_in_bytes();

        size += 2*sizeof(void*);        // 2 class references

        return size;
    }
};
