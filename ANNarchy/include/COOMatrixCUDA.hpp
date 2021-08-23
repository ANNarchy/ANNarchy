/*
 *
 *    COOMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/**
 *  @brief      Implementation of the *coordinate* format on CUDA devices.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class COOMatrixCUDA: public COOMatrix<IT, ST> {

  protected:
    IT* gpu_row_indices_;
    IT* gpu_column_indices_;

    void host_to_device_transfer() {
        cudaMalloc((void**)&gpu_row_indices_, this->row_indices_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_column_indices_, this->column_indices_.size()*sizeof(IT));

        cudaMemcpy(gpu_row_indices_, this->row_indices_.data(), this->row_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_column_indices_, this->column_indices_.data(), this->column_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "COOMatrixCUDA::host_to_device_transfer: " << cudaGetErrorString(err) << std::endl;
        }
    }

  public:
    explicit COOMatrixCUDA<IT, ST>(const IT num_rows, const IT num_columns) : COOMatrix<IT, ST>(num_rows, num_columns) {

    }

    COOMatrixCUDA<IT, ST>( COOMatrix<IT, ST>* other ) : COOMatrix<IT, ST>( other ) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::copy constructor"<< std::endl;
    #endif
        host_to_device_transfer();
    }

    inline IT* gpu_row_indices() {
        return gpu_row_indices_;
    }

    inline IT* gpu_column_indices() {
        return gpu_column_indices_;
    }

    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        static_cast<COOMatrix<IT, ST>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);

        host_to_device_transfer();
    }

    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {
        assert( (this->row_indices_.size() == host_variable.size()) );

        VT* gpu_variable;
        cudaMalloc((void**)&gpu_variable, this->row_indices_.size()*sizeof(VT));
        cudaMemcpy(gpu_variable, host_variable.data(), this->row_indices_.size()*sizeof(VT), cudaMemcpyHostToDevice);

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
};