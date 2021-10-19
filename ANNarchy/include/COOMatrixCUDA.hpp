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

    bool check_free_memory(size_t required) {
        size_t free, total;
        cudaMemGetInfo( &free, &total );
    #ifdef _DEBUG
        std::cout << "Allocate " << required << " and have " << free << "( " << (double(required)/double(total)) * 100.0 << " percent of total memory)" << std::endl;
    #endif
        return required < free;
    }

    void free_device_memory() {
        cudaFree(gpu_row_indices_);
        cudaFree(gpu_column_indices_);

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "COOMatrixCUDA::free_device_memory(): " << cudaGetErrorString(err) << std::endl;
    }

    bool host_to_device_transfer() {

        if(!check_free_memory(this->row_indices_.size()*sizeof(IT) + this->column_indices_.size()*sizeof(IT)))
            return true;

        cudaMalloc((void**)&gpu_row_indices_, this->row_indices_.size()*sizeof(IT));
        cudaMalloc((void**)&gpu_column_indices_, this->column_indices_.size()*sizeof(IT));

        cudaMemcpy(gpu_row_indices_, this->row_indices_.data(), this->row_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_column_indices_, this->column_indices_.data(), this->column_indices_.size()*sizeof(IT), cudaMemcpyHostToDevice);

        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "COOMatrixCUDA::host_to_device_transfer: " << cudaGetErrorString(err) << std::endl;
            return false;
        } else {
            return true;
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
    void clear() {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::clear()" << std::endl;
    #endif
        // clear host
        static_cast<COOMatrix<IT, ST>*>(this)->clear();

        // clear device
        free_device_memory();
    }


    inline IT* gpu_row_indices() {
        return gpu_row_indices_;
    }

    inline IT* gpu_column_indices() {
        return gpu_column_indices_;
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "COOMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif

        bool success = static_cast<COOMatrix<IT, ST>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);
        if (!success)
            return false;

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
        check_free_memory(required);

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
};