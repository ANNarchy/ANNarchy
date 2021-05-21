/*
 *
 *    ELLRMatrixCUDA.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
template<typename IT=unsigned int>
class ELLRMatrixCUDA: public ELLMatrix<IT, false> {

    std::vector<IT> rl_;

    IT* gpu_post_ranks_;
    IT* gpu_col_idx_;
    IT* gpu_rl_;

public:

    ELLRMatrixCUDA<IT>(const IT num_rows, const IT num_columns) : ELLMatrix<IT, false>(num_rows, num_columns) {

    }

    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrixCUDA::init_matrix_from_lil()" << std::endl;
    #endif
        static_cast<ELLMatrix<IT, false>*>(this)->init_matrix_from_lil(post_ranks, pre_ranks);

    }

    //
    //  Init variables
    //
    template<typename VT>
    VT* init_matrix_variable_gpu(const std::vector<VT> &host_variable) {

        return nullptr;
    }

    template<typename VT>
    VT* init_vector_variable_gpu(const std::vector<VT> &host_variable) {
        
        return nullptr;
    }

    //
    // Read-out variables from GPU and return as LIL
    //
    template <typename VT>
    std::vector<std::vector<VT>> get_device_matrix_variable_as_lil(VT* gpu_variable) {
        auto tmp = std::vector<std::vector<VT>>();
        return tmp;
    }

    ~ELLRMatrixCUDA() {
        cudaFree(gpu_col_idx_);
        cudaFree(gpu_rl_);
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