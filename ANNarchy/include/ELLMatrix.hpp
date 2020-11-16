/*
 * ELLMatrix.hpp
 *
 * Copyright (c) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

/**
 *  \brief      ELLPACK sparse matrix representation according to Kincaid et al. 1989 with some
 *              minor modifications as described below.
 * 
 *  \details    The ELLPACK format encodes the nonzeros of a sparse matrix in dense matrices
 *              where one represent the column indices and another one for each variable.
 *
 *              Let's consider the following example matrix
 * 
 *                      | 0 1 0 |
 *                  A = | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 *
 *              First we need to determine the maximum number of column-entries per row (we
 *              call this *maxnzr*), in our example 2.
 * 
 *              Then we read out the column entries and fill the created dense matrix from
 *              left. Important is the encoding of non-existing entries some authors suggest -1
 *              but this would require every time a check if a contained index is valid. We
 *              fill non-existing places with 0.
 * 
 *                              | 1 0 |
 *                  col_idx_ =  | 0 2 |
 *                              | 2 0 |
 * 
 *              To allow learning on the matrix and encoding of 0 as existing value, we also
 *              introduce a row-length array (rl):
 * 
 *                  rl_ = [ 1, 2, 1 ]
 * 
 *              As for LILMatrix and others one need to highlight that rows with no nonzeros are
 *              compressed.
 *
 *  \tparam     IT          index type
 *  \tparam     row_major   determines the matrix storage for the dense sub matrices. If
 *                          set to true, the matrix will be stored as row major, otherwise
 *                          in column major. 
 *                          Please note that the original format stores in row-major to ensure a
 *                          partial caching of data on CPUs. The column-major ordering is only
 *                          intended for the usage on GPUs.
 */
template<typename IT=unsigned int, bool row_major=true>
class ELLMatrix {
protected:
    IT maxnzr_;                     ///< maximum row length of nonzeros
    std::vector<IT> post_ranks_;    ///< which rows does contain entries
    std::vector<IT> col_idx_;       ///< column indices for accessing dense vector
    std::vector<IT> rl_;            ///< number of nonzeros in each row
public:
    /**
     *  \brief      Constructor
     *  \details    Does not initialize any data.
     *  \param[in]  num_rows        number of rows of the original matrix (this value is only provided to have an unified interface)
     *  \param[in]  num_columns     number of columns of the original matrix (this value is only provided to have an unified interface)
     */
    ELLMatrix(const IT num_rows, const IT num_columns) {
    }

    /**
     *  @details    get row indices
     *  @returns    a list of row indices for all rows comprising of at least one element
     */
    std::vector<IT> get_post_rank() {
        return post_ranks_;
    }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() { 
        auto pre_ranks = std::vector<std::vector<IT>>();

        for(IT r = 0; r < post_ranks_.size(); r++) {
            auto beg = col_idx_.begin() + r*maxnzr_;
            auto end = col_idx_.begin() + r*maxnzr_ + rl_[r];
            pre_ranks.push_back(std::vector<IT>(beg, end));
        }

        return pre_ranks; 
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        auto beg = col_idx_.begin() + lil_idx*maxnzr_;
        auto end = col_idx_.begin() + lil_idx*maxnzr_ + rl_[lil_idx];

        return std::vector<IT>(beg, end);
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses across all rows
     */
    unsigned int nb_synapses() {
        int size = 0;
        for (auto it = rl_.begin(); it != rl_.end(); it++) {
            size += *it;
        }

        return size;
    }

    /**
     *  @details    returns the stored connections in this matrix for a given row. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    number of synapses across all rows of a given row.
     */
    unsigned int nb_synapses(int lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        return rl_[lil_idx];
    }

    /**
     *  @details    returns the number of stored rows. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    unsigned int nb_dendrites() {
        return 0;
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    First we scan *pre_ranks* to determine the value maxnzr_. Then we convert pre_ranks.
     *  @todo       Currently we ignore post_ranks ...
     */
    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::init_matrix_from_lil()" << std::endl;
    #endif
        assert( (post_ranks.size() == pre_ranks.size()) );

        //
        // 1st step:    iterate across the LIL to identify maximum
        //              row length
        post_ranks_ = post_ranks;
        maxnzr_ = std::numeric_limits<IT>::min();
        rl_ = std::vector<int>(post_ranks.size());

        auto pre_it = pre_ranks.begin();
        IT idx = 0;

        for(; pre_it != pre_ranks.end(); pre_it++, idx++) {
            rl_[idx] = pre_it->size();
        }

        maxnzr_ = *std::max_element(rl_.begin(), rl_.end());

    #ifdef _DEBUG
        std::cout << "Create " << post_ranks_.size() << " times " << maxnzr_ << " dense connectivity matrix " << std::endl;
        std::cout << "row lengths = [ ";
        for(auto it = rl_.begin(); it != rl_.end(); it++)
            std::cout << *it << " ";
        std::cout << "]" << std::endl;
    #endif

        if (row_major) {
        //
        // 2nd step:    iterate across the LIL to copy indices
        //
        // Contrary to many reference implementations we take 0 here but we have rl_
        // to encode the "real" row length.
        col_idx_ = std::vector<IT>(maxnzr_ * post_ranks_.size(), 0);

        pre_it = pre_ranks.begin();
        idx = 0;

        for(; pre_it != pre_ranks.end(); pre_it++, idx++) {
            IT col_off = idx * maxnzr_;
            for (auto col_it = pre_it->begin(); col_it != pre_it->end(); col_it++) {
                col_idx_[col_off++] = *col_it;
            }
        }
    
    #ifdef _DEBUG
        std::cout << "column_indices = [ " << std::endl;
        for (IT r = 0; r < post_ranks_.size(); r++ ) {
            std::cout << "[ ";
            for( IT c = 0; c < maxnzr_; c++) {
                std::cout << col_idx_[r*maxnzr_+c] << " ";
            }
            std::cout << "]," << std::endl;
        }

        std::cout << "]" << std::endl;
    #endif

        }else{
            std::cerr << "ELLMatrix for column major is not yet implemented ... " << std::endl;
        }
    }

    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   the default value for all nonzeros in the matrix.
     *  @returns    Determines a flattened dense matrix of dimension num_rows_ times maxnzr_
     */
    template <typename VT>
    std::vector< VT > init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with constant " << default_value << std::endl;
    #endif
        return std::vector<VT> (post_ranks_.size() * maxnzr_, default_value);
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
        for(IT r = 0; r < post_ranks_.size(); r++) {
            assert( (rl_[r] == data[r].size()) );
            auto beg = variable.begin() + r*maxnzr_;

            std::copy(data[r].begin(), data[r].end(), beg);
        }
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
        assert( (rl_[lil_idx] == data.size()) );

        auto beg = variable.begin() + lil_idx*maxnzr_;
        std::copy(data.begin(), data.end(), beg);
    }

    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT row_idx, const IT column_idx, const VT value) {
        std::cerr << "Not implemented" << std::endl;
    }
   
    /**
     *  @brief      retrieve a LIL representation for a given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @returns    a LIL representation from the given variable.
     */
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector<VT> &variable) {
        auto lil_variable = std::vector< std::vector < VT > >();

        for(IT r = 0; r < post_ranks_.size(); r++) {
            auto beg = variable.begin() + r*maxnzr_;
            auto end = variable.begin() + r*maxnzr_ + rl_[r];
            lil_variable.push_back(std::vector<VT>(beg, end));
        }

        return lil_variable;
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface to retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a vector containing all elements of the provided variable and lil_idx
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector< VT >& variable, const IT &lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        auto beg = variable.begin() + lil_idx*maxnzr_;
        auto end = variable.begin() + lil_idx*maxnzr_ + rl_[lil_idx];

        return std::vector < VT >(beg, end);
    }

    /**
     *  @brief      retruns a single value from the given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @param[in]  col_idx     index of the selected column.
     *  @returns    the value at position (lil_idx, col_idx)
     */
    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT &lil_idx, const IT &col_idx) {

        return static_cast<VT>(0.0); // should not happen
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        size_t size = 0;

        return size;
    }
};
