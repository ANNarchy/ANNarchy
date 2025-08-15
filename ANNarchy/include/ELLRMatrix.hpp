/*
 *    ELLRMatrix.hpp
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

#include "LILMatrix.hpp"

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
 *              compressed. This means, that we don't allocate empty rows instead we have a row rank
 *              array which encode which row in the stored matrix corresponds to the dense row index.
 *              For the above matrix this would be:
 * 
 *                  post_ranks_ = [0, 1, 3]
 *
 *  \tparam     IT          index type.
 *  \tparam     ST          size type, if IT would overflow.
 *  \tparam     row_major   determines the matrix storage for the dense sub matrices. If
 *                          set to true, the matrix will be stored as row major, otherwise
 *                          in column major. 
 *                          Please note that the original format stores in row-major to ensure a
 *                          partial caching of data on CPUs. The column-major ordering is only
 *                          intended for the usage on GPUs.
 */
template<typename IT=unsigned int, typename ST=unsigned long int, bool row_major=true>
class ELLRMatrix {
protected:
    IT maxnzr_;                     ///< maximum row length of nonzeros
    const IT dense_num_rows_;       ///< maximum number of rows (dense matrix)
    const IT dense_num_columns_;    ///< maximum number of columns (dense matrix)

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
    explicit ELLRMatrix(const IT num_rows, const IT num_columns):
        dense_num_rows_(num_rows), dense_num_columns_(num_columns) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::ELLRMatrix()" << std::endl;
    #endif
    }

    ELLRMatrix(ELLRMatrix<IT, ST, row_major>* other):
        dense_num_rows_(other->dense_num_rows_), dense_num_columns_(other->dense_num_columns_) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::copy constructor"<< std::endl;
    #endif
        this->maxnzr_ = other->maxnzr_;
        this->post_ranks_ = other->post_ranks_;
        this->col_idx_ = other->col_idx_;
        this->rl_ = other->rl_;
    }

    /**
     *  @brief      Destructor
     *  @details    responsible to delete the allocated GPU memory.
     */
    ~ELLRMatrix() {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::~ELLRMatrix()" << std::endl;
    #endif
    }

    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        col_idx_.clear();
        col_idx_.shrink_to_fit();

        rl_.clear();
        rl_.shrink_to_fit();

        maxnzr_ = 0;
    }

    /**
     *  @brief      returns number of rows of the dense matrix.
     *  @details    this value can differ but should be larger than the number of ELLMatrix::nb_dendrites()
     *  @returns    number of rows of the dense matrix.
     */
    IT num_rows() {
        return dense_num_rows_;
    }

    /**
     *  @brief      returns number of columns of the dense matrix.
     *  @details    this value can differ but should be larger than the number of ELLMatrix::dendrite_size(int lil_idx)
     *  @returns    number of columns of the dense matrix.
     */
    IT num_columns() {
        return dense_num_columns_;
    }

    inline IT get_maxnzr() {
        return maxnzr_;
    }

    inline IT* get_rl() {
        return rl_.data();
    }

    inline IT* get_column_indices() {
        return col_idx_.data();
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

        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++)
            pre_ranks.push_back(std::move(get_dendrite_pre_rank(lil_idx)));

        return pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        if (row_major) {
            auto beg = col_idx_.begin() + lil_idx*maxnzr_;
            auto end = col_idx_.begin() + lil_idx*maxnzr_ + rl_[lil_idx];

            return std::vector<IT>(beg, end);
        } else {
            auto tmp = std::vector < IT >(rl_[lil_idx]);
            int num_rows = post_ranks_.size();
            for (int c = 0; c < rl_[lil_idx]; c++) {
                tmp[c] = col_idx_[c*num_rows+lil_idx];
            }
            return tmp;
        }
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses across all rows
     */
    size_t nb_synapses() {
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
    IT dendrite_size(IT lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        return rl_[lil_idx];
    }

    /**
     *  @details    returns the number of stored rows. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    IT nb_dendrites() {
        return post_ranks_.size();
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    First we scan *pre_ranks* to determine the value maxnzr_. Then we convert pre_ranks.
     *  @todo       Currently we ignore post_ranks ...
     */
    bool init_matrix_from_lil(std::vector<IT> post_ranks, std::vector< std::vector<IT> > pre_ranks) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::init_matrix_from_lil()" << std::endl;
    #endif

        // Sanity check
        assert( (post_ranks.size() == pre_ranks.size()) );

        //
        // Store the LIL ranks
        post_ranks_ = post_ranks;

        //
        // 1st step:    iterate across the LIL to identify maximum
        //              row length
        maxnzr_ = std::numeric_limits<IT>::min();
        rl_ = std::vector<IT>(post_ranks_.size());

        auto pre_it = pre_ranks.begin();
        IT idx = 0;

        for(; pre_it != pre_ranks.end(); pre_it++, idx++) {
            rl_[idx] = pre_it->size();
        }

        maxnzr_ = *std::max_element(rl_.begin(), rl_.end());

        // Test if we produce an overflow for ST
        assert( (static_cast<unsigned long int>(post_ranks_.size() * maxnzr_) < static_cast<unsigned long int>(std::numeric_limits<ST>::max())) );

        // Test if the matrix fits into memory
        if (!check_free_memory(maxnzr_ * post_ranks_.size() * sizeof(IT)))
            return false;

        //
        // 2nd step:    iterate across the LIL to copy indices
        //
        // Contrary to many reference implementations we take 0 here but we have rl_
        // to encode the "real" row length.
        if (row_major) {
            col_idx_ = std::vector<IT>(maxnzr_ * post_ranks_.size(), 0);

            pre_it = pre_ranks.begin();
            idx = 0;

            for(; pre_it != pre_ranks.end(); pre_it++, idx++) {
                size_t col_off = idx * maxnzr_;
                for (auto col_it = pre_it->begin(); col_it != pre_it->end(); col_it++) {
                    col_idx_[col_off++] = *col_it;
                }
            }

        } else {
            int num_rows = post_ranks_.size();
            col_idx_ = std::vector<IT>(maxnzr_ * num_rows, 0);

            for (int r = 0; r < num_rows; r++) {
                int c = 0;
                for (auto col_it = pre_ranks[r].begin(); col_it != pre_ranks[r].end(); col_it++, c++) {
                    col_idx_[c*num_rows+r] = *col_it;
                }
            }
        }
    #ifdef _DEBUG
        std::cout << "created ELLRMatrix:" << std::endl;
        this->print_matrix_statistics();
    #endif
        return true;
    }

    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @details    this function creates also the variable array, which is usually performed afterwards.
     *  @tparam     VT          value type of the nonzero
     *  @tparam     zero_based  set to true if the contained data in csv has as minimum possible index 0. If
     *                          set to false, the read-in indices will be decremented by 1.
     */
    template<typename VT, bool zero_based=true>
    std::vector<VT> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
        auto tmp_col_idx = std::vector< std::vector < IT > >(dense_num_rows_, std::vector<IT>());
        auto tmp_values = std::vector< std::vector < VT > >(dense_num_rows_, std::vector<VT>());

        // Load as LIL
        std::ifstream mat_file( filename );
        if(!mat_file.is_open()) {
            std::cerr << "Could not open the file: " << filename << std::endl;
        } else {
            std::string item;
            auto coo_triplet = std::vector<std::string>(3);

            std::string line = "";
            IT r_cast, c_cast;
            VT v_cast;

            // Iterate through each line and split the content using delimeter
            while (getline(mat_file, line))
            {
                if (line.size() == 0)
                    continue;   // fetched an empty line

                std::stringstream ss(line);
                for (int i = 0; i < 3; i++) {
                    std::getline(ss, item, delimiter);
                    coo_triplet[i] = std::move(item);
                }

                if (zero_based) {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()));
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()));
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                } else {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()) -1);
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()) -1);
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                }
                //std::cout << r_cast << ", " << c_cast << ", " << v_cast << std::endl;
                tmp_col_idx[r_cast].push_back(c_cast);
                tmp_values[r_cast].push_back(v_cast);
            }
        }

        // create a LIL from the read data
        auto lil_ranks = std::vector<IT>();
        auto lil_col_idx = std::vector<std::vector<IT>>();
        auto lil_values = std::vector<std::vector<VT>>();

        // iterate over all possible rows
        for(auto row = 0; row < dense_num_rows_; row++) {
            // do we have an entry?
            if (tmp_col_idx[row].size() > 0) {
                lil_ranks.push_back(row);
                lil_col_idx.push_back(std::move(tmp_col_idx[row]));
                lil_values.push_back(std::move(tmp_values[row]));
            }
        }

        // create connectivity
        init_matrix_from_lil(lil_ranks, lil_col_idx);

        // create the values matrix
        auto value = init_matrix_variable<VT>(0.0);
        update_matrix_variable_all<VT>(value, lil_values);

        return value;
    }

    //
    //  ANNarchy connectivity patterns
    //
    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::fixed_number_pre_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " nnz_per_row: " << nnz_per_row << std::endl;
    #endif
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT, ST>(this->dense_num_rows_, this->dense_num_columns_);
        bool success = lil_mat->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);
        if (!success) {
            std::cerr << "ELLRMatrix::fixed_number_pre_pattern(): construction of intermediate LIL failed.";
            return false;
        }

        // Generate CSRC_T from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;

        return true;
    }

    bool fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::fixed_probability_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " p: " << p << std::endl;
    #endif
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT, ST>(this->dense_num_rows_, this->dense_num_columns_);
        bool success = lil_mat->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);
        if (!success) {
            std::cerr << "ELLRMatrix::fixed_probability_pattern(): construction of intermediate LIL failed.";
            return false;
        }

        // Generate ELLPACK-R from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;

        return true;
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
        std::cout << "ELLRMatrix::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        check_free_memory(post_ranks_.size() * maxnzr_ * sizeof(VT));

        // fill all places with 0
        auto new_variable = std::vector<VT> (post_ranks_.size() * maxnzr_, static_cast<VT>(0.0));

        // only "set" nonzeros should be updated
        if (row_major) {
            for (IT row_idx = 0; row_idx < post_ranks_.size(); row_idx++) {
                auto beg = new_variable.begin() + row_idx * maxnzr_;
                auto end = new_variable.begin() + row_idx * maxnzr_ + rl_[row_idx];
                std::generate(beg, end, [&]{ return default_value; });
            }
        } else {
            for (IT row_idx = 0; row_idx < post_ranks_.size(); row_idx++) {
                for (IT col_idx = 0; col_idx < rl_[row_idx]; col_idx++) {
                    new_variable[col_idx * post_ranks_.size() + row_idx] = default_value;
                }
            }
        }
        
        return new_variable;
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "ELLRMatrix::initialize_variable_uniform(): arguments = (" << a << ", " << b << ")" << std::endl;
    #endif
        check_free_memory(post_ranks_.size() * maxnzr_ * sizeof(VT));

        // Init RNG distribution object
        std::uniform_real_distribution<VT> dis (a,b);

        // fill all places with 0
        auto new_variable = std::vector<VT>(post_ranks_.size() * maxnzr_, 0.0);

        // only "set" nonzeros should be updated
        if (row_major) {
            for (IT row_idx = 0; row_idx < post_ranks_.size(); row_idx++) {
                auto beg = new_variable.begin() + row_idx * maxnzr_;
                auto end = new_variable.begin() + row_idx * maxnzr_ + rl_[row_idx];
                std::generate(beg, end, [&]{ return dis(rng); });
            }
        } else {
            // HD (5th Aug 2025):   I'm aware the fact that this access pattern
            //                      is not run-time efficient, but this ensures that
            //                      forward and backward matrix is initialized in the same way
            for (IT row_idx = 0; row_idx < post_ranks_.size(); row_idx++) {
                for (IT col_idx = 0; col_idx < rl_[row_idx]; col_idx++) {
                    new_variable[col_idx * post_ranks_.size() + row_idx] = dis(rng);
                }
            }

        }

        return new_variable;
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_normal(VT mean, VT sigma, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Normal(" << mean << ", " << sigma << ")" << std::endl;
    #endif
        check_free_memory(post_ranks_.size() * maxnzr_ * sizeof(VT));

        std::normal_distribution<VT> dis (mean, sigma);
        auto new_variable = std::vector<VT>(post_ranks_.size() * maxnzr_, 0.0);

        // only "set" nonzeros should be updated
        if (row_major) {
            for (IT row_idx = 0; row_idx < post_ranks_.size(); row_idx++) {
                auto beg = new_variable.begin() + row_idx * maxnzr_;
                auto end = new_variable.begin() + row_idx * maxnzr_ + rl_[row_idx];
                std::generate(beg, end, [&]{ return dis(rng); });
            }
        } else {
            std::cerr << "ELLRMatrix::init_matrix_variable is not implemented for column-major." << std::endl;
        }

        return new_variable;
    }

    /**
     *  @details    Updates all matrix values based on a LIL representation
     *  @tparam     VT              data type of the variable.
     *  @param[in]  variable        ELLPACK variable container
     *  @param[in]  data            LIL variable container
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
        assert( (post_ranks_.size() == data.size()) );
        assert( (rl_.size() == data.size()) );

        for(IT r = 0; r < post_ranks_.size(); r++) {
            update_matrix_variable_row(variable, r, data[r]);
        }
    }

    /**
     *  @details    Updates a row of the matrix.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  variable        ELLPACK variable container
     *  @param[in]  lil_idx
     *  @param[in]  data
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
        if (row_major) {
            assert( (rl_[lil_idx] == data.size()) );

            auto beg = variable.begin() + lil_idx*maxnzr_;
            std::copy(data.begin(), data.end(), beg);
        } else {
            for(IT c = 0; c < rl_[lil_idx]; c++) {
                variable[c*post_ranks_.size()+lil_idx] = data[c];
            }
        }
    }

    /**
     *  @details    Updates a single position in the matrix.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  variable        ELLPACK variable container
     *  @param[in]  lil_idx
     *  @param[in]  column_idx
     *  @param[in]  data
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT lil_idx, const IT column_idx, const VT value) {
        if (row_major) {
            for (ST idx = lil_idx*maxnzr_; idx < lil_idx*maxnzr_+rl_[lil_idx]; idx++) {
                if (col_idx_[idx] == column_idx) {
                    variable[idx] = value;
                }
            }
        } else {
            IT num_rows = post_ranks_.size();

            for (ST idx = lil_idx; idx < lil_idx+rl_[lil_idx]*num_rows; idx += num_rows) {
                if (col_idx_[idx] == column_idx) {
                    variable[idx] = value;
                }
            }
        }
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

        if (row_major) {
            for(IT r = 0; r < post_ranks_.size(); r++) {
                auto beg = variable.begin() + r*maxnzr_;
                auto end = variable.begin() + r*maxnzr_ + rl_[r];
                lil_variable.push_back(std::vector<VT>(beg, end));
            }
        } else {
            for(IT r = 0; r < post_ranks_.size(); r++) {
                lil_variable.push_back(std::move(get_matrix_variable_row(variable, r)));
            }            
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
        assert( (lil_idx < rl_.size()) );

        if (row_major) {
            auto beg = variable.begin() + lil_idx*maxnzr_;
            auto end = variable.begin() + lil_idx*maxnzr_ + rl_[lil_idx];

            return std::vector < VT >(beg, end);
        } else {
            auto tmp = std::vector < VT >(rl_[lil_idx]);
            IT num_rows = post_ranks_.size();
            for (int c = 0; c < rl_[lil_idx]; c++) {
                tmp[c] = variable[c*num_rows+lil_idx];
            }
            return tmp;
        }
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
        if (row_major) {
            for (ST idx = lil_idx*maxnzr_; idx < lil_idx*maxnzr_+rl_[lil_idx]; idx++) {
                if (col_idx_[idx] == col_idx) {
                    return variable[idx];
                }
            }
        }else{
            IT num_rows = post_ranks_.size();

            for (ST idx = lil_idx; idx < lil_idx+rl_[lil_idx]*num_rows; idx += num_rows) {
                if (col_idx_[idx] == col_idx) {
                    return variable[idx];
                }
            }
        }

        return static_cast<VT>(0.0); // should not happen
    }

    /**
     *  @brief      Initialize a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   value to initialize all elements in the vector
     */
    template <typename VT>
    inline std::vector<VT> init_vector_variable(VT default_value) {
        return std::vector<VT>(post_ranks_.size(), default_value);
    }

    /**
     *  @brief      Update the complete vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_vector_variable_all(std::vector<VT> &variable, std::vector<VT> values) {
        assert ( (variable.size() == values.size()) );

        std::copy(values.begin(), values.end(), variable.begin());
    }

    /**
     *  @brief      Update a single entry of the vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_vector_variable(std::vector<VT> &variable, int lil_idx, VT value) {
        assert( (lil_idx < post_ranks_.size()) );

        variable[lil_idx] = value;
    }

    /**
     *  @brief      Get a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline std::vector<VT> get_vector_variable_all(std::vector<VT> variable) {
        return variable;
    }

    /**
     *  @brief      Get a single item from a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
    */
    template <typename VT>
    inline VT get_vector_variable(std::vector<VT> variable, int lil_idx) {
        assert( (lil_idx < post_ranks_.size()) );

        return variable[lil_idx];
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    virtual size_t size_in_bytes() {
        size_t size = 3 * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += col_idx_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += rl_.capacity() * sizeof(IT);

        return size;
    }

    void print_matrix_statistics() {
        size_t sum = 0;
        for (auto it = rl_.begin(); it != rl_.end(); it++ ) {
            if (*it > 0) {
                sum += *it;
            }
        } 
        double avg_nnz_per_row = static_cast<double>(sum) / static_cast<double>(post_ranks_.size());

        std::cout << "  #rows (dense): " << static_cast<unsigned long>(dense_num_rows_) << std::endl;
        std::cout << "  #columns (dense): " << static_cast<unsigned long>(dense_num_columns_) << std::endl;
        std::cout << "  #nnz (sparse): " << nb_synapses() << std::endl;
        std::cout << "  empty rows: " << dense_num_rows_ - post_ranks_.size() << std::endl;
        std::cout << "  avg_nnz_per_row: " << avg_nnz_per_row << std::endl;
        std::cout << "  allocated dense matrix for ELL-R = (" << static_cast<unsigned long>(post_ranks_.size()) << ", " <<  static_cast<unsigned long>(maxnzr_) << ")" <<\
                     " stored as " << ((row_major) ? "row_major" : "column_major") << std::endl;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. Please note, that type casts are
     *              required to print-out the numbers encoded in unsigned char as numbers. 
     */
    virtual void print_data_representation() {
        std::cout << "ELLRMatrix instance at " << this << std::endl;
        print_matrix_statistics();

        std::cout << "  post_ranks = [ " << std::endl;
        for (IT r = 0; r < post_ranks_.size(); r++ ) {
            std::cout << static_cast<unsigned long>(post_ranks_[r]) << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  column_indices = [ " << std::endl;
        for (IT r = 0; r < post_ranks_.size(); r++ ) {
            std::cout << "[ ";
            for( IT c = 0; c < maxnzr_; c++) {
                std::cout << static_cast<unsigned long>(col_idx_[r*maxnzr_+c]) << " ";
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]" << std::endl;
    }

};
