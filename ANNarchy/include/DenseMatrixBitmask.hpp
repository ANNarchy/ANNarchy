/*
 *    DenseMatrixBitmask.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2025  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

/**
 *  @brief              Connectivity representation using a full matrix.
 *  @details            Contrary to all other classes in this template library this matrix format is not a sparse matrix.
 *                      Please take care on the indices. Many accessors in this class uses the row_idx directly and not the lil_idx.
 *  @tparam     IT      data type to represent the ranks within the matrix. Generally unsigned data types should be chosen.
 *                      The data type determines the maximum size for the number of elements in a column respectively the number
 *                      of rows encoded in the matrix:
 * 
 *                      - unsigned char (1 byte):        [0 .. 255]
 *                      - unsigned short int (2 byte):   [0 .. 65.535]
 *                      - unsigned int (4 byte):         [0 .. 4.294.967.295]
 *
 *                      The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and LILMatrix::num_columns_)
 * 
 *   @tparam    ST      the second type should be used if the index type IT could overflow. For instance, the nb_synapses method should return ST as
 *                      the maximum value in case a full dense matrix would be IT times IT entries.
 *   @tparam    MT      As a zero can represent a non-existing entry or a existing entry, we need an additional array which encodes if a position is
 *                      a non-zero entry in the matrix. In this implementation each value of the mask corresponds to a set of non-zero entries of the
 *                      matrix. The number of elements encoded depdends on the selected datatype, more precise the number of bits representing this
 *                      type. For instance:
 *
 *                      - unsigned char (1 byte):        8
 *                      - unsigned short int (2 byte):   16
 *                      - unsigned int (4 byte):         32
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major=true>
class DenseMatrixBitmask {
protected:
    const IT num_rows_;             ///< maximum number of rows which equals the maximum length of post_rank as well as maximum size of top-level of pre_rank.
    const IT num_columns_;          ///< maximum number of columns which equals the maximum available size in the sub-level vectors.
    const IT mask_size_;            ///< number of elements encoded in one mask entry (depends on MT).
    const IT num_mask_cols_;        ///< number of mask entries in one row.
    std::vector<MT> mask_;          ///< encodes if an entry in the full matrix is a nonzero. Please note, in many C++ implementations bool will default to an integer. Therefore we use char here to ensure that we really use only 1 byte.
    std::vector<IT> post_ranks_;    ///< encodes the indices of rows with at least one non-zero.

    /*
     *  @brief      Decode the column indices for nonzeros in the matrix.
     */
    virtual std::vector<IT> decode_column_indices(IT row_idx) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::decode_column_indices(rk_post = "<< row_idx <<")" << std::endl;
    #endif
        auto indices = std::vector<IT>();

        for (IT c = 0; c < num_columns_; c++) {
            IT mask_idx = c / this->mask_size_;
            IT mask_pos = c % this->mask_size_;

            if (mask_[row_idx * this->num_mask_cols_ + mask_idx] & (1<<mask_pos))
                indices.push_back(c);
        }

        return indices;
    }

public:

    /**
     * @brief       Construct a new dense matrix object.
     * @details     This function does not allocate the matrix.
     *
     * @param[in]   num_rows      number of rows in the matrix
     * @param[in]   num_columns   number of columns in the matrix
     * @note        The matrix dimensions can not be changed afterwards. This function also
     *              initializes DenseMatrixBitmask::mask_size_ and DenseMatrixBitmask::num_mask_cols_.
     */
    explicit DenseMatrixBitmask(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns),
        mask_size_(static_cast<IT>(sizeof(MT)) * 8),
        num_mask_cols_(static_cast<IT>(ceil(static_cast<double>(num_columns_)/static_cast<double>(this->mask_size_))))
    {
        // we check if we can encode all possible values
        assert( (static_cast<long long>(num_rows_ * num_columns_) < static_cast<long long>(std::numeric_limits<ST>::max())) );

    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::DenseMatrixBitmask(num_rows="<<num_rows<<", num_columns="<<num_columns<<")" << std::endl;
        std::cout << "  using per row " << this->num_mask_cols_ <<  " mask entries that encodes " << this->mask_size_ << " matrix elements each." << std::endl;
    #endif
    }

    /**
     *  @brief      Destructor
     *  @details    calls the DenseMatrixBitmask::clear method. Is not declared as virtual as inheriting classes in our
     *              framework should never be destroyed by the base pointer.
     */
    ~DenseMatrixBitmask() {
    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::~DenseMatrixBitmask()" << std::endl;
    #endif
    }

    /**
     *  @brief      Clear the dense matrix.
     *  @details    Clears the connectivity data stored in the *post_rank* and *pre_rank* STL containers and free
     *              the allocated memory. **Important**: allocated variables are not effected by this!
     */
    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        mask_.clear();
        mask_.shrink_to_fit();
    }

    IT num_rows() {
        return num_rows_;
    }

    IT num_columns() {
        return num_columns_;
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
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            pre_ranks.push_back(std::move(get_dendrite_pre_rank(lil_idx)));
        }
        return pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row.
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        assert(lil_idx < post_ranks_.size());
        return decode_column_indices(post_ranks_[lil_idx]);
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses in the whole matrix.
     */
    ST nb_synapses() {
        ST size = 0;
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            size += dendrite_size(lil_idx);
        }
        return size;
    }

    /**
     *  @brief      Get the number of stored connections in this matrix for a given row.
     *  @details    The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    number of synapses across all rows of a given row.
     */
    IT dendrite_size(IT lil_idx) {
        assert(lil_idx < post_ranks_.size());

        IT row_idx = post_ranks_[lil_idx];
        IT size = 0;
        ST idx;

        if (row_major) {
            for (IT c = 0; c < num_columns_; c++) {
                IT mask_idx = c / this->mask_size_;
                IT mask_pos = c % this->mask_size_;

                if (mask_[row_idx * this->num_mask_cols_ + mask_idx] & (1<<mask_pos))
                    size++;
            }
        } else {
            // HD: currently the mask is implemented only using row-ordering
            for (IT c = 0; c < num_columns_; c++) {
                IT mask_idx = c / this->mask_size_;
                IT mask_pos = c % this->mask_size_;

                if (mask_[row_idx * this->num_mask_cols_ + mask_idx] & (1<<mask_pos))
                    size++;
            }
        }
        return size;
    }

    /**
     *  @brief      Get the number of stored rows.
     *  @details    The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    IT nb_dendrites() {
        return post_ranks_.size();
    }

    /**
     *  @brief      get a list of pre-synaptic neuron ranks and their efferent connections.
     *  @details    while the LILMatrix::nb_synapses and LILMatrix::nb_synapses_per_dendrite are row-centered this
     *              function contains the number of row entries for all columns with at least one row entry.
     *  @returns    a std::map with the pre-synaptic ranks as index and the number of nonzeros per column.
     */
    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();

        if (row_major) {
            for (IT i = 0; i < this->num_rows_; i++) {
                for (IT j = 0; j < this->num_columns_; j++) {
                    ST idx = i*this->num_columns_ + j;
                    if (mask_[idx]) num_efferents[j]++;
                }
            }
        } else {
            for(IT j = 0; j < this->num_columns_; j++) {
                for(IT i = 0; i < this->num_rows_; i++) {
                    ST idx = j*this->num_rows_ + i;
                    if (mask_[idx]) num_efferents[j]++;
                }
            }
        }

        return num_efferents;
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    simply sets the post_rank and pre_rank arrays without further sanity checking.
     *  @param      post_ranks          contains row indices
     *  @param      pre_ranks           contains for each row the corresponding column indices
     */
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
        // Sanity checks
        assert ( (post_ranks.size() == pre_ranks.size()) );
        assert ( (static_cast<unsigned long int>(post_ranks.size()) <= static_cast<unsigned long int>(std::numeric_limits<IT>::max())) );

        // Sanity check: enough memory?
        if (!check_free_memory( num_mask_cols_ * num_rows_ * sizeof(MT)))
            return false;

        // Store row indices
        post_ranks_ = post_ranks;

        // Allocate mask
        mask_ = std::vector<MT>(num_rows_ * num_mask_cols_, static_cast<MT>(0.0));

        // Iterate over LIL and update mask entries to *true* if nonzeros are existing.
        for (IT lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            IT row_idx = post_ranks[lil_idx];
            for(auto inner_col_it = pre_ranks[lil_idx].cbegin(); inner_col_it != pre_ranks[lil_idx].cend(); inner_col_it++) {
                IT mask_idx = *inner_col_it / this->mask_size_;
                IT mask_pos = *inner_col_it % this->mask_size_;

                mask_[row_idx * this->num_mask_cols_ + mask_idx] |= 1 << mask_pos;
            }                
        }

        return true;
    }

    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   the default value for all nonzeros in the matrix.
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<VT> init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::init_matrix_variable()" << std::endl;
        std::cout << "  using constant value " << default_value << std::endl;
    #endif
        if (!check_free_memory(num_columns_ * num_rows_ * sizeof(VT)))
            return std::vector<VT>();

        auto new_variable = std::vector<VT>(num_columns_ * num_rows_, static_cast<VT>(0.0));
        for (IT row_idx = 0; row_idx < num_rows_; row_idx++) {
            auto col_idx = decode_column_indices(row_idx);

            for(auto inner_col_it = col_idx.cbegin(); inner_col_it != col_idx.cend(); inner_col_it++) {
                if (row_major)
                    new_variable[row_idx * num_columns_ + *inner_col_it] = default_value;
                else
                    new_variable[*inner_col_it * num_rows_ + row_idx] = default_value;
            }
        }

        return new_variable;
    }

    /**
     *  @details    Allocates and initialize a num_rows_ by num_columns_ matrix based on the stored
     *              connectivity and where the nonzero values serves an uniform distribution (a, b).
     *  @tparam     VT      data type of the variable.
     *  @param[in]  a       minimum of the distribution
     *  @param[in]  b       maximum of the distribution
     *  @param[in]  rng     a merseanne twister generator (need to be seeded in prior if necessary)
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        if (!check_free_memory(num_columns_ * num_rows_ * sizeof(VT)))
            return std::vector<VT>();

        std::uniform_real_distribution<VT> dis (a,b);
        auto new_variable = std::vector<VT>(num_columns_ * num_rows_, static_cast<VT>(0.0));

        for (IT row_idx = 0; row_idx < num_rows_; row_idx++) {
            // draw the values
            auto col_idx = decode_column_indices(row_idx);
            auto tmp_val = std::vector<VT>(col_idx.size());
            std::generate(tmp_val.begin(), tmp_val.end(), [&]{ return dis(rng); });

            // assign the values
            auto col_it = col_idx.cbegin();
            auto val_it = tmp_val.cbegin();
            for(; col_it != col_idx.cend(); col_it++, val_it++) {
                if (row_major)
                    new_variable[row_idx * num_columns_ + *col_it] = *val_it;
                else
                    new_variable[(*col_it) * num_rows_ + row_idx] = *val_it;
            }
        }
        return new_variable;
    }

    /**
     *  @details    Updates all *existing* entries of a matrix.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  values      new values for the row indicated by lil_idx stored as a list of list according to LILMatrix::pre_rank
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "DenseMatrixBitmask::update_matrix_variable_all()" << std::endl;
    #endif
        // sanity check: target large enough?
        assert( (num_rows_ * num_columns_ == variable.size()) );
        assert( (post_ranks_.size() == data.size()) );

        for (IT lil_idx = 0; lil_idx < data.size(); lil_idx++) {
            update_matrix_variable_row(variable, lil_idx, data[lil_idx]);
        }
    }

    /**
     *  @details        Updates all *existing* entries of a matrix row.
     *  @tparam         VT          data type of the variable.
     *  @param[inout]   variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]      lil_idx     index of the selected row.
     *  @param[in]      values      new values for the row indicated by row_idx.
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> values)
    {
        // select correct row
        assert(lil_idx < post_ranks_.size());
        auto row_idx = post_ranks_[lil_idx];

        // get the indices of nonzeros in the present row
        auto col_idx = decode_column_indices(row_idx);

        // sanity check: enough values for this row?
        assert( (col_idx.size() == values.size()) );

        // copy the data
        auto col_it = col_idx.cbegin();
        auto data_it = values.cbegin();
        for (; col_it != col_idx.cend(); col_it++, data_it++) {
            if (row_major) {
                variable[row_idx * num_columns_ + *col_it] = *data_it;
            } else {
                variable[*col_it * num_rows_ + row_idx] = *data_it;
            }
        }
    }

    /**
     *  @details    Updates a single *existing* entry within the matrix.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  lil_idx     index of the selected row.
     *  @param[in]  value       new matrix value
     *  @todo       Maybe one should check the mask if the nonzero existed before?
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT lil_idx, const IT col_idx, const VT value) {
        // select correct row
        assert(lil_idx < post_ranks_.size());
        auto row_idx = post_ranks_[lil_idx];

        // update the position
        if (row_major) {
            variable[row_idx * num_columns_ + col_idx] = value;
        } else {
            variable[col_idx * num_rows_ + row_idx] = value;
        }
    }

    /**
     *  @brief      retrieve a LIL representation for a given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @returns    a LIL representation from the given variable.
     */
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector<VT>& variable) {
        auto values = std::vector< std::vector < VT > >();

        for (auto idx = 0; idx < post_ranks_.size(); idx++) {
            values.push_back(std::move(get_matrix_variable_row(variable, idx)));
        }

        return values;
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface to retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  lil_idx     index of the selected row.
     *  @returns    a vector containing all elements of the provided variable and row_idx
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT>& variable, const IT &lil_idx) {
        assert(lil_idx < post_ranks_.size());
        auto row_idx = post_ranks_[lil_idx];
        auto col_idx = decode_column_indices(row_idx);
        auto values = std::vector< VT >();
        values.reserve(col_idx.size());

        // gather the data
        for (auto col_it = col_idx.cbegin(); col_it != col_idx.cend(); col_it++) {
            if (row_major) {
                values.push_back(variable[row_idx * num_columns_ + *col_it]);
            } else {
                values.push_back(variable[*col_it * num_rows_ + row_idx]);
            }
        }

        return values;
    }

    /**
     *  @brief      retruns a single value from the given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  lil_idx     index of the selected row.
     *  @param[in]  col_idx     index of the selected column.
     *  @returns    the value at position (lil_idx, col_idx)
     */
    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT &lil_idx, const IT &col_idx) {
        assert(lil_idx < post_ranks_.size());
        auto row_idx = post_ranks_[lil_idx];
        if (row_major) {
            return variable[row_idx * num_columns_ + col_idx];
        } else {
            return variable[col_idx * num_rows_ + row_idx];
        }
    }

    /**
     *  @brief      Initialize a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   value to initialize all elements in the vector
     */
    template <typename VT>
    inline std::vector<VT> init_vector_variable(VT default_value) {
        return std::vector<VT>(num_rows_, default_value);
    }

    /**
     *  @brief      Update the complete vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_vector_variable_all(std::vector<VT> &variable, std::vector<VT> values) {
        // storage target has the right size?
        assert (variable.size() == num_rows_);
        // enough elements?
        assert (post_ranks_.size() == values.size());

        // Update the vector
        if (post_ranks_.size() < num_rows_) {
            for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
                variable[post_ranks_[lil_idx]] = values[lil_idx]; 
            }
        }else{
            std::copy(values.begin(), values.end(), variable.begin());
        }
    }

    /**
     *  @brief      Update a single entry of the vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_vector_variable(std::vector<VT> &variable, const IT lil_idx, const VT value) {
        assert(num_rows_ != variable.size());
        assert(lil_idx < post_ranks_.size());

        variable[post_ranks_[lil_idx]] = value;
    }

    /**
     *  @brief      Get a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline std::vector<VT> get_vector_variable_all(std::vector<VT> variable) {
        if (post_ranks_.size() < num_rows_) {
            std::vector<VT> res;
            res.reserve(post_ranks_.size());
            for (const auto idx : post_ranks_)
                res.push_back(variable[idx]);
            return res;
        } else {
            return variable;
        }
    }

    /**
     *  @brief      Get a single item from a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
    */
    template <typename VT>
    inline VT get_vector_variable(std::vector<VT> variable, const IT lil_idx) {
        assert(lil_idx < post_ranks_.size());
        return variable[post_ranks_[lil_idx]];
    }

    /**
     *  @brief      print the some information on the nonzeros to console.
     *  @details    The print-out will contain among others number rows, number columns, number nonzeros.
     *              Please note, that type casts are required to print-out the numbers encoded if IT or ST
     *              is e.g. unsigned char.
     */
    void print_matrix_statistics() {
        std::cout << "  #rows: " << static_cast<unsigned long>(num_rows_) << std::endl;
        std::cout << "  #columns: " << static_cast<unsigned long>(num_columns_) << std::endl;
        std::cout << "  #nnz: " << static_cast<unsigned long>(nb_synapses()) << std::endl;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. Please note, that type casts are
     *              required to print-out the numbers encoded if IT or ST is e.g. unsigned char.
     */
    void print_data_representation() {
        std::cout << "Dense Matrix instance at " << this << std::endl;

        print_matrix_statistics();
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    virtual size_t size_in_bytes() {
        size_t size = 4 * sizeof(IT);               // scalar values

        size += mask_.capacity() * sizeof(MT);

        return size;
    }
};
