/*
 *    CSRMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
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

#include "LILMatrix.hpp"

/**
 *  @brief      Implementation of a *compressed sparse row* (CSR) format.
 *  @details    Probably the most common sparse matrix format in computer science. The major idea is that only nonzeros are
 *              stored in a long continuous array and a second array provides a lookup, which slice in this array contains
 *              to a row.
 *
 *              Let's consider the following example matrix
 * 
 *                      | 0 1 0 |
 *                  A = | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 * 
 *              So the array containing the column indices would be:
 * 
 *                  col_idx_ = [ 1 , 0, 3, 3 ]
 *
 *              And the row assignment would be:
 * 
 *                  row_begin_ = [ 0, 1, 3, 3, 4 ]
 * 
 *              So the i-th row in the matrix reaches from row_begin_[i] to row_begin_[i+1] and could select the column indices.
 *              As visible on the third row, an empty row is indicated if both values are equal. So, contrary to the LIL which does
 *              not store empty rows, the CSR does. We therefore need the post_ranks_ array to have a mapping between LIL indices
 *              and matrix rows.
 * 
 *  @tparam     IT      data type to represent the ranks within the matrix. Generally unsigned data types should be chosen.
 *                      The data type determines the maximum size of the matrix:
 * 
 *                      - unsigned char (1 byte):        [0 .. 255]
 *                      - unsigned short int (2 byte):   [0 .. 65.535]
 *                      - unsigned int (4 byte):         [0 .. 4.294.967.295]
 * 
 *              The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and ::num_columns_)
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class CSRMatrix {

  protected:
    std::vector<IT> post_ranks_;        ///< Needed to translate LIL indices to row_indicies.
    std::vector<ST> row_begin_;         ///< i-th element marks the begin of the i-th row. The chosen type for encoding should be able to
                                        ///< contain num_rows_ * num_columns_ elements (we choose size_t to be on the safe side)
    std::vector<IT> col_idx_;           ///< contains the column indices in row major order order. To access row i, get indices from row_begin_.

    IT num_rows_;                       ///< number of rows in the dense matrix
    IT num_columns_;                    ///< number of columns in the dense matrix
    ST num_non_zeros_;                  ///< number of nonzeros

  public:

    explicit CSRMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
    #ifdef _DEBUG
        std::cout << "Created CSR matrix " << this << " with dense dimension: " << static_cast<long>(num_rows_) << " times " << static_cast<long>(num_columns_) << std::endl;
    #endif

        row_begin_ = std::vector<ST>(num_rows+1, 0);
        post_ranks_ = std::vector<IT>();
        col_idx_ = std::vector<IT>();
        num_non_zeros_ = 0;
    }

    /*
     *  @brief      Destructor.
     *  @details    Is not declared as virtual as inheriting classes in our framework should never be destroyed by the base pointer.
     */
    ~CSRMatrix() {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::~CSRMatrix()" << std::endl;
    #endif
        // not destroyed by clear()
        row_begin_.clear();
        row_begin_.shrink_to_fit();
    }

    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::clear()" << std::endl;
    #endif
        std::fill(row_begin_.begin(), row_begin_.end(), 0);

        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        col_idx_.clear();
        col_idx_.shrink_to_fit();
        num_non_zeros_ = 0;
    }

    //
    //  Accessor to member variables
    //
    inline IT num_rows() {
        return num_rows_;
    }

    inline IT num_columns() {
        return num_columns_;
    }

    inline std::vector<IT> column_indices() {
        return col_idx_;
    }

    inline std::vector<ST> row_ptr() {
        return row_begin_;
    }

    //
    //  Initialization methods
    //

    /**
     *  @brief      Initialize CSR based on a LIL representation.
     *  @see        LILMatrix::init_matrix_from_lil()
     */
    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::init_matrix_from_lil()" << std::endl;
    #endif
        // sanity check of inputs
        assert( (row_indices.size() == column_indices.size()) );
        assert( (row_indices.size() < std::numeric_limits<IT>::max()) );
        assert( (row_indices.size() <= num_rows_) );

        // construct the CSR from LIL
        post_ranks_ = row_indices;

        IT lil_row_idx = 0;
        for (IT r = 0; r < num_rows_; r++) {
            row_begin_[r] = col_idx_.size();

            // We are already done with the LIL matrix
            if (lil_row_idx == row_indices.size()) {
                // HD (1st Sep. 2022):
                //  don't break the loop here, otherwise the row_begin_ array is
                //  not correctly updated. Which then could crash the
                //  inverse_connectivity_matrix() call
                continue;
            }

            // check if this row is in list
            if (r == row_indices[lil_row_idx]) {
                col_idx_.insert(col_idx_.end(), column_indices[lil_row_idx].begin(), column_indices[lil_row_idx].end());
                num_non_zeros_ += column_indices[lil_row_idx].size();

                // next row in LIL
                lil_row_idx++;
            }
        }
        row_begin_[num_rows_] = col_idx_.size();

        // sanity check after transformation
        if (lil_row_idx != row_indices.size())
            std::cerr << "something went wrong ..." << std::endl;
        if (num_non_zeros_ != col_idx_.size())
            std::cerr << "something went wrong ... " << num_non_zeros_ << std::endl;

        // remove unneccessary allocated space
        col_idx_.shrink_to_fit();

    #if defined(_DEBUG_CONN)
        print_data_representation(2, true);
    #elif defined(_DEBUG)
        print_data_representation(2, false);
    #endif
        return true;
    }

    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @see        LILMatrix::init_matrix_from_lil()
     */
    template<typename VT, bool zero_based=true>
    std::vector<VT> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::init_matrix_from_csv()" << std::endl;
    #endif
        auto tmp_col_idx = std::vector< std::vector < IT > >(num_rows_, std::vector<IT>());
        auto tmp_values = std::vector< std::vector < VT > >(num_rows_, std::vector<VT>());
        ST coo_pairs = 0;

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

                coo_pairs++; // for sanity check
            }
        }

        // Build up CSR connectivity and LIL values
        // (the latter is required for the update call)
        post_ranks_.clear();
        auto lil_values = std::vector<std::vector<VT>>();
        for(auto row = 0; row < row_begin_.size()-1; row++) {
            // CSR connectivity
            row_begin_[row] = col_idx_.size();
            col_idx_.insert(col_idx_.end(), tmp_col_idx[row].begin(), tmp_col_idx[row].end());

            // LIL values
            if (tmp_col_idx[row].size() > 0) {
                post_ranks_.push_back(row);
                lil_values.push_back(std::move(tmp_values[row]));
            }
        }
        row_begin_[row_begin_.size()-1]=col_idx_.size();
        num_non_zeros_ = col_idx_.size();

        // remove unneccessary allocated space
        col_idx_.shrink_to_fit();
        post_ranks_.shrink_to_fit();

        // Sanity check
        assert( (num_non_zeros_ == coo_pairs) );

    #ifdef _DEBUG
        std::cout << "Extracted " << coo_pairs << " from " << filename << std::endl;
    #endif

        // return value
        auto value = init_matrix_variable<VT>(0.0);
        update_matrix_variable_all<VT>(value, lil_values);

        return value;
    }

    //
    //  ANNarchy connectivity patterns
    //
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::fixed_number_pre_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " nnz per row: " << nnz_per_row << std::endl;
    #endif
        post_ranks_ = post_ranks;

        // for each row we select a subset of the provided pre ranks
        for(auto lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            row_begin_[lil_idx] = num_non_zeros_;
            // shuffle indices (source vector is modified!)
            std::shuffle(pre_ranks.begin(), pre_ranks.end(), rng);

            // select nnz_per_row elements
            auto tmp_col_indices = std::vector<IT>(pre_ranks.begin(), pre_ranks.begin()+nnz_per_row);

            // sort the indices before storage
            std::sort(tmp_col_indices.begin(), tmp_col_indices.end());

            // store in CSR
            col_idx_.insert(col_idx_.end(), tmp_col_indices.begin(), tmp_col_indices.end());
            num_non_zeros_ += nnz_per_row;
        }
        row_begin_[num_rows_] = num_non_zeros_;

        // remove unneccessary allocated space
        col_idx_.shrink_to_fit();
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::fixed_probability_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " p: " << p << std::endl;
    #endif
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT, ST>(this->num_rows_, this->num_columns_);
        lil_mat->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // Generate CSR from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;
    }

    //
    //  Connectivity Accessor
    //
    std::vector<IT> get_post_rank() { return post_ranks_; }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() {
        std::vector<std::vector<IT>> lil_pre_ranks;

        for(auto post_it=post_ranks_.begin(); post_it != post_ranks_.end(); post_it++) {
            auto beg = col_idx_.begin() + row_begin_[*post_it];
            auto end = col_idx_.begin() + row_begin_[*post_it+1];

            lil_pre_ranks.push_back(std::vector<IT>(beg, end));
        }

        return lil_pre_ranks;
    }

    typename std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        IT row_idx = post_ranks_[lil_idx];
        auto beg = col_idx_.begin() + row_begin_[row_idx];
        auto end = col_idx_.begin() + row_begin_[row_idx+1];
        return std::vector<IT>(beg, end);
    }

    /**
     *  @brief      number of elements in one row
     *  @returns    can be at maximum the number of columns and is therefore type IT
     */
    IT dendrite_size(IT lil_idx) {
        int post_rank = post_ranks_[lil_idx];
        return (row_begin_[post_rank+1] - row_begin_[post_rank]);
    }

    /**
     *  @brief      number of synapses in the complete matrix
     *  @returns    can easily exceed the number of columns and is therefore of type ST
     */
    inline ST nb_synapses() {
        return this->num_non_zeros_;
    }

    inline IT nb_dendrites() {
        return post_ranks_.size();
    }

    //
    //  Initialize Variables
    //
    template <typename VT>
    std::vector<VT> init_matrix_variable(VT default_value) {
        return std::vector<VT>(num_non_zeros_, default_value);
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::initialize_variable_uniform(): arguments = (" << a << ", " << b << ") and num_non_zeros_ = " << num_non_zeros_ << std::endl;
    #endif
        std::uniform_real_distribution<VT> dis (a,b);
        auto new_variable = std::vector<VT>(num_non_zeros_, 0.0);
        std::generate(new_variable.begin(), new_variable.end(), [&]{ return dis(rng); });
        return new_variable;
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_normal(VT mean, VT sigma, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Normal(" << mean << ", " << sigma << ")" << std::endl;
    #endif
        std::normal_distribution<VT> dis (mean, sigma);
        auto new_variable = std::vector<VT>(num_non_zeros_, 0.0);
        std::generate(new_variable.begin(), new_variable.end(), [&]{ return dis(rng); });
        return new_variable;
    }

    //
    //  Update Variables
    //
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT lil_idx, const IT column_idx, const VT value) {
        IT row_idx = post_ranks_[lil_idx];
        for(int j = row_begin_[row_idx]; j < row_begin_[row_idx+1]; j++) {
            if ( col_idx_[j] == column_idx ) {
                variable[j] = value;
                break;
            }
        }
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
        IT row_idx = post_ranks_[lil_idx];
        std::copy(data.begin(), data.end(), variable.begin() + row_begin_[row_idx]);
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) 
    {
        if (data.size() != post_ranks_.size())
            std::cerr << "Update variable failed: mismatch of data field sizes." << std::endl;

        for (auto i = 0; i < post_ranks_.size(); i++) {
            update_matrix_variable_row(variable, i, data[i]);
        }
    }

    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT> &variable, const IT lil_idx, const IT column_idx) {
        IT row_idx = post_ranks_[lil_idx];
        for(int j = row_begin_[row_idx]; j < row_begin_[row_idx+1]; j++)
            if ( col_idx_[j] == column_idx )
                return variable[j];
        return 0; // should not happen ...
    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT> &variable, const IT lil_idx) {
        IT row_idx = post_ranks_[lil_idx];
        auto beg = variable.begin()+row_begin_[row_idx];
        auto end = variable.begin()+row_begin_[row_idx+1];
        return std::vector<VT>(beg, end);
    }

    template <typename VT>
    inline std::vector< std::vector <VT> > get_matrix_variable_all(const std::vector<VT> &variable) {
        auto values = std::vector< std::vector <VT> >();
        for (unsigned int lil_idx=0; lil_idx < post_ranks_.size(); lil_idx++) {
            values.push_back(std::move(get_matrix_variable_row(variable, lil_idx)));
        }
        return values;
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

    // Returns size in bytes for connectivity
    virtual size_t size_in_bytes() {
        size_t size = 0;
        
        size += 2 * sizeof(IT);
        size += sizeof(ST);

        size += sizeof(std::vector<ST>);
        size += row_begin_.capacity() * sizeof(ST);

        size += sizeof(std::vector<IT>);
        size += col_idx_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        return size;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. Please note, that type casts are
     *              required to print-out the numbers encoded in unsigned char as numbers. 
     */
    void print_data_representation(int indent_spaces=0, bool print_container=true) {
        std::cout << std::string(indent_spaces, ' ') << "#rows: " << static_cast<unsigned long>(num_rows_) << std::endl;
        std::cout << std::string(indent_spaces, ' ') << "#columns: " << static_cast<unsigned long>(num_columns_) << std::endl;
        std::cout << std::string(indent_spaces, ' ') << "#nnz: " << num_non_zeros_ << std::endl;
        int empty_rows = 0;
        for (IT r = 0; r < post_ranks_.size()-1; r++ ) {
            if (post_ranks_[r+1]-post_ranks_[r] == 0)
                empty_rows++;
        }
        std::cout << std::string(indent_spaces, ' ') << "#empty rows: " << empty_rows << std::endl;
        if (print_container) {
            std::cout << std::string(indent_spaces, ' ') << "CSRMatrix instance at " << this << std::endl;
            std::cout << std::string(indent_spaces+2, ' ') << "post_ranks = [ ";
            for (IT r = 0; r < post_ranks_.size(); r++ ) {
                std::cout << static_cast<unsigned long>(post_ranks_[r]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "row_begin_ = [ ";
            for (auto i = 0; i < row_begin_.size(); i++ ) {
                std::cout << row_begin_[i] << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "col_idx_ = [ ";
            for (auto i = 0; i < col_idx_.size(); i++ ) {
                std::cout << static_cast<unsigned long>(col_idx_[i]) << " ";
            }
            std::cout << "]" << std::endl;
        }
    }
};
