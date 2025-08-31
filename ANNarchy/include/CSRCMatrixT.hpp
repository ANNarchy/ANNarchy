/*
 *    CSRCMatrixT.hpp
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
 *  @brief      A flipped CSRC representation.
 *  @details    By default is the connectivy matrix in ANNarchy post times pre. For spiking networks it can
 *              be beneficial for performance if the matrix is transposed.
 *  @tparam     IT      index data type, i. e. the type to represent the column or row values. Please note
 *                      that the maximum possible number of rows/columns should be representable.
 *  @tparam     ST      size data type, i. e. this type is used for number of nonzeros, or to encode the array
 *                      position. This value should be able to scope with a maximum of number of rows times
 *                      number of columns.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class CSRCMatrixT{
  protected:
    std::vector<IT> post_ranks_;    ///< required for accessors

    // intended as pre-synaptic view
    std::vector<ST> row_ptr_;       ///< i-th element marks the begin of the i-th row
    std::vector<IT> col_idx_;       ///< contains the column indices in row major order order. To access row i, get indices from row_begin_.

    // intended as post-synaptic view
    std::vector<ST> col_ptr_;       ///<
    std::vector<IT> row_idx_;
    std::vector<ST> inv_idx_;

    IT num_rows_;
    IT num_columns_;
    ST num_non_zeros_;

    /**
     *
     */
    void init_matrix_from_transposed_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::init_matrix_from_transposed_lil()" << std::endl;
    #ifdef _DEBUG_CONN
        auto d_row_it = row_indices.begin();
        auto d_col_it = column_indices.begin();
        for( ; d_row_it != row_indices.end(); d_row_it++, d_col_it++) {
            std::cout << "post_rank: " << *d_row_it << " with pre_ranks = [";
            for(auto it = d_col_it->begin(); it != d_col_it->end(); it++)
                std::cout << *it << " ";
            std::cout << "]" << std::endl;
        }
    #endif
    #endif
        assert( (row_indices.size() == column_indices.size()) );

        if (row_indices.empty())
            // nothing to do here ...
            return;

        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();

        // Build up CSR from LIL: scan all rows and fill col_idx with data (Notice the
        // flip of dimensions here!)
        for (auto row_idx = 0; row_idx < num_columns_; row_idx++) {
            row_ptr_[row_idx] = col_idx_.size();

            if ( (row_idx == *row_it) && (row_it != row_indices.end()) ) {
                col_idx_.insert(col_idx_.end(), col_it->begin(), col_it->end());
                row_it++;
                col_it++;
            }
        }
        row_ptr_[num_columns_] = col_idx_.size();
        num_non_zeros_ = col_idx_.size();

        // compute now the inverse view again (this forms the backward view)
        inverse_connectivity_matrix();

        // create post ranks array needed for accessors
        auto post_ranks = std::vector<IT>();
        for (auto i = 0; i < col_ptr_.size() - 1; i++) {
            ST row_len = col_ptr_[i+1] - col_ptr_[i];
            if (row_len > 0)
                post_ranks.push_back(i);
        }

        post_ranks_ = std::move(post_ranks);

        // remove unneccessary allocated space
        col_idx_.shrink_to_fit();
    }

 public:
    explicit CSRCMatrixT(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::CSRCMatrixT() with dense dimensions " << this->num_rows_ << " times " << this->num_columns_ << std::endl;
    #endif
        row_ptr_ = std::vector<ST>(num_columns_+1, 0);
        col_idx_ = std::vector<IT>();

        col_ptr_ = std::vector<ST>(num_rows_+1, 0);
        row_idx_ = std::vector<IT>();

        num_non_zeros_ = 0;
    }

    //
    //  Attribute accessors
    //
    inline ST* row_ptr() {
        return row_ptr_.data();
    }

    inline IT* col_idx() {
        return col_idx_.data();
    }

    inline ST* col_ptr() {
        return col_ptr_.data();
    }

    inline IT* row_idx() {
        return row_idx_.data();
    }

    inline IT* inverse_indices() {
        return inv_idx_.data();
    }

    /*
     *  Create CSRC_T from LIL while ensuring an ascending index in rows. This function is called from Python.
     */
    bool init_matrix_from_lil(std::vector<IT> post_ranks, std::vector< std::vector<IT> > pre_ranks) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::init_matrix_from_lil()" << std::endl;
    #endif
        // post_to_pre LIL
        auto lil_mat = new LILMatrix<IT>(num_rows_, num_columns_);
        lil_mat->init_matrix_from_lil(post_ranks, pre_ranks);

        // switch dimensions
        auto lil_mat_t = lil_mat->transpose();

        // sanity check
        if (lil_mat->nb_synapses() != lil_mat_t->nb_synapses())
            std::cerr << "Transpose of the LIL matrix went possibly wrong ..." << std::endl;

        // delete original LIL
        delete lil_mat;

        // Generate CSRC from transposed LIL
        init_matrix_from_transposed_lil(lil_mat_t->get_post_rank(), lil_mat_t->get_pre_ranks());

        // cleanup transposed LIL
        delete lil_mat_t;

        return true;
    }

    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @see        LILMatrix::init_matrix_from_lil()
     */
    template<typename VT, bool zero_based=true>
    std::vector<VT> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
    #ifdef _DEBUG
        std::cout << "LILMatrix::init_matrix_from_csv()" << std::endl;
    #endif
        auto tmp_col_idx = std::vector< std::vector < IT > >(num_rows_, std::vector<IT>());
        auto tmp_values = std::vector< std::vector < VT > >(num_rows_, std::vector<VT>());

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
        for(auto row = 0; row < num_rows_; row++) {

            if (tmp_col_idx[row].size() > 0) {
                lil_ranks.push_back(row);
                lil_col_idx.push_back(std::move(tmp_col_idx[row]));
                lil_values.push_back(std::move(tmp_values[row]));
            }
        }

        // create connectivity
        auto lil_mat = new LILMatrix<IT>(num_rows_, num_columns_);
        lil_mat->init_matrix_from_lil(lil_ranks, lil_col_idx);

        // switch dimensions
        auto lil_mat_t = lil_mat->transpose();

        // Generate CSRC from this LIL
        init_matrix_from_transposed_lil(lil_mat_t->get_post_rank(), lil_mat_t->get_pre_ranks());

        // create the value matrix
        auto value = this->template init_matrix_variable<VT>(0.0);
        this->template update_matrix_variable_all<VT>(value, lil_values);

        // cleanup
        delete lil_mat;
        delete lil_mat_t;

        return value;
    }

    /***
     *  @brief      generate a projection with fixed number pre pattern.
     *  @details    As we have no idea how many entries we will have per row we need to
     *              temporary create a LIL structure (post_to_re) and convert afterwards
     */
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::fixed_number_pre_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " nnz per row: " << nnz_per_row << std::endl;
    #endif
        post_ranks_ = post_ranks;

        // temporary container
        auto tmp_transposed_lil = std::vector<std::vector<IT>>(num_columns_, std::vector<IT>());

        // for each row we select a subset of the provided pre ranks
        for(auto lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            // shuffle indices (source vector is modified!)
            std::shuffle(pre_ranks.begin(), pre_ranks.end(), rng);

            // select nnz_per_row elements which are the
            // pre-synaptic entries
            for(int i = 0; i < nnz_per_row; i++) {
                tmp_transposed_lil[pre_ranks[i]].push_back(post_ranks[lil_idx]);
            }
        }

        // Create forward view (pre-synaptic rank as rows)
        num_non_zeros_ = 0;
        for(int r = 0; r < num_columns_; r++) {
            row_ptr_[r] = num_non_zeros_;
            if (tmp_transposed_lil[r].empty())
                continue;

            col_idx_.insert(col_idx_.end(), tmp_transposed_lil[r].begin(), tmp_transposed_lil[r].end());
            num_non_zeros_ += tmp_transposed_lil[r].size();
        }
        row_ptr_[num_columns_] = num_non_zeros_;

        // Create backward view (post-synaptic rank as rows)
        inverse_connectivity_matrix();

        // remove unneccessary allocated space
        col_idx_.shrink_to_fit();
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT, ST>(num_rows_, num_columns_);
        lil_mat->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng);

        // Generate CSRC_T from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;
    }

    /**
     *  @brief      computes the inverted view on the matrix
     */
    void inverse_connectivity_matrix() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::inverse_connectivity_matrix()" << std::endl;
    #endif
        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        auto inv_post_rank = std::map< IT, std::vector< IT > >();
        auto inv_idx = std::map< IT, std::vector< ST > >();

        // iterate over rows
        for( IT i = 0; i < this->num_columns_; i++ ) {
            ST row_begin = this->row_ptr_[i];
            ST row_end = this->row_ptr_[i+1];

            // iterate over synapses, update both result containers
            for( ST syn_idx = row_begin; syn_idx < row_end; syn_idx++ ) {
                inv_post_rank[this->col_idx_[syn_idx]].push_back(i);
                inv_idx[this->col_idx_[syn_idx]].push_back(syn_idx);
            }
        }

        // Ensure that the columns are sorted ascending
        // This is required for the accessors
        for (IT c =0; c < this->num_columns_; c++) {
            pairsort<IT, ST>(inv_post_rank[c].data(), inv_idx[c].data(), inv_post_rank[c].size() );
        }

        //
        // store as CSR
        //
        row_idx_.clear();
        inv_idx_.clear();
        ST curr_off = 0;

        // iterate over pre-neurons
        for (IT i = 0; i < this->num_rows_; i++) {
            col_ptr_[i] = curr_off;
            if ( !inv_post_rank[i].empty() ) {
                row_idx_.insert(row_idx_.end(), inv_post_rank[i].begin(), inv_post_rank[i].end());
                inv_idx_.insert(inv_idx_.end(), inv_idx[i].begin(), inv_idx[i].end());

                curr_off += static_cast<ST>(inv_post_rank[i].size());
            }
        }
        col_ptr_[this->num_rows_] = curr_off;

        // remove unneccessary allocated space
        row_idx_.shrink_to_fit();
        inv_idx_.shrink_to_fit();

        // sanity check
        if ( num_non_zeros_ != curr_off ) {
            std::cerr << "Something went wrong:" << std::endl;
            std::cerr << " - fwd dimensions: " << row_ptr_.size() << std::endl;
            std::cerr << " - bwd dimensions: " << col_ptr_.size() << std::endl;
            std::cerr << " - nb_synapes = " << num_non_zeros_ << " ( differs from curr_off = " << curr_off << ")" << std::endl;
        }
    #ifdef _DEBUG_CONN
        std::cout << "Pre to Post:" << std::endl;
        for (IT i = 0; i < this->num_rows_; i++) {
            std::cout << i << ": " << col_ptr_[i] << " -> " << col_ptr_[i+1] << std::endl;
        }
    #endif
    }

    //
    //  Connectivity
    //
    /*
     *  @brief      Get the neuron ranks for all existing dendrites.
     *  @details    As the matrix is internally flipped, we need to reconstruct the post-ranks.
     */
    std::vector<IT> get_post_rank() {
        return post_ranks_;
    }

    std::vector<std::vector<IT>> get_pre_ranks() {
        std::vector<std::vector<IT>> pre_ranks;

        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            pre_ranks.push_back(std::move(get_dendrite_pre_rank(lil_idx)));
        }

        return pre_ranks;
    }

    std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        auto rank = post_ranks_[lil_idx];
        auto beg = row_idx_.begin()+col_ptr_[rank];
        auto end = row_idx_.begin()+col_ptr_[rank+1];
        return std::vector<IT>(beg, end);
    }

    IT nb_dendrites() {
        return post_ranks_.size();
    }

    IT dendrite_size(IT lil_idx) {
        auto rank = post_ranks_[lil_idx];
        return col_ptr_[rank+1] - col_ptr_[rank];
    }

    ST nb_synapses() {
        return col_idx_.size();
    }

    std::map<IT, IT> nb_efferent_synapses() {
        auto num_efferents = std::map<IT, IT>();

        for (IT i = 0; i < this->num_columns_; i++) {
            if ((row_ptr_[i+1] - row_ptr_[i]) == 0)
                continue;

            num_efferents[i] = row_ptr_[i+1] - row_ptr_[i];
        }

        return num_efferents;
    }

    //
    //  Initialize variables
    //
    template <typename VT>
    std::vector<VT> init_matrix_variable(VT default_value) {
        return std::vector<VT>(num_non_zeros_, default_value);
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937 &rng) {
        std::uniform_real_distribution<VT> dis (a,b);

        auto var = std::vector<VT>(num_non_zeros_, 0.0);
        std::generate(var.begin(), var.end(), [&]{ return dis(rng); });

        return var;
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_normal(VT mean, VT sigma, std::mt19937 &rng) {
        std::normal_distribution<VT> dis (mean, sigma);

        auto var = std::vector<VT>(num_non_zeros_, 0.0);
        std::generate(var.begin(), var.end(), [&]{ return dis(rng); });

        return var;
    }

    //
    //  Update variables
    //
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT lil_idx, const IT col_idx, const VT value) {
        IT rank = post_ranks_[lil_idx];
        for (auto c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++) {
            if (row_idx_[c] == col_idx) {
                variable[inv_idx_[c]] = value;
            }
        }
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
        IT rank = post_ranks_[lil_idx];
        auto beg = data.begin();
        for (auto c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++, beg++) {
            variable[inv_idx_[c]] = *beg;
        }
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
        for (auto r = 0; r < post_ranks_.size(); r++) {
            IT rank = post_ranks_[r];
            auto beg = data[r].begin();
            for (auto c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++, beg++) {
                variable[inv_idx_[c]] = *beg;
            }
        }
    }

    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT> &variable, const IT lil_idx, const IT column_idx) {
        IT rank = post_ranks_[lil_idx];
        for (auto c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++) {
            if (row_idx_[c] == column_idx) {
                return variable[inv_idx_[c]];
            }
        }
        return 0.0; // should not happen ...
    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT> &variable, const IT lil_idx) {
        auto tmp = std::vector<VT>();
        IT rank = post_ranks_[lil_idx];
        for (IT c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++) {
            tmp.push_back(variable[inv_idx_[c]]);
        }
        return tmp;
    }

    template <typename VT>
    inline std::vector< std::vector <VT> > get_matrix_variable_all(const std::vector<VT> &variable) {
        auto values = std::vector< std::vector <VT> >();
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
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
     *  @brief      Initialize a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_vector_variable_all(std::vector<VT> &variable, std::vector<VT> values) {
        assert ( (variable.size() == values.size()) );

        std::copy(values.begin(), values.end(), variable.begin());
    }

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

    ~CSRCMatrixT() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::~CSRCMatrixT()" << std::endl;
    #endif
    }

    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::clear()" << std::endl;
    #endif
        std::fill(row_ptr_.begin(), row_ptr_.end(), 0);
        std::fill(col_ptr_.begin(), col_ptr_.end(), 0);
        col_idx_.clear();
        col_idx_.shrink_to_fit();
        row_idx_.clear();
        row_idx_.shrink_to_fit();
        num_non_zeros_ = 0;
    }

    // Returns size in bytes for connectivity
    virtual size_t size_in_bytes() {
        //constants
        size_t size = 3 * sizeof(unsigned int);

        // pre-synaptic
        size += row_ptr_.capacity() * sizeof(IT);
        size += col_idx_.capacity() * sizeof(IT);

        // post-synaptic
        size +=  col_ptr_.capacity() * sizeof(IT);
        size +=  row_idx_.capacity() * sizeof(IT);
        size +=  inv_idx_.capacity() * sizeof(IT);

        return size;
    }

    void print_data_representation(int indent_spaces=0, bool print_container=true) {
        std::cout << "CSRCMatrixT instance at " << this << std::endl;
        std::cout << "  #rows: " << num_columns_ << std::endl;
        std::cout << "  #columns: " << num_rows_ << std::endl;
        std::cout << "  #nnz: " << num_non_zeros_ << std::endl;

        int empty_columns = 0;
        for (IT r = 0; r < col_ptr_.size()-1; r++ ) {
            if (col_ptr_[r+1]-col_ptr_[r] == 0)
                empty_columns++;
        }

        if (print_container) {
            std::cout << std::string(indent_spaces, ' ') << "Forward view:" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "row_begin = [ ";
            for (IT r = 0; r < row_ptr_.size(); r++ ) {
                std::cout << static_cast<unsigned long>(row_ptr_[r]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "col_idx = [ ";
            for (auto i = 0; i < col_idx_.size(); i++ ) {
                std::cout << static_cast<unsigned long>(col_idx_[i]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces, ' ') << "Backward view:" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "col_begin = [ ";
            for (IT c = 0; c < col_ptr_.size(); c++ ) {
                std::cout << static_cast<unsigned long>(col_ptr_[c]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "row_idx = [ ";
            for (auto i = 0; i < row_idx_.size(); i++ ) {
                std::cout << static_cast<unsigned long>(row_idx_[i]) << " ";
            }
            std::cout << "]" << std::endl;

            std::cout << std::string(indent_spaces+2, ' ') << "inv_idx = [ ";
            for (auto i = 0; i < inv_idx_.size(); i++ ) {
                std::cout << static_cast<unsigned long>(inv_idx_[i]) << " ";
            }
            std::cout << "]" << std::endl;
        }else {
            std::cout << std::string(indent_spaces, ' ') << "#empty columns: " << empty_columns << std::endl;
        }

    }
};
