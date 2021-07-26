/*
 *
 *    CSRCMatrixT.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *    Julien Vitay <julien.vitay@gmail.com>
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

    // intended as pre-synaptic view
    std::vector<ST> row_ptr_;       ///< i-th element marks the begin of the i-th row
    std::vector<IT> col_idx_;       ///< contains the column indices in row major order order. To access row i, get indices from row_begin_.

    // intended as post-synaptic view
    std::vector<IT> post_ranks_;    ///< required for accessors
    std::vector<ST> col_ptr_;       ///< 
    std::vector<IT> row_idx_;
    std::vector<IT> inv_idx_;

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
    }

 public:
    explicit CSRCMatrixT(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {

        row_ptr_ = std::vector<ST>(num_columns_+1, 0);
        col_idx_ = std::vector<IT>();

        col_ptr_ = std::vector<ST>(num_rows_+1, 0);
        row_idx_ = std::vector<IT>();

        num_non_zeros_ = 0;
    }

    /*
     *  Create CSRC_T from LIL while ensuring an ascending index in rows. This function is called from Python.
     */
    void init_matrix_from_lil(std::vector<IT> post_ranks, std::vector< std::vector<IT> > pre_ranks) {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::init_matrix_from_lil()" << std::endl;
    #endif
        clear();

        // post_to_pre LIL
        auto lil_mat = new LILMatrix<IT>(num_rows_, num_columns_);
        lil_mat->init_matrix_from_lil(post_ranks, pre_ranks);

        // switch dimensions
        auto lil_mat_t = lil_mat->transpose();

        // sanity check
        if (lil_mat->nb_synapses() != lil_mat_t->nb_synapses())
            std::cerr << "Transpose of the LIL matrix went possibly wrong ..." << std::endl;

        // Generate CSRC from this LIL
        init_matrix_from_transposed_lil(lil_mat_t->get_post_rank(), lil_mat_t->get_pre_ranks());

        // cleanup
        delete lil_mat;
        delete lil_mat_t;
    }

    /***
     *  @brief      generate a projection with fixed number pre pattern.
     *  @details    As we have no idea how many entries we will have per row we need to
     *              temporary create a LIL structure (post_to_re) and convert afterwards
     */
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::mt19937& rng) {
        clear();

        // Generate post_to_pre LIL (need to switch row/columns!)
        auto lil_mat = new LILMatrix<IT>(num_rows_, num_columns_);
        
        // execute the pattern generation
        lil_mat->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // switch dimensions
        auto lil_mat_t = lil_mat->transpose();

        // sanity check
        if (lil_mat->nb_synapses() != lil_mat_t->nb_synapses())
            std::cerr << "Transpose of the LIL matrix went possibly wrong ..." << std::endl;

        // Generate CSRC from this LIL
        init_matrix_from_transposed_lil(lil_mat_t->get_post_rank(), lil_mat_t->get_pre_ranks());

        // cleanup
        delete lil_mat;
        delete lil_mat_t;
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections) {
        clear();

        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT>();
        lil_mat->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections);

        // Generate CSRC_T from this LIL
        init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;
    }

    // DIRECTLY COPIED from CSRCMatrixInv
    void inverse_connectivity_matrix() {
        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        auto inv_post_rank = std::map< int, std::vector< int > >();
        auto inv_idx = std::map< int, std::vector< int > >();

        // iterate over rows
        for( int i = 0; i < this->num_columns_; i++ ) {
            int row_begin = this->row_ptr_[i];
            int row_end = this->row_ptr_[i+1];

            // iterate over synapses, update both result containers
            for( int syn_idx = row_begin; syn_idx < row_end; syn_idx++ ) {
                inv_post_rank[this->col_idx_[syn_idx]].push_back(i);
                inv_idx[this->col_idx_[syn_idx]].push_back(syn_idx);
            }
        }

        // Ensure that the columns are sorted ascending
        // This is required for the accessors
        for(unsigned int c =0; c < this->num_columns_; c++) {
            pairsort<IT, IT>(inv_post_rank[c].data(), inv_idx[c].data(), inv_post_rank[c].size() );
        }

        //
        // store as CSR
        //
        row_idx_.clear();
        inv_idx_.clear();
        int curr_off = 0;

        // iterate over pre-neurons
        for ( int i = 0; i < this->num_rows_; i++) {
            col_ptr_[i] = curr_off;
            if ( !inv_post_rank[i].empty() ) {
                row_idx_.insert(row_idx_.end(), inv_post_rank[i].begin(), inv_post_rank[i].end());
                inv_idx_.insert(inv_idx_.end(), inv_idx[i].begin(), inv_idx[i].end());

                curr_off += inv_post_rank[i].size();
            }
        }
        col_ptr_[this->num_rows_] = curr_off;

        if ( num_non_zeros_ != curr_off ) {
            std::cerr << "Something went wrong:" << std::endl;
            std::cerr << " - fwd dimensions: " << row_ptr_.size() << std::endl;
            std::cerr << " - bwd dimensions: " << col_ptr_.size() << std::endl;
            std::cerr << " - nb_synapes = " << num_non_zeros_ << " ( differs from curr_off = " << curr_off << ")" << std::endl;
        }
    #ifdef _DEBUG_CONN
        std::cout << "Pre to Post:" << std::endl;
        for ( int i = 0; i < this->num_rows_; i++ ) {
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
        int rank = post_ranks_[lil_idx];
        for(int c = col_ptr_[rank]; c < col_ptr_[rank+1]; c++) {
            tmp.push_back(variable[inv_idx_[c]]);
        }
        return tmp;
    }

    template <typename VT>
    inline std::vector< std::vector <VT> > get_matrix_variable_all(const std::vector<VT> &variable) {
        auto values = std::vector< std::vector <VT> >();
        for(auto it = post_ranks_.begin(); it != post_ranks_.end(); it++) {
            values.push_back(std::move(get_matrix_variable_row(variable, *it)));
        }
        return values;
    }

    ~CSRCMatrixT() {
    #ifdef _DEBUG
        std::cout << "CSRCMatrixT::~CSRCMatrixT()" << std::endl;
    #endif
        clear();
    }

    void clear() {
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
    size_t size_in_bytes() {
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

    void print_structure() {
        std::cout << "CSRCMatrixT instance at " << this << std::endl;
        std::cout << "  #rows: " << num_columns_ << std::endl;
        std::cout << "  #columns: " << num_rows_ << std::endl;
        std::cout << "  #nnz: " << num_non_zeros_ << std::endl;
    }
};
