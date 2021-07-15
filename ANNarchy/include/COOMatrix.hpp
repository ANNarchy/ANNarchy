/*
 * COOMatrix.hpp
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

/**
 *  @brief      Implementation of the *coordinate* (COO) sparse matrix format.
 *  @details    The coordinate format is probably the easiest format to represent sparse connectivity.
 *              Each nonzero entry is represented as index pair and one additional array per variable.
 * 
 *              Let's consider the following example matrix
 *
 *                      | 0 1 0 |
 *                  A = | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 *
 *              Then we have two index arrays:
 * 
 *                  rows = [ 0, 1, 1, 3 ]
 * 
 *                  columns = [ 1, 0, 2, 3 ]
 * 
 *              While this offers benefits for computation, e. g. used in HYBMatrix format, there is a noticeable
 *              overhead for random access based on either row- or column index. To improve the performance
 *              we expect the entries in the list to be sorted by the row index.
 */
template<typename IT = unsigned int>
class COOMatrix {
  protected:
    const IT num_rows_;
    const IT num_columns_;

    std::vector<IT> post_ranks_;
    std::vector<IT> row_indices_;
    std::vector<IT> column_indices_;

  public:
    COOMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
    }

    COOMatrix(COOMatrix<IT>* other):
        num_rows_(other->num_rows_), num_columns_(other->num_columns_) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::copy constructor"<< std::endl;
    #endif
        this->post_ranks_ = other->post_ranks_;
        this->row_indices_ = other->row_indices_;
        this->column_indices_ = other->column_indices_;
    }

    ~COOMatrix() {
    #ifdef _DEBUG
        std::cout << "COOMatrix::~COOMatrix()" << std::endl;
    #endif
        clear();
    }

    inline IT* get_row_indices() {
        return row_indices_.data();
    }

    inline IT* get_column_indices() {
        return column_indices_.data();
    }

    /**
     *  @details    get row indices
     *  @returns    a list of row indices for all rows comprising of at least one element
     */
    std::vector<IT> get_post_rank() {
        return post_ranks_;
    }

    /**
     *  @brief      Get column indices
     *  @details    As described in the class' details we demand that entries are sorted by row. 
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() { 
        auto pre_ranks = std::vector<std::vector<IT>>();

        for ( int lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++ ) {
            pre_ranks.push_back(get_dendrite_pre_rank(lil_idx));
        }

        return pre_ranks; 
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        auto beg = std::find(row_indices_.begin(), row_indices_.end(), post_ranks_[lil_idx]);
        auto end = std::find(row_indices_.rbegin(), row_indices_.rend(), post_ranks_[lil_idx]);

        if ( (beg == row_indices_.end()) && (end == row_indices_.rend()) )
            return std::vector<IT>(); // empty row

        auto beg_idx = std::distance(row_indices_.begin(), beg);
        auto end_idx = std::distance(row_indices_.rend(), end) * -1;

        return std::vector<IT>(column_indices_.begin()+beg_idx, column_indices_.begin()+end_idx);
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses across all rows
     */
    unsigned int nb_synapses() {
        return row_indices_.size();
    }

    /**
     *  @details    returns the number of stored rows. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    unsigned int nb_dendrites() {
        return post_ranks_.size();
    }

    /**
     *  @details    returns the stored connections in this matrix for a given row. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    number of synapses across all rows of a given row.
     */
    IT dendrite_size(int lil_idx) {
        IT post_idx = post_ranks_[lil_idx];
        IT count = 0;
        for (auto it = row_indices_.cbegin(); it != row_indices_.cend(); it++) {
            if (*it == post_idx)
                count++;
        }

        return count;
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    simply sets the post_rank and pre_rank arrays without further sanity checking.
     */
    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::init_matrix_from_lil()" << std::endl;
    #endif
        assert( (post_ranks.size() == pre_ranks.size()) );
        clear();

        post_ranks_ = post_ranks;

        auto post_it = post_ranks.begin();
        auto pre_it = pre_ranks.begin();

        for( ; post_it != post_ranks.end(); post_it++, pre_it++) {
            // #elements in this row we need the post index
            auto tmp = std::vector<IT>(pre_it->size(), *post_it);

            row_indices_.insert(row_indices_.end(), tmp.begin(), tmp.end());
            column_indices_.insert(column_indices_.end(), pre_it->begin(), pre_it->end());
        }
    
    #ifdef _DEBUG
        std::cout << row_indices_.size() << " coordinate pairs created." << std::endl;

    #ifdef _DEBUG_CONN
        auto row_it = row_indices_.begin();
        auto col_it = column_indices_.begin();

        for( ; row_it != row_indices_.end(); row_it++, col_it++) {
            std::cout << "(" << *row_it << ", " << *col_it << ") ";
        }
        std::cout << std::endl;
    #endif
    #endif
    }

    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   the default value for all nonzeros in the matrix.
     *  @returns    initialized STL container
     */
    template <typename VT>
    std::vector< VT > init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        return std::vector<VT> (row_indices_.size(), default_value);
    }

    template <typename VT>
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::init_matrix_variable_uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        std::uniform_real_distribution<VT> dis (a,b);
        auto new_variable = std::vector<VT>(row_indices_.size(), 0.0);
        std::generate(new_variable.begin(), new_variable.end(), [&]{ return dis(rng); });
        return new_variable;
    }

        // ATTENTION: we assume sorted indices (otherwise the copy here is not correct)
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::update_matrix_variable_all()" << std::endl;
    #endif
        assert( (post_ranks_.size() == data.size()) );

        auto search_iter = row_indices_.begin();

        // HD (15th July 2021:)
        // normally we use a loop over update_matrix_variable_row() but in this case we would #rows time 
        // search from the start, which makes no sense.
        for (int lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            auto beg = std::find(search_iter, row_indices_.end(), post_ranks_[lil_idx]);

            auto beg_idx = std::distance(row_indices_.begin(), beg);
            std::copy(data[lil_idx].begin(), data[lil_idx].end(), variable.begin()+beg_idx);

            search_iter = (beg++); // start the next iteration on next element
        }
    }

        // ATTENTION: we assume sorted indices (otherwise the copy here is not correct)
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
    #ifdef _DEBUG
        std::cout << "COOMatrix::update_matrix_variable_row(" << lil_idx << ")" << std::endl;
    #endif
        assert( (lil_idx < post_ranks_.size()) );

        // find the slice to copy data to
        auto beg = std::find(row_indices_.begin(), row_indices_.end(), post_ranks_[lil_idx]);
        auto beg_idx = std::distance(row_indices_.begin(), beg);

        std::copy(data.begin(), data.end(), variable.begin()+beg_idx);
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
        std::cerr << "Not implemented" << std::endl;
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

        auto beg = std::find(row_indices_.begin(), row_indices_.end(), post_ranks_[lil_idx]);
        auto end = std::find(row_indices_.rbegin(), row_indices_.rend(), post_ranks_[lil_idx]);

        if ( (beg == row_indices_.end()) && (end == row_indices_.rend()) )
            return std::vector<VT>(); // empty row

        auto beg_idx = std::distance(row_indices_.begin(), beg);
        auto end_idx = std::distance(row_indices_.rend(), end) * -1; // reversed!

        return std::vector<VT>(variable.begin()+beg_idx, variable.begin()+end_idx);
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

    void clear() {
    #ifdef _DEBUG
        std::cout << "COOMatrix::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();
        row_indices_.clear();
        row_indices_.shrink_to_fit();
        column_indices_.clear();
        column_indices_.shrink_to_fit();
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        size_t size = 2 * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += row_indices_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += column_indices_.capacity() * sizeof(IT);

        return size;
    }
};