/*
 *    DiaMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2024  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = bool>
class DiaMatrix{
protected:
    const IT num_rows_;             ///< maximum number of rows
    const IT num_columns_;          ///< maximum number of columns

    std::vector<IT> post_ranks_;
    std::map<IT, IT> offsets_;
    std::vector<std::vector<MT>> diagonals_;

public:
    explicit DiaMatrix(IT num_rows, IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {

    }

    /**
     *  @brief      Destructor
     *  @details    calls the DiaMatrix::clear method. Is not declared as virtual as inheriting classes in our
     *              framework should never be destroyed by the base pointer.
     */
    ~DiaMatrix() {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::~DiaMatrix()" << std::endl;
    #endif
        clear();
    }

    /**
     *  @brief      Clear the sparse matrix representation.
     *  @details    Clears the connectivity data stored in the *post_rank* and *pre_rank* STL containers and free
     *              the allocated memory. **Important**: allocated variables are not effected by this!
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        for (auto it = diagonals_.begin(); it != diagonals_.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }

        offsets_.clear();

        diagonals_.clear();
        diagonals_.shrink_to_fit();
    }

    /**
     *  @details    get row indices
     *  @returns    a list of row indices for all rows comprising of at least one element
     */
    std::vector<IT> get_post_rank() { return post_ranks_; }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() {

        if (post_ranks_.size() == num_rows_) {
            // no empty rows
            auto pre_ranks = std::vector<std::vector<IT>>(num_rows_, std::vector<IT>());
            for (auto map_it = offsets_.begin(); map_it != offsets_.end(); map_it++) {
                for (IT row_idx = 0; row_idx < num_rows_; row_idx++) {
                    if (diagonals_[map_it->second][row_idx]) {
                        pre_ranks[row_idx].push_back(map_it->first + row_idx);
                    }
                }
            }
            return pre_ranks;
        }else{
            // the matrix contains empty rows
            std::cerr << "Not implemented: Diagonal format and post-synaptic PopulationViews." << std::endl;

            auto pre_ranks = std::vector<std::vector<IT>>();
            return pre_ranks;
        }

        return std::vector<std::vector<IT>>();
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        // TODO: it's not efficient to build up the complete pre_ranks and then
        //       return only a single line from it (but I have no better idea yet ...)
        auto pre_rank = get_pre_ranks();
        // sanity check
        assert( (lil_idx < pre_rank.size()) );

        return pre_rank[lil_idx];
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses in the whole matrix.
     */
    ST nb_synapses() {
        ST size = 0;
        auto pre_rank = get_pre_ranks();
        for(auto it = pre_rank.begin(); it != pre_rank.end(); it++) {
            size += it->size();
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
        assert( (lil_idx < post_ranks_.size()) );

        return static_cast<IT>(get_dendrite_pre_rank(lil_idx).size());
    }

    /**
     *  @brief      Get the number of stored rows.
     *  @details    The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    IT nb_dendrites() {
        return static_cast<IT>(post_ranks_.size());
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    simply sets the post_rank and pre_rank arrays without further sanity checking.
     */
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::init_matrix_from_lil()" << std::endl;
    #endif

        // Sanity check
        assert( (post_ranks.size() == pre_ranks.size()) );

        post_ranks_ = post_ranks;
        // gather data
        auto tmp = std::map<IT, std::vector<MT>>();
        for (IT i = 0; i < post_ranks.size(); i++) {
            IT row_idx = post_ranks[i];

            for (IT j = 0; j < pre_ranks[i].size(); j++) {
                IT col_idx = pre_ranks[i][j];
                IT off = col_idx - row_idx;

                if (tmp.find(off) == tmp.end()) {
                    tmp[off] = std::vector<MT>(num_rows_, static_cast<MT>(false));
                }

                tmp[off][row_idx] = static_cast<MT>(true);
            }
        }

        offsets_.clear();
        diagonals_.clear();
        IT vec_idx = 0;
        auto map_it = tmp.begin();
        for ( ; map_it != tmp.end(); map_it++, vec_idx++) {
            offsets_[map_it->first] = vec_idx;
            diagonals_.push_back(map_it->second);
        }

    #ifdef _DEBUG
        this->print_data_representation();
    #endif

        return true;
    }

    template <typename VT>
    std::vector<std::vector< VT >> init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        // fill all places with 0
        auto variable = std::vector<std::vector<VT>> (offsets_.size(), std::vector<VT>(num_columns_, static_cast<VT>(0.0)));

        for (auto map_it = offsets_.begin(); map_it != offsets_.end(); map_it++) {
            for (IT row_idx = 0; row_idx < num_columns_; row_idx++) {
                if (diagonals_[map_it->second][row_idx]) {
                    variable[map_it->second][row_idx] = default_value;
                }
            }
        }

        return variable;
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
    std::vector<std::vector<VT>> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        std::uniform_real_distribution<VT> dis (a,b);

        // fill all places with 0
        auto variable = std::vector<std::vector<VT>> (offsets_.size(), std::vector<VT>(num_columns_));

        for (auto map_it = offsets_.begin(); map_it != offsets_.end(); map_it++) {
            for (IT row_idx = 0; row_idx < num_columns_; row_idx++) {
                if (diagonals_[map_it->second][row_idx]) {
                    variable[map_it->second][row_idx] = dis(rng);
                }
            }
        }

        return variable;
    }

    /**
     *  @details    Updates all matrix values based on a LIL representation
     *  @tparam     VT              data type of the variable.
     *  @param[in]  variable        Diagonal variable container
     *  @param[in]  data            LIL variable container
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<std::vector<VT>> &variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::update_matrix_variable_all()" << std::endl;
    #endif
        assert( (post_ranks_.size() == data.size()) );

        for (IT i = 0; i < post_ranks_.size(); i++) {
            update_matrix_variable_row(variable, i, data[i]);
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
    inline void update_matrix_variable_row(std::vector<std::vector<VT>> &variable, const IT lil_idx, const std::vector<VT> data) {
    #ifdef _DEBUG
        std::cout << "DiaMatrix::update_matrix_variable_row(lil_idx = " << lil_idx << ")" << std::endl;
    #endif
        auto pre_ranks = get_dendrite_pre_rank(lil_idx);
        assert( (pre_ranks.size() == data.size()) );

        for (IT i = 0; i < pre_ranks.size(); i++) {
            update_matrix_variable(variable, lil_idx, pre_ranks[i], data[i]);
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
    inline void update_matrix_variable(std::vector<std::vector<VT>> &variable, const IT lil_idx, const IT column_idx, const VT value) {
        assert( (lil_idx < post_ranks_.size()) );

        IT row_idx = post_ranks_[lil_idx];
        IT diag_idx = offsets_[column_idx - row_idx];

        variable[diag_idx][row_idx] = value;
    }

    /**
     *  @brief      retrieve a LIL representation for a given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @returns    a LIL representation from the given variable.
     */
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector< std::vector<VT> > &variable) {
        auto values = std::vector< std::vector < VT > >();

        for (IT i = 0; i < post_ranks_.size(); i++) {
            values.push_back(get_matrix_variable_row(variable, i));
        }
        return values;
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface to retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a vector containing all elements of the provided variable and lil_idx
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector< std::vector<VT> >& variable, const IT &lil_idx) {
        auto values = std::vector< VT >();

        // retrieve dense indices
        IT row_idx = post_ranks_[lil_idx];
        auto pre_ranks = get_dendrite_pre_rank(lil_idx);

        for (auto col_it = pre_ranks.begin(); col_it != pre_ranks.end(); col_it++) {
            IT off_idx = offsets_[*col_it - row_idx];
            values.push_back(variable[off_idx][row_idx]);
        }

        assert( (values.size() == pre_ranks.size()) );

        return values;
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
    inline VT get_matrix_variable(const std::vector< std::vector<VT> >& variable, const IT &lil_idx, const IT &col_idx) {
        assert ( (lil_idx < variable.size()) );


        return static_cast<VT>(0.0); // should not happen
    }

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    size_t size_in_bytes() {
        size_t size = 2 * sizeof(IT);               // scalar values

        // post_ranks
        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        // diagonal offsets
        size += sizeof(std::map<IT, IT>);
        //size += offsets_.capacity() * sizeof(IT);

        // diagonals data
        size += sizeof(std::vector<std::vector<MT>>);
        size += diagonals_.capacity() * sizeof(std::vector<MT>);
        for( auto it = diagonals_.cbegin(); it != diagonals_.cend(); it++ )
            size += it->capacity() * sizeof(MT);

        return size;
    }

    /**
     *  @brief      print some matrix characteristics to the standard out (i. e. command-line)
     */
    void print_matrix_statistics() {
        std::cout << "  #rows: " << num_rows_ << std::endl;
        std::cout << "  #columns: " << num_columns_ << std::endl;
        std::cout << "  #nnz: " << nb_synapses() << std::endl;
        std::cout << "  #empty rows: " << num_rows_ - nb_dendrites() << std::endl;
    }

    void print_data_representation() {
        std::cout << "DiaMatrix instance at " << this << std::endl;
        print_matrix_statistics();

        std::cout << "  post_ranks = [ ";
        for (auto it = post_ranks_.begin(); it != post_ranks_.end(); it++) {
            std::cout << *it << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  #diagonals:" << offsets_.size() << std::endl;
        // for each diagonal depict: offset/bool mask
        for (auto map_it = offsets_.begin(); map_it != offsets_.end(); map_it++) {
            std::cout << "  " << map_it->first << " = [";
            for (auto col_it = diagonals_[map_it->second].begin(); col_it != diagonals_[map_it->second].end(); col_it++) {
                std::cout << static_cast<int>(*col_it) << ",";
            }
            std::cout << "]" << std::endl;
        }
    }

    template<typename VT>
    void print_variable(const std::vector<std::vector<VT>> &variable) {
        std::cout << "Variable instance (" << &variable << ") stored as DiaMatrix (" << this << "):" << std::endl;

        // for each diagonal depict: offset/bool mask
        for (IT ndiags = 0; ndiags < offsets_.size(); ndiags++) {
            std::cout << "  " << offsets_[ndiags] << " = [";
            for (auto col_it = variable[ndiags].begin(); col_it != variable[ndiags].end(); col_it++) {
                std::cout << *col_it << ",";
            }
            std::cout << "]" << std::endl;
        }
    }
};
