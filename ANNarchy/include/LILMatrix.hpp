/*
 *
 *    LILMatrix.hpp
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
 *
 */
#pragma once

#include "helper_functions.hpp"

/**
 *  @brief      Implementation of the *list-in-list* (LIL) sparse matrix format.
 *  @details    The LIL format comprises of a nested vector *pre_rank*, where the top-level indicates a row and the sub-level vector
 *              the column indices. To consider the existance of empty-rows, we have an additional array *post_rank* who assigns the
 *              row index to the entry in the *pre_rank* structure.
 *
 *              Let's consider the following example matrix
 *
 *                      | 0 1 0 |
 *                  A = | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 *
 *              As all rows, except the second, contain at least one entry the *post_rank* array would be:
 *
 *                  post_rank = [ 0 1 3 ]
 *
 *              The *pre_rank* array contains in each sub vector the column indices:
 *
 *                  pre_rank = [ [ 1 ], [ 0, 2  ], [ 2 ] ]
 *
 *              Please note, that contrary to some SpMV implementations in all ANNarchy versions the values of the matrix are stored in
 *              a seperate LIL structure which follows the same scheme:
 *
 *                  w = [ [ 1 ], [ 2, 3 ], [ 4 ] ]
 *
 *              The background is simply the fact that we have often multiple variables and one want to store only one time pre- and post-synaptic ranks.
 *              Such a variable is initialized with the init_matrix_variable<> methods. A second special case is that some operations require only a row
 *              vector of size (*num_rows_*) these variables are initialized with a seperate init_row_variable<> methods.
 *
 *              The Python interface of ANNarchy can access the connectivity data through the following functions:
 *
 *              - LILMatrix<IT>::get_post_rank()
 *              - LILMatrix<IT>::get_pre_ranks()
 *              - LILMatrix<IT>::get_dendrite_pre_rank()
 *              - LILMatrix<IT>::nb_synapses()
 *              - LILMatrix<IT>::nb_dendrites()
 *
 *              The template class is also responsible for the init/get/update of variables marked as *semiglobal* and *local*. A *local* variable can be
 *              filled with either constant or randomly drawn values, please note the specialized functions for more details:
 *
 *              - LILMatrix<IT>::init_matrix_variable() for a (at maximum) num_rows_ by num_column_ matrix with a constant value
 *              - LILMatrix<IT>::init_matrix_variable_uniform() for a (at maximum) num_rows_ by num_column_ matrix with a constant value
 *              - LILMatrix<IT>::init_matrix_variable_normal() for a (at maximum) num_rows_ by num_column_ matrix with a constant value
 *
 *              The *local* variables can be updated by single values, a row or the complete variable. Please note, that a *lil_idx* is required for two of
 *              these functions which could be obtained by usage of the LILMatrix::post_rank array.
 *
 *              - LILMatrix<IT>::update_matrix_variable()
 *              - LILMatrix<IT>::update_matrix_variable_row()
 *              - LILMatrix<IT>::update_matrix_variable_all()
 *
 *              The same applies for the read-out of the allocated variable containers:
 *
 *              - LILMatrix<IT>::get_matrix_variable()
 *              - LILMatrix<IT>::get_matrix_variable_row()
 *              - LILMatrix<IT>::get_matrix_variable_all()
 *
 *              All these methods are also available as vector_variable methods which are responsible for the *semiglobal* variables. Contrary to local variables
 *              those are stored in 1D-vectors of the same size as LILMatrix::post_rank.
 *
 *              **Implementation notice**: this functions might appear as obsolete, as they only return the provided containers back, but they are part of a common
 *              interface for single-thread and multi-thread interface (this again eases the code generation).
 *
 *  @tparam     IT      data type to represent the ranks within the matrix. Generally unsigned data types should be chosen.
 *                      The data type determines the maximum size for the number of elements in a column respectively the number
 *                      of rows encoded in the matrix:
 *
 *                      - unsigned char (1 byte):        [0 .. 255]
 *                      - unsigned short int (2 byte):   [0 .. 65.535]
 *                      - unsigned int (4 byte):         [0 .. 4.294.967.295]
 *
 *                      The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and ::num_columns_)
 *
 *              ST      the second type should be used if the index type IT could overflow. For instance, the nb_synapses method should return ST as
 *                      the maximum value in case a full dense matrix would be IT times IT entries.
 */
template<typename IT = unsigned int, typename ST = unsigned long int>
class LILMatrix {
public:
    const IT num_rows_;                     ///< maximum number of rows which equals the maximum length of post_rank as well as maximum size of top-level of pre_rank.
    const IT num_columns_;                  ///< maximum number of columns which equals the maximum available size in the sub-level vectors.

    std::vector<IT> post_rank;              ///< indices of existing rows
    std::vector<std::vector<IT> > pre_rank; ///< column indices sorted by rows

public:
    /**
     *  @brief      Constructor
     *  @details    Does **not** initialize any data of the matrix. It's just to store the maximum dimension
     *              for post-synaptic (num_rows) and pre-synaptic (num_columns).
     */
    explicit LILMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
        assert ( (static_cast<unsigned long int>(num_rows) <= static_cast<unsigned long int>(std::numeric_limits<IT>::max())) );
        assert ( (static_cast<unsigned long int>(num_columns) <= static_cast<unsigned long int>(std::numeric_limits<IT>::max())) );

    #ifdef _DEBUG
        std::cout << "LILMatrix::LILMatrix() with dense dimensions " << static_cast<long>(this->num_rows_) << " times " << static_cast<long>(this->num_columns_) << std::endl;
    #endif
    }

    /**
     *  @brief      Destructor
     *  @details    calls the LILMatrix::clear method. Is not declared as virtual as inheriting classes in our
     *              framework should never be destroyed by the base pointer.
     */
    ~LILMatrix() {
    #ifdef _DEBUG
        std::cout << "LILMatrix::~LILMatrix()" << std::endl;
    #endif
    }

    /**
     *  @brief      Clear the sparse matrix representation.
     *  @details    Clears the connectivity data stored in the *post_rank* and *pre_rank* STL containers and free
     *              the allocated memory. **Important**: allocated variables are not effected by this!
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "LILMatrix::clear()" << std::endl;
    #endif
        post_rank.clear();
        post_rank.shrink_to_fit();

        pre_rank.clear();
        pre_rank.shrink_to_fit();
    }

    /**
     *  @details    get row indices
     *  @returns    a list of row indices for all rows comprising of at least one element
     */
    std::vector<IT> get_post_rank() { return post_rank; }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() { return pre_rank; }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        assert( (lil_idx < pre_rank.size()) );

        return pre_rank[lil_idx];
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses in the whole matrix.
     */
    ST nb_synapses() {
        ST size = 0;
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
        assert( (lil_idx < pre_rank.size()) );

        return static_cast<IT>(pre_rank[lil_idx].size());
    }

    /**
     *  @brief      Get the number of stored rows.
     *  @details    The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    IT nb_dendrites() {
        return static_cast<IT>(post_rank.size());
    }

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.
     *  @details    simply sets the post_rank and pre_rank arrays without further sanity checking.
     */
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "LILMatrix::init_matrix_from_lil()" << std::endl;
    #endif

        // Sanity checks
        assert ( (post_ranks.size() == pre_ranks.size()) );
        assert ( (post_ranks.size() <= num_rows_) );

        // store the data
        this->post_rank = post_ranks;
        this->pre_rank = pre_ranks;

    #ifdef _DEBUG
        print_matrix_statistics();
    #endif
        return true;
    }

    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @details    this function creates also the variable array which is usually created in a separate
     *              function call afterwards.
     *  @tparam     VT          value type of the nonzero entries
     *  @tparam     zero_based  set to true if the contained data in csv has as minimum possible index 0. If
     *                          set to false, the read-in indices will be decremented by 1.
     */
    template<typename VT, bool zero_based=true>
    std::vector<std::vector<VT>> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
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
        init_matrix_from_lil(lil_ranks, lil_col_idx);

        // create the value matrix
        auto value = init_matrix_variable<VT>(0.0);
        update_matrix_variable_all<VT>(value, lil_values);

        return value;
    }

    /**
     *  @brief      initialize connectivity using an all-to-all pattern.
     *  @details    For more details on this pattern see the ANNarchy Documentation.
     *  @param[in]  post_ranks              list of row indices of all rows which contain at least on elements to be accounted.
     *  @param[in]  pre_ranks               list of list, where the i-th sub-vector should contain a list of potential
     *                                      connection candidates for the i-th post-synaptic neuron.
     *  @param[in]  allow_self_connections  whether neurons with the same rank should be connected or not.
     */
    bool all_to_all_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, bool allow_self_connections) {
    #ifdef _DEBUG
        std::cout << "LILMatrix::all_to_all_pattern()" << std::endl;
        std::cout << " #rows: " << post_ranks.size() << std::endl;
        std::cout << " #columns: " << pre_ranks.size() << std::endl;
        std::cout << " self-connections: " << allow_self_connections << std::endl;
    #endif
        post_rank = post_ranks;

        if (allow_self_connections) {
            // copy N times the pre-rank vector
            pre_rank = std::vector< std::vector<IT> >(post_rank.size(), pre_ranks);

        } else {
            pre_rank = std::vector< std::vector<IT> >(post_rank.size(), std::vector<IT>());

            // need to remove the i-th column index for row i
            for (IT i = 0; i < post_ranks.size(); i++) {
                pre_rank[i].insert(pre_rank[i].begin(), pre_ranks.begin(), pre_ranks.begin()+i);
                pre_rank[i].insert(pre_rank[i].begin()+i, pre_ranks.begin()+i+1, pre_ranks.end());
            }
        }

        return true;
    }

    /**
     *  @brief      initialize connectivity using a fixed_number_pre pattern
     *  @details    For more details on this pattern see the ANNarchy Documentation.
     *  @param[in]  post_ranks  list of row indices of all rows which contain at least on elements to be accounted.
     *  @param[in]  pre_ranks   list of list, where the i-th sub-vector should contain a list of potential connection candidates for the i-th post-synaptic neuron.
     *  @param[in]  nnz_per_row number of pre-synaptic neurons which should be randomly selected from the list.
     *  @param[in]  rng         a merseanne twister generator (need to be seeded in prior if necessary)
     */
    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILMatrix::fixed_number_pre_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " nnz per row: " << nnz_per_row << std::endl;
    #endif
        post_rank = post_ranks;
        pre_rank = std::vector< std::vector<IT> >(post_rank.size(), std::vector<IT>());

        // for each row we select a subset of the provided pre ranks
        for(auto lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            // shuffle indices (source vector is modified!)
            std::shuffle(pre_ranks.begin(), pre_ranks.end(), rng);

            // select nnz_per_row elements
            auto tmp_col_indices = std::vector<IT>(pre_ranks.begin(), pre_ranks.begin()+nnz_per_row);

            // sort the indices before storage
            std::sort(tmp_col_indices.begin(), tmp_col_indices.end());
            pre_rank[lil_idx] = std::move(tmp_col_indices);
        }

        return true;
    }

    /**
     *  @brief      initialize connectivity using a fixed_probability pattern
     *  @details    For more details on this pattern see the ANNarchy Documentation.
     *  @param[in]  post_ranks  list of row indices of all rows which contain at least on elements to be accounted.
     *  @param[in]  pre_ranks   list of list, where the i-th sub-vector should contain a list of potential connection candidates for the i-th post-synaptic neuron.
     *  @param[in]  p           probability for a connection being set between two neurons.
     *  @param[in]  rng         a merseanne twister generator (need to be seeded in prior if necessary)
     */
    bool fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILMatrix::fixed_probability_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " p: " << p << std::endl;
        std::cout << " self_connections: " << allow_self_connections << std::endl;
    #endif
        auto dis = std::uniform_real_distribution<double>(0.0, 1.0);

        post_rank = post_ranks;
        pre_rank = std::vector< std::vector<IT> >(post_rank.size(), std::vector<IT>());

        for(auto lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            // only relevant if allow_self_connections == false
            IT rk_post = post_ranks[lil_idx];

            // over all possible connections
            for(auto it=pre_ranks.begin(); it != pre_ranks.end(); it++) {
                if ( (!allow_self_connections) && (rk_post == *it) )
                    continue;

                if (dis(rng) < p)
                    pre_rank[lil_idx].push_back(*it);
            }

            // free unnecessary allocated memory
            pre_rank[lil_idx].shrink_to_fit();
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
    std::vector< std::vector<VT> > init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with constant " << default_value << std::endl;
    #endif
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());

        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), default_value);
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
    std::vector<std::vector<VT>> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        std::uniform_real_distribution<VT> dis (a,b);
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());
        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), 0.0);
            std::generate(new_variable[post].begin(), new_variable[post].end(), [&]{ return dis(rng); });
        }
        return new_variable;
    }

    /**
     *  @details    Allocates and initialize a num_rows_ by num_columns_ matrix based on the stored
     *              connectivity and where the nonzero values serves an discrete uniform distribution (a, b).
     *  @tparam     VT      data type of the variable (should be an Integer-like data type).
     *  @param[in]  a       minimum of the distribution
     *  @param[in]  b       maximum of the distribution
     *  @param[in]  rng     a merseanne twister generator (need to be seeded in prior if necessary)
     *  @todo       Maybe we could use template specialization instead of a seperate function
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<std::vector<VT>> init_matrix_variable_discrete_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with discrete Uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        std::uniform_int_distribution<VT> dis (a,b);
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());
        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), 0.0);
            std::generate(new_variable[post].begin(), new_variable[post].end(), [&]{ return dis(rng); });
        }
        return new_variable;
    }

    /**
     *  @details    Allocates and initialize a num_rows_ by num_columns_ matrix based on the stored
     *              connectivity and where the nonzero values serves a normal distribution (mean, sigma).
     *  @tparam     VT      data type of the variable (should be an Integer-like data type).
     *  @param[in]  mean    mean of the distribution
     *  @param[in]  sigma   sigma of the distribution
     *  @param[in]  rng     a merseanne twister generator (need to be seeded in prior if necessary)
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<std::vector<VT>> init_matrix_variable_normal(VT mean, VT sigma, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with normal distribution (" << mean << ", " << sigma << ")" << std::endl;
    #endif
        std::normal_distribution<VT> dis (mean, sigma);
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());
        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), 0.0);
            std::generate(new_variable[post].begin(), new_variable[post].end(), [&]{ return dis(rng); });
        }
        return new_variable;
    }

    /**
     *  @details    Allocates and initialize a num_rows_ by num_columns_ matrix based on the stored
     *              connectivity and where the nonzero values serves a log-normal distribution (mean, sigma).
     *  @tparam     VT      data type of the variable (should be an Integer-like data type).
     *  @param[in]  mean    mean of the distribution
     *  @param[in]  sigma   sigma of the distribution
     *  @param[in]  rng     a merseanne twister generator (need to be seeded in prior if necessary)
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<std::vector<VT>> init_matrix_variable_log_normal(VT mean, VT sigma, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with log-normal distribution (" << mean << ", " << sigma << ")" << std::endl;
    #endif
        std::lognormal_distribution<VT> dis (mean, sigma);
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());
        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), 0.0);
            std::generate(new_variable[post].begin(), new_variable[post].end(), [&]{ return dis(rng); });
        }
        return new_variable;
    }

    /**
     *  @details    Allocates and initialize a num_rows_ by num_columns_ matrix based on the stored
     *              connectivity and where the nonzero values serves a log-normal distribution (mean, sigma).
     *              This function additionaly clips the generated values within the interval [min,max]
     *  @tparam     VT      data type of the variable (should be an Integer-like data type).
     *  @param[in]  mean    mean of the distribution
     *  @param[in]  sigma   sigma of the distribution
     *  @param[in]  rng     a merseanne twister generator (need to be seeded in prior if necessary)
     *  @param[in]  min     minimum border
     *  @param[in]  max     maximum border
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<std::vector<VT>> init_matrix_variable_log_normal_clip(VT mean, VT sigma, std::mt19937& rng, VT min, VT max) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with log-normal distribution (" << mean << ", " << sigma << ") clipped to [" << min << "," << max << "]" << std::endl;
    #endif
        std::lognormal_distribution<VT> dis (mean, sigma);
        auto new_variable = std::vector< std::vector<VT> >(post_rank.size(), std::vector<VT>());
        for (auto post = 0; post < post_rank.size(); post++) {
            new_variable[post] = std::vector<VT>(pre_rank[post].size(), 0.0);
            for (auto it = new_variable[post].begin(); it != new_variable[post].end(); it++) {
                VT tmp = dis(rng);
                if ((tmp >= min) && (tmp <= max)) {
                    *it = tmp;
                }else{
                    if (tmp < min)
                        *it = min;
                    else
                        *it = max;
                }
            }
        }
        return new_variable;
    }

    /**
     *  @details    Updates a single *existing* entry within the matrix.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @param[in]  value       new matrix value
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector< std::vector<VT> > &variable, const IT lil_idx, const IT col_idx, const VT value) {
    #ifdef _DEBUG_ACCESSOR
        std::cout << "LILMatrix::update_matrix_variable(row_idx = " << post_rank[lil_idx] << ", col_idx = " << col_idx << ", value = " << value << ")" << std::endl;
        bool updated = false;
    #endif
        for (auto idx = 0; idx < pre_rank[lil_idx].size(); idx++) {
            if (pre_rank[lil_idx][idx] == col_idx) {
                variable[lil_idx][idx] = value;
            #ifdef _DEBUG_ACCESSOR
                updated = true;
            #endif
            }
        }
    #ifdef _DEBUG_ACCESSOR
        if (!updated)
            std::cerr << "Update failed ..." << std::endl;
    #endif
    }

    /**
     *  @details    Updates all *existing* entries of a matrix row.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @param[in]  values      new values for the row indicated by lil_idx.
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector< std::vector<VT> > &variable,
                             const IT lil_idx,
                             const std::vector<VT> values)
    {
        assert( (lil_idx < variable.size()) );
        assert( (values.size() == variable[lil_idx].size()) );

        std::copy(values.begin(), values.end(), variable[lil_idx].begin());
    }

    /**
     *  @details    Updates all *existing* entries of a matrix.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  values      new values for the row indicated by lil_idx stored as a list of list according to LILMatrix::pre_rank
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector< std::vector<VT> > &variable,
                             const std::vector< std::vector<VT> > &data)
    {
        assert( (data.size() == post_rank.size()) );

        for (auto i = 0; i < post_rank.size(); i++) {
            update_matrix_variable_row(variable, i, data[i]);
        }
    }

    /**
     *  @brief      retrieve a LIL representation for a given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @returns    a LIL representation from the given variable.
     */
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector< std::vector<VT> > &variable) {
        return variable;
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
        assert ( (lil_idx < variable.size()) );

        return variable[lil_idx];
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

        for (auto idx = 0; idx < pre_rank[lil_idx].size(); idx++) {
            if (pre_rank[lil_idx][idx] == col_idx) {
                return variable[lil_idx][idx];
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
        return std::vector<VT>(post_rank.size(), default_value);
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

    /**
     *  @brief      Get a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline std::vector<VT> get_vector_variable_all(std::vector<VT> variable) {
        return variable;
    }

    template <typename VT>
    inline void update_vector_variable(std::vector<VT> &variable, IT lil_idx, VT value) {
        assert ( (lil_idx < post_rank.size()) );

        variable[lil_idx] = value;
    }

    /**
     *  @brief      Get a single item from a vector variable
     *  @details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline VT get_vector_variable(std::vector<VT> variable, IT lil_idx) {
        assert( (lil_idx < post_rank.size()) );

        return variable[lil_idx];
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
        size += sizeof(std::vector<IT>);            // container
        size += post_rank.capacity() * sizeof(IT);  // data

        // pre ranks
        size += sizeof(std::vector<std::vector<IT>>);                   // top-level container
        size += pre_rank.capacity() * sizeof(std::vector<IT>);          // inner container
        for( auto it = pre_rank.cbegin(); it != pre_rank.cend(); it++ )
            size += it->capacity() * sizeof(IT);                        // data of inner container

        return size;
    }

    LILMatrix<IT>* slice_across_rows(IT beg, IT end) {
        assert( (beg >= 0) );
        assert( (end >= 0) );

        auto sliced_matrix = new LILMatrix<IT>(num_rows_, num_columns_);

        sliced_matrix->post_rank = std::vector<IT>(post_rank.begin()+beg, post_rank.begin()+end);
        sliced_matrix->pre_rank = std::vector<std::vector<IT>>(pre_rank.begin()+beg, pre_rank.begin()+end);

        return sliced_matrix;
    }

    LILMatrix<IT>* transpose() {
    #ifdef _DEBUG
        std::cout << "LILMatrix::transpose() - origin matrix ( " << num_rows_ << ", " << num_columns_ << " )" << std::endl;
    #endif
        auto transposed_matrix = new LILMatrix<IT>(num_columns_, num_rows_);
        auto tmp_row_idx = std::vector< std::vector<IT> >(num_columns_, std::vector<IT>());

        // we need to use the forward view, the column indices in bwd_view_ are relative to the CSR
        for (auto r = 0; r < this->post_rank.size(); r++) {
            for (auto c = 0; c < pre_rank[r].size(); c++) {
                tmp_row_idx[pre_rank[r][c]].push_back(post_rank[r]);
            }
        }

        for (auto c =0; c < num_columns_; c++) {
            if  (tmp_row_idx[c].empty())
                continue;

            // post ranks are automatically sorted
            transposed_matrix->post_rank.push_back(c);

            // pre ranks could be mixed
            std::sort(tmp_row_idx[c].begin(),tmp_row_idx[c].end());
            transposed_matrix->pre_rank.push_back(tmp_row_idx[c]);
        }

        if (this->nb_synapses() != transposed_matrix->nb_synapses())
            std::cerr << "Something went wrong during transpose: " << this->nb_synapses() << "!=" << transposed_matrix->nb_synapses() << std::endl;

	#ifdef _DEBUG
        std::cout << "Original post_rank.size() = " << this->post_rank.size() << std::endl;
        std::cout << "Transposed post_rank.size() = " << transposed_matrix->post_rank.size() << std::endl;
	#endif
        return transposed_matrix;
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

    /**
     *  @brief      print the matrix representation to the standard out (i. e. command-line)
     */
    void print_data_representation() {
        std::cout << "LILMatrix instance at " << this << std::endl;
        print_matrix_statistics();

        std::cout << "  post_ranks = [ ";
        for (auto it = post_rank.begin(); it != post_rank.end(); it++) {
            std::cout << *it << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  pre_ranks = [ ";
        for (auto it = pre_rank.begin(); it != pre_rank.end(); it++) {
            std::cout << "[ ";
            for( auto it2 = it->begin(); it2 != it->end(); it2++)
                std::cout << *it2 << " ";
            std::cout << "], ";
        }
        std::cout << "]" << std::endl;
    }

    template<typename VT>
    void print_variable(const std::vector<std::vector<VT>> &variable) {
        std::cout << "LILMatrix variable instance: " << this << std::endl;
        // for each diagonal depict: offset/bool mask
        std::cout << "[ ";
        for (IT i = 0; i < variable.size(); i++) {
            std::cout << "[";
            for (auto col_it = variable[i].begin(); col_it != variable[i].end(); col_it++) {
                std::cout << *col_it << ",";
            }
            std::cout << "] ";
        }
        std::cout << "]" << std::endl;
    }
};
