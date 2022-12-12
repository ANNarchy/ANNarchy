/*
 *    DenseMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                        Julien Vitay <julien.vitay@gmail.com>
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
 */
#pragma once

/*
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
 *                      The chosen data type should be able to represent the maximum values (LILMatrix::num_rows_ and ::num_columns_)
 * 
 *              ST      the second type should be used if the index type IT could overflow. For instance, the nb_synapses method should return ST as
 *                      the maximum value in case a full dense matrix would be IT times IT entries.
 *              MT      We need to store if a matrix value is set in a boolean mask. The size of each entry is determined by MT (we recommend char as its only 1 byte).
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major=true>
class DenseMatrix {
protected:
    const IT num_rows_;         ///< maximum number of rows which equals the maximum length of post_rank as well as maximum size of top-level of pre_rank.
    const IT num_columns_;      ///< maximum number of columns which equals the maximum available size in the sub-level vectors.
    std::vector<MT> mask_;      ///< encodes if an entry in the full matrix is a nonzero. Please note, in many C++ implementations bool will default to an integer. Therefore we use char here to ensure that we really use only 1 byte.

    /**
     *  @brief      check if the matrix fits into RAM
     *  @details    Unlike CUDA it appears that the standard C++ API does not
     *              provide a function to get the available RAM at a present time.
     *              Many sources recommended to use the /proc/meminfo file
     */
    bool check_free_memory(size_t required) {
        FILE *meminfo = fopen("/proc/meminfo", "r");

        // TODO:    I'm not completely sure, what we want to do
        //          in this case. Currently, we would hope for the best ...
        if(meminfo == nullptr) {
            std::cerr << "Could not read '/proc/meminfo'. ANNarchy can not catch to large allocations ..." << std::endl;
            return true;
        }

        char line[256];
        int ram;

        while(fgets(line, sizeof(line), meminfo))
        {
            if(sscanf(line, "MemFree: %d kB", &ram) == 1)
                break;  // hit
        }

        fclose(meminfo);
        size_t available = static_cast<size_t>(ram) * 1024;
    #ifdef _DEBUG
        std::cout << "DenseMatrix: allocate " << required << " from " << available << " bytes " << std::endl;
    #endif
        return required < available;
    }

    /*
     *  @brief      Decode the column indices for nonzeros in the matrix.
     */
    virtual std::vector<IT> decode_column_indices(IT row_idx) {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::decode_column_indices()" << std::endl;
    #endif
        auto indices = std::vector<IT>();
        ST idx;
        if (row_major) {
            for (IT c = 0; c < num_columns_; c++) {
                idx = row_idx * num_columns_ + c;
                if (mask_[idx])
                    indices.push_back(c);
            }
        } else {
            for (IT c = 0; c < num_columns_; c++) {
                idx = c * num_rows_ + row_idx;
                if (mask_[idx])
                    indices.push_back(c);
            }
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
     */
    explicit DenseMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::DenseMatrix(num_rows="<<num_rows<<", num_columns="<<num_columns<<")" << std::endl;
    #endif

        // we check if we can encode all possible values
        assert( (static_cast<long long>(num_rows_ * num_columns_) < static_cast<long long>(std::numeric_limits<ST>::max())) );
    }

    /**
     *  @brief      Destructor
     *  @details    calls the DenseMatrix::clear method. Is not declared as virtual as inheriting classes in our
     *              framework should never be destroyed by the base pointer.
     */
    ~DenseMatrix() {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::~DenseMatrix()" << std::endl;
    #endif
        clear();
    }

    /**
     *  @brief      Clear the dense matrix.
     *  @details    Clears the connectivity data stored in the *post_rank* and *pre_rank* STL containers and free
     *              the allocated memory. **Important**: allocated variables are not effected by this!
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::clear()" << std::endl;
    #endif
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
        auto post_ranks = std::vector<IT>(num_rows_, 0);
        for (IT r = 0; r < num_rows_; r++)
            post_ranks[r] = r;
        return post_ranks;
    }

    /**
     *  @details    get column indices
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() {
        auto pre_ranks = std::vector<std::vector<IT>>();
        for (IT row_idx = 0; row_idx < num_rows_; row_idx++) {
            pre_ranks.push_back(std::move(get_dendrite_pre_rank(row_idx)));
        }
        return pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  row_idx     index of the selected row.
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(IT row_idx) {
        return decode_column_indices(row_idx);
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses in the whole matrix.
     */
    ST nb_synapses() {
        ST size = 0;
        for (IT row_idx = 0; row_idx < num_rows_; row_idx++) {
            size += dendrite_size(row_idx);
        }
        return size;
    }

    /**
     *  @brief      Get the number of stored connections in this matrix for a given row.
     *  @details    The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    number of synapses across all rows of a given row.
     */
    IT dendrite_size(int row_idx) {
        IT size = 0;
        ST idx;
        if (row_major) {
            for (IT c = 0; c < num_columns_; c++) {
                idx = row_idx * num_columns_ + c;
                if (mask_[idx])
                    size++;
            }
        } else {
            for (IT c = 0; c < num_columns_; c++) {
                ST idx = c * num_rows_ + row_idx;
                if (mask_[idx])
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
        IT num_dendrites = 0;
        for(IT row_idx = 0; row_idx < num_rows_; row_idx++) {
            if (dendrite_size(row_idx) > 0)
                num_dendrites++;
        }
        return num_dendrites;
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
     */
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::init_matrix_from_lil()" << std::endl;
    #endif
        // Sanity checks
        assert ( (post_ranks.size() == pre_ranks.size()) );
        assert ( (static_cast<unsigned long int>(post_ranks.size()) <= static_cast<unsigned long int>(std::numeric_limits<IT>::max())) );

        // Sanity check: enough memory?
        if (!check_free_memory(num_columns_ * num_rows_ * sizeof(MT)))
            return false;

        // Allocate mask
        mask_ = std::vector<MT>(num_rows_ * num_columns_, static_cast<MT>(false));

        // Iterate over LIL and update mask entries to *true* if nonzeros are existing.
        for (IT lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            IT row_idx = post_ranks[lil_idx];
            for(auto inner_col_it = pre_ranks[lil_idx].cbegin(); inner_col_it != pre_ranks[lil_idx].cend(); inner_col_it++) {
                if (row_major)
                    mask_[row_idx * num_columns_ + *inner_col_it] = static_cast<MT>(true);
                else
                    mask_[(*inner_col_it) * num_rows_ + row_idx] = static_cast<MT>(true);
            }
        }

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
    std::vector<VT> init_matrix_from_csv(const std::string filename, const char delimiter=',') {
    #ifdef _DEBUG
        std::cout << "DenseMatrix::init_matrix_from_csv()" << std::endl;
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
     *  @brief      initialize connectivity using a fixed_probability pattern
     *  @details    For more details on this pattern see the ANNarchy Documentation.
     *  @param[in]  post_ranks  list of row indices of all rows which contain at least on elements to be accounted.
     *  @param[in]  pre_ranks   list of list, where the i-th sub-vector should contain a list of potential connection candidates for the i-th post-synaptic neuron.
     *  @param[in]  p           probability for a connection being set between two neurons.
     *  @param[in]  rng         a merseanne twister generator (need to be seeded in prior if necessary)
     */
    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "LILMatrix::fixed_probability_pattern()" << std::endl;
        std::cout << " rows: " << post_ranks.size() << std::endl;
        std::cout << " p: " << p << std::endl;
        std::cout << " self_connections: " << allow_self_connections << std::endl;
    #endif
        auto dis = std::uniform_real_distribution<double>(0.0, 1.0);

        // Allocate mask
        mask_ = std::vector<MT>(num_rows_ * num_columns_, static_cast<MT>(false));

        for(auto lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            int row_idx = post_ranks[lil_idx];

            // over all possible connections
            for (auto inner_col_it=pre_ranks.cbegin(); inner_col_it != pre_ranks.cend(); inner_col_it++) {
                if ( (!allow_self_connections) && (row_idx == *inner_col_it) )
                    continue;

                if (dis(rng) < p) {
                    if (row_major)
                        mask_[row_idx * num_columns_ + *inner_col_it] = static_cast<MT>(true);
                    else
                        mask_[*inner_col_it * num_rows_ + row_idx] = static_cast<MT>(true);
                }
            }
        }
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
        std::cout << "DenseMatrix::init_matrix_variable()" << std::endl;
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
        std::cout << "DenseMatrix::update_matrix_variable_all()" << std::endl;
    #endif
        // sanity check: target large enough?
        assert( (num_rows_ * num_columns_ == variable.size()) );
        assert( (num_rows_ == data.size()) );

        for (IT row_idx = 0; row_idx < data.size(); row_idx++) {
            update_matrix_variable_row(variable, row_idx, data[row_idx]);
        }
    }

    /**
     *  @details    Updates all *existing* entries of a matrix row.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  variable    Variable container initialized with LILMatrix::init_matrix_variable() and similiar functions.
     *  @param[in]  row_idx     index of the selected row.
     *  @param[in]  values      new values for the row indicated by row_idx.
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT row_idx, const std::vector<VT> values)
    {
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
     *  @param[in]  row_idx     index of the selected row.
     *  @param[in]  value       new matrix value
     *  @todo       Maybe one should check the mask if the nonzero existed before?
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT row_idx, const IT col_idx, const VT value) {
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
        auto values = std::vector< std::vector < VT > >(nb_dendrites(), std::vector < VT >());

        for (IT row_idx = 0; row_idx < nb_dendrites(); row_idx++) {
            auto col_idx = decode_column_indices(row_idx);

            // copy the data
            for (auto col_it = col_idx.cbegin(); col_it != col_idx.cend(); col_it++) {
                if (row_major) {
                    values[row_idx].push_back(variable[row_idx * num_columns_ + *col_it]);
                } else {
                    values[row_idx].push_back(variable[*col_it * num_rows_ + row_idx]);
                }
            }
        }

        return values;
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface to retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     *  @param[in]  row_idx     index of the selected row.
     *  @returns    a vector containing all elements of the provided variable and row_idx
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT>& variable, const IT &row_idx) {
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
     *  @param[in]  row_idx     index of the selected row.
     *  @param[in]  col_idx     index of the selected column.
     *  @returns    the value at position (lil_idx, col_idx)
     */
    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT &row_idx, const IT &col_idx) {
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
    inline void update_vector_variable(std::vector<VT> &variable, const IT lil_idx, const VT value) {
        assert( (num_rows_ != variable.size()) );
        assert( (lil_idx < num_rows_) );

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
    inline VT get_vector_variable(std::vector<VT> variable, const IT lil_idx) {
        assert( (lil_idx < num_rows_) );

        return variable[lil_idx];
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
    size_t size_in_bytes() {
        size_t size = 2 * sizeof(IT);               // scalar values

        size += mask_.capacity() * sizeof(MT);

        return size;
    }
};
