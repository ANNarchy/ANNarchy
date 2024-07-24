/*
 *    ELLMatrix.hpp
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

#include "helper_functions.hpp"

/**
 *  @brief      ELLPACK sparse matrix representation according to Kincaid et al. (1989) with some
 *              minor modifications as described below.
 * 
 *              Format description by Kincaid et al. (1989):
 *
 *                  https://web.ma.utexas.edu/CNA/ITPACK/manuals/userv2d/node3.html
 * 
 *  @details    The ELLPACK format encodes the nonzeros of a sparse matrix in dense matrices
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
 *              left. The original authors recommend that missing elements in a row should be
 *              padded with zeros other authors suggest -1, or the maximum value of a type.
 *              We choose the latter variant, as we want to use a) favourly unsigned data types
 *              to represent indices and b) want to be flexible as much as possible.
 * 
 *                              | 1 0 |
 *                  col_idx_ =  | 0 2 |
 *                              | 2 0 |
 * 
 *              As for LILMatrix and others one need to highlight that rows with no nonzeros are
 *              compressed. This means, that we don't allocate empty rows instead we have a row rank
 *              array which encode which row in the stored matrix corresponds to the dense row index.
 *              For the above matrix this would be:
 * 
 *                  post_ranks_ = [0, 1, 3]
 *
 *  @tparam     IT          index type, i. e. a data type, which can represent num_rows_ respectively num_column_ elements 
 *  @tparam     ST          size type, i. e. a data type, which can represent maxnzr_ * num_rows_, at maximum num_rows_ * 
 *                          num_column_, elements (consequently this can differ from IT for small data types)
 *  @tparam     row_major   determines the matrix storage for the dense sub matrices. If
 *                          set to true, the matrix will be stored as row major, otherwise
 *                          in column major. 
 *                          Please note that the original format stores in row-major to ensure a
 *                          partial caching of data on CPUs. The column-major ordering is only
 *                          intended for the usage on GPUs.
 * 
 *  @todo       Maybe it would be a good idea, to distinguish the case of a) all rows are filled with at least
 *              one non-zero and b) the compressed case. So we could avoid to store the post_ranks_ if they are
 *              technically not needed.
 */
template<typename IT=unsigned int, typename ST=unsigned long int, bool row_major=true>
class ELLMatrix {
  protected:
    IT maxnzr_;                     ///< maximum row length of nonzeros
    const IT num_rows_;             ///< maximum number of rows
    const IT num_columns_;          ///< maximum number of columns
    const IT zero_marker_;          ///< we need to identify the end of the existant entries in the row

    std::vector<IT> post_ranks_;    ///< which rows does contain entries
    std::vector<IT> col_idx_;       ///< column indices for accessing dense vector

    /**
     *  @brief      check if the matrix fits into RAM
     *  @details    Unlike CUDA it appears that the standard C++ API does not
     *              provide a function to get the available RAM at a present time.
     *              Many sources recommended to use the /proc/meminfo file
     */
    bool check_free_memory(size_t required) {
    #ifdef __linux__
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
        std::cout << "ELLMatrix: allocate " << required << " from " << available << " bytes " << std::endl;
    #endif
        return required < available;

    #else
        return true;
    #endif
    }

  public:
    /**
     *  @brief      Constructor
     *  @details    Does not initialize any data.
     *  @param[in]  num_rows        number of rows of the original matrix (this value is only provided to have an unified interface)
     *  @param[in]  num_columns     number of columns of the original matrix (this value is only provided to have an unified interface)
     */
    explicit ELLMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns), zero_marker_(std::numeric_limits<IT>::max()) {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::default constructor"<< std::endl;
    #endif
        // Ensure that the data type can represent all possible indices
        assert( (num_rows < std::numeric_limits<IT>::max()) );
        // The max is in this format reserved for the non-existing nonzero
        assert( (num_columns < (std::numeric_limits<IT>::max()-1)) );
    }

    /**
     *  @brief      Copy constructor
     *  @details    Does initialize the data based on the provided instance *other*.
     *  @param[in]  other   the ELLMatrix instance to copy data from.
     */
    ELLMatrix(ELLMatrix<IT, ST, row_major>* other):
        num_rows_(other->num_rows_), num_columns_(other->num_columns_), zero_marker_(std::numeric_limits<IT>::max()) {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::copy constructor"<< std::endl;
    #endif
        // Ensure that the data type can represent all possible indices
        assert( (other->num_rows_ < std::numeric_limits<IT>::max()) );
        // The max is in this format reserved for the non-existing nonzero
        assert( (other->num_columns_ < (std::numeric_limits<IT>::max()-1)) );

        this->maxnzr_ = other->maxnzr_;
        this->post_ranks_ = other->post_ranks_;
        this->col_idx_ = other->col_idx_;
    }

    /**
     *  @brief      Destructor.
     */
    ~ELLMatrix() {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::~ELLMatrix()" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the contained data.
     *  @details    Either called from init_* methods or from the destructor this function clears
     *              the STL containers and resets ELLMatrix::maxnzr_
     */
    void clear() {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        col_idx_.clear();
        col_idx_.shrink_to_fit();

        maxnzr_ = 0;
    }

    /**
     *  @brief      returns number of rows of the dense matrix.
     *  @details    this value can differ but should be larger than the number of ELLMatrix::nb_dendrites()
     *  @returns    number of rows of the dense matrix.
     */
    IT dense_num_rows() {
        return num_rows_;
    }

    /**
     *  @brief      returns number of columns of the dense matrix.
     *  @details    this value can differ but should be larger than the number of ELLMatrix::dendrite_size(int lil_idx)
     *  @returns    number of columns of the dense matrix.
     */
    IT dense_num_columns() {
        return num_columns_;
    }

    /**
     *  @brief      Accessor to ELLMatrix::maxnzr_
     *  @details    this value is need to compute the index position.
     *  @returns    the maximum number of nonzeros in a row
     */
    inline IT get_maxnzr() {
        return maxnzr_;
    }

    inline const IT zero_marker() {
        return zero_marker_;
    }

    /**
     *  @brief      Accessor to ELLMatrix::col_idx_ 
     *  @returns    the raw pointer of ELLMatrix::col_idx_
     */
    inline const IT* get_column_indices() {
        return col_idx_.data();
    }

    /**
     *  @brief      get row indices
     *  @details    a list of indices for all rows comprising of at least one element
     *  @returns    the row indices as std::vector<IT>
     */
    std::vector<IT> get_post_rank() {
        return post_ranks_;
    }

    /**
     *  @details    get all stored column indices as LIL
     *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() { 
        auto pre_ranks = std::vector<std::vector<IT>>();

        if (row_major) {
            for(IT r = 0; r < post_ranks_.size(); r++) {
                auto beg = col_idx_.begin() + r*maxnzr_;
                auto end = std::find(beg, beg+maxnzr_, zero_marker_);

                pre_ranks.push_back(std::vector<IT>(beg, end));
            }
        } else {
            std::cerr << "ELLMatrix::get_pre_ranks() is not implemented for column major" << std::endl;
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

        if (row_major) {
            auto beg = col_idx_.begin() + lil_idx*maxnzr_;
            auto end = std::find(col_idx_.begin() + lil_idx*maxnzr_, col_idx_.begin() + (lil_idx+1)*maxnzr_, std::numeric_limits<IT>::max());

            return std::vector<IT>(beg, end);
        } else {
            auto tmp = std::vector < IT >();
            auto num_rows = post_ranks_.size();
            for(int c = 0; c < maxnzr_; c++) {
                if (col_idx_[c*num_rows+lil_idx] == std::numeric_limits<IT>::max()) // hit the end of line
                    break;

                tmp.push_back(col_idx_[c*num_rows+lil_idx]);
            }

            return tmp;
        }
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses across all rows
     */
    ST nb_synapses() {
        int size = 0;
        for (int r = 0; r < post_ranks_.size(); r++) {
            size += dendrite_size(r);
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

        if (row_major) {
            auto beg = col_idx_.begin() + lil_idx*maxnzr_;
            auto end = std::find(col_idx_.begin() + lil_idx*maxnzr_, col_idx_.begin() + (lil_idx+1)*maxnzr_, std::numeric_limits<IT>::max());

            return static_cast<IT>(std::distance(beg, end));
        } else {
            IT nnz_curr_row = 0;
            auto num_rows = post_ranks_.size();
            for(int c = 0; c < maxnzr_; c++) {
                if (col_idx_[c*num_rows+lil_idx] == std::numeric_limits<IT>::max()) // hit the end of line
                    break;

                nnz_curr_row++;
            }
            return nnz_curr_row;
        }
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
    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks) {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::init_matrix_from_lil()" << std::endl;
        std::cout << "received " << post_ranks.size() << " rows." << std::endl;
    #endif

        // Sanity check
        assert( (post_ranks.size() == pre_ranks.size()) );
        assert( (post_ranks.size() <= num_rows_) );

        //
        // 1st step:    iterate across the LIL to identify maximum
        //              row length
        post_ranks_ = post_ranks;
        maxnzr_ = std::numeric_limits<IT>::min();
        for(auto pre_it = pre_ranks.begin(); pre_it != pre_ranks.end(); pre_it++) {
            if ( maxnzr_ < static_cast<IT>(pre_it->size()) ) {
                maxnzr_ = pre_it->size();
            }
        }

    #ifdef _DEBUG
        std::cout << "Determined maxnzr = " << maxnzr_ << std::endl;
    #endif

        // Test if we produce an overflow for ST
        assert( (static_cast<unsigned long int>(post_ranks.size() * maxnzr_) < static_cast<unsigned long int>(std::numeric_limits<ST>::max())) );

        // Test if the matrix fits into memory
        if (!check_free_memory(maxnzr_ * post_ranks_.size() * sizeof(IT))) {
            clear();
            return false;
        }

        //
        // 2nd step:    iterate across the LIL to copy indices
        //
        // std::numeric_limits<IT>::max() will encode the non-existing elements
        // In contrast to other existing implementations, we do not represent empty rows.
        col_idx_ = std::vector<IT>(maxnzr_ * post_ranks_.size(), std::numeric_limits<IT>::max());
        if (row_major) {

            auto pre_it = pre_ranks.begin();
            IT row_idx = 0;
            for(; pre_it != pre_ranks.end(); pre_it++, row_idx++) {
                ST col_off = row_idx * maxnzr_;
                for (auto col_it = pre_it->begin(); col_it != pre_it->end(); col_it++) {
                    col_idx_[col_off++] = *col_it;
                }
            }

        } else {
            int num_rows = post_ranks_.size();

            for (int r = 0; r < num_rows; r++) {
                int c = 0;
                for (auto col_it = pre_ranks[r].begin(); col_it != pre_ranks[r].end(); col_it++, c++) {
                    col_idx_[c*num_rows+r] = *col_it;
                }
            }
        }
    #ifdef _DEBUG
        std::cout << "created ELLMatrix:" << std::endl;
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

        // create the values matrix
        auto value = init_matrix_variable<VT>(0.0);
        update_matrix_variable_all<VT>(value, lil_values);

        return value;
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
        std::cout << "ELLMatrix::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        check_free_memory(maxnzr_ * post_ranks_.size() * sizeof(VT));

        // fill all places with 0
        auto variable = std::vector<VT> (post_ranks_.size() * maxnzr_, 0.0);

        // only "set" nonzeros should be updated
        for (IT r = 0; r < post_ranks_.size(); r++) {
            for(IT c = 0; c < this->maxnzr_; c++) {
                if (this->col_idx_[c] != zero_marker_)
                    if (row_major)
                        variable[r*this->maxnzr_+c] = default_value;
                    else
                        variable[c*this->maxnzr_+r] = default_value;
            }
        }

        return variable;
    }

    /**
     *  @details    Updates all matrix values based on a LIL representation
     *  @tparam     VT              data type of the variable.
     *  @param[in]  variable        ELLPACK variable container
     *  @param[in]  data            LIL variable container
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "ELLMatrix::update_matrix_variable_all()" << std::endl;
    #endif
        assert( (post_ranks_.size() == data.size()) );

        if (row_major) {
            for(IT r = 0; r < post_ranks_.size(); r++) {
                auto beg = variable.begin() + r*maxnzr_;
                std::copy(data[r].begin(), data[r].end(), beg);
            }
        } else {
            int num_rows = post_ranks_.size();
            for(IT r = 0; r < num_rows; r++) {
                for(IT c = 0; c < data[r].size(); c++) {
                    variable[c*num_rows+r] = data[r][c];
                }
            }
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
            auto beg = variable.begin() + lil_idx*maxnzr_;
            std::copy(data.begin(), data.end(), beg);
        } else {
            int num_rows = post_ranks_.size();
            for(IT c = 0; c < data.size(); c++) {
                variable[c*num_rows+lil_idx] = data[c];
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
            for (ST idx = lil_idx * maxnzr_; idx < (lil_idx+1) * maxnzr_; idx++) {
                if (col_idx_[idx] == std::numeric_limits<IT>::max())
                    return;

                if (col_idx_[idx] == column_idx) {
                    variable[idx] = value;
                }
            }
        } else {
            std::cerr << "ELLMatrix::update_matrix_variable() is not implemented for column major" << std::endl;
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
                int row_size = dendrite_size(r);

                auto beg = variable.begin() + r*maxnzr_;
                auto end = variable.begin() + r*maxnzr_ + row_size;
                lil_variable.push_back(std::vector<VT>(beg, end));
            }
        } else {
            auto num_rows = post_ranks_.size();
            for(IT r = 0; r < num_rows; r++) {
                auto row_size = dendrite_size(r);
                auto tmp = std::vector<VT>(row_size);
                for (IT c = 0; c < row_size; c++) {
                    tmp[c] = variable[c*num_rows+r];
                }

                lil_variable.push_back(std::move(tmp));
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

        if (row_major) {
            int row_size = dendrite_size(lil_idx);
            auto beg = variable.begin() + lil_idx*maxnzr_;
            auto end = variable.begin() + lil_idx*maxnzr_ + row_size;

            return std::vector < VT >(beg, end);
        } else {
            auto num_rows = post_ranks_.size();
            auto row_size = dendrite_size(lil_idx);
            auto tmp = std::vector<VT>(row_size);
            for (IT c = 0; c < row_size; c++) {
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
        std::cerr << "ELLMatrix::get_matrix_variable() is not implemented" << std::endl;
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
    size_t size_in_bytes() {
        size_t size = 4 * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += col_idx_.capacity() * sizeof(IT);

        return size;
    }

    /**
     *  @brief      print the some information on the nonzeros to console.
     *  @details    The print-out will contain among others number rows, number columns, number nonzeros.
     *              Please note, that type casts are required to print-out the numbers encoded if IT or ST
     *              is e.g. unsigned char. 
     */
    void print_matrix_statistics() {
        ST sum = 0;
        IT num_rows_with_nonzeros = 0;

        for (int r = 0; r < post_ranks_.size(); r++) {
            if (row_major) {
                auto beg = col_idx_.begin() + r*maxnzr_;
                auto end = std::find(col_idx_.begin() + r*maxnzr_, col_idx_.begin() + (r+1)*maxnzr_, std::numeric_limits<IT>::max());

                auto dist = static_cast<IT>(std::distance(beg, end));
                if (dist > 0)
                {
                    sum += dist;
                    num_rows_with_nonzeros++;
                }
            } else {
                auto num_rows = post_ranks_.size();
                int nnz_curr_row = 0;
                for(int c = 0; c < maxnzr_; c++) {
                    if (col_idx_[c*num_rows+r] == std::numeric_limits<IT>::max()) // hit the end of line
                        break;
 
                    nnz_curr_row++;
                }
                if (nnz_curr_row > 0) {
                    sum += nnz_curr_row;
                    num_rows_with_nonzeros++;
                }
            }
        }

        double avg_nnz_per_row = static_cast<double>(sum) / static_cast<double>(num_rows_with_nonzeros);

        std::cout << "  #rows: " << static_cast<unsigned long>(num_rows_) << std::endl;
        std::cout << "  #columns: " << static_cast<unsigned long>(num_columns_) << std::endl;
        std::cout << "  #nnz: " << static_cast<unsigned long>(nb_synapses()) << std::endl;
        std::cout << "  empty rows: " << num_rows_ - num_rows_with_nonzeros << std::endl;
        std::cout << "  avg_nnz_per_row: " << avg_nnz_per_row << std::endl;
        std::cout << "  dense matrix = (" << static_cast<unsigned long>(nb_dendrites()) << ", " <<  static_cast<unsigned long>(maxnzr_) << ")" <<\
                     " stored as " << ((row_major) ? "row_major" : "column_major") << std::endl;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. Please note, that type casts are
     *              required to print-out the numbers encoded if IT or ST is e.g. unsigned char. 
     */
    virtual void print_data_representation() {
        std::cout << "ELLMatrix instance at " << this << std::endl;
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
