/*
 *    SELLMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *	  Copyright (C) 2021-22  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                  2021-22  Qi Tang <kevin2014tq@gmail.com>
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
 *   @brief     sliced ELLPACK sparse matrix representation according to Alexander Monakov et al. (2010) 
 *              and Moritz Kreutzer et al. (2013) with some minor modifications as described below.

 *   @details   more details can be found in the paper:
 *
 *              Automatically tuning sparse matrix-vector multiplication for GPU architectures by Alexander Monakov et al. (2010)
 *              A unified sparse matrix data format for modern processors with wide SIMD units by Moritz Kreutzer et al. (2013)
 *  
 *              The sliced ELLPACK format encodes the nonzeros of a sparse matrix in dense matrices where one represent the column
 *              indices, one for values and  another one  for row_ptr. The sliced ELLPACK divides the sparse matrix into equal-sized 
 *              blocks, and each block is stored in ELLPACK format.
 * 
 *              Let's consider the following example matrix
 *
 *                       | 0 1 0 |
 *                  A =  | 2 0 3 |
 *                       | 0 0 0 |
 *                       | 0 0 4 |
 * 
 *              First we need to determine the block size of the matrix , in our example block_size = 2. This means that every two 
 *              rows form a block. In addition, if the number of rows is not evenly divisible by the block size, then we need to 
 *              add empty rows.
 * 
 *              Next, we need to determine the maximum number of column-entries per block (we call this *maxnzr*), in our case maxnzr = 2 
 *              for the first block and maxnzr = 1 for the second block.
 * 
 *                           | 1 0 |
 *               col_idx_ =  | 0 2 |
 *                           | 0 |		
 *                           | 2 |
 * 
 *              row_ptr points to the position of the first element of each block. The last element in row_ptr stores the total number
 *              of sliced ELLPACK elements. 
 *
 * 	            row_ptr = [0, 4, 6]
 *              
 *  @tparam     IT          intended to be used for all values related to row/column
 *  @tparam     ST          intended to be used for running indices across the matrix (e.g. if the value would overflow for IT)
 *  @tparam     row_major   the individual slices are stored as *dense* blocks. They can either stored using row-major (=true) or
 *                          column-major (=false) scheme.
 */
template<typename IT = unsigned int, typename ST = unsigned long int, bool row_major = true>
class SELLMatrix {
  public:

    /**
     *  @brief      Constructor
     *  @details    Does not initialize any data.
     *  @param[in]  num_rows        number of rows of the original matrix 
     *  @param[in]  num_columns     number of columns of the original matrix 
     *  @param[in]  blocksize       size of each block (number of rows in each block)
     */
    explicit SELLMatrix(const IT num_rows, const IT num_columns, const IT blocksize):
        num_rows_(num_rows), num_columns_(num_columns), block_size_(blocksize) {
        num_blocks_ = 0;
        post_ranks_ = std::vector<IT>();
        col_idx_ = std::vector<IT>();
        row_ptr_ = std::vector<ST>();
        mask_ = std::vector<char>();
        num_non_zeros_ = 0;
    }

    /**
     *  @brief      Destructor.
     */
    ~SELLMatrix() {
    #ifdef _DEBUG
        std::cout << "SELLMatrix::~SELLMatrix(this=" << this << ")" << std::endl;
    #endif
    }

    /**
     *  @brief      clear the contained data
     */
    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "SELLMatrix::clear(this=" << this << ")" << std::endl;
    #endif
        num_blocks_ = 0;

        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        col_idx_.clear();
        col_idx_.shrink_to_fit();

        row_ptr_.clear();
        row_ptr_.shrink_to_fit();

        mask_.clear();
        mask_.shrink_to_fit();

        num_non_zeros_ = 0;
    }

/************************************************************************************************************/
/*  Accessors to member variables                                                                            */
/************************************************************************************************************/

    /**
     *  @brief      returns number of rows of the dense matrix.
     */
    IT num_rows() {
        return num_rows_;
    }

    /**
     *  @brief      returns number of columns of the dense matrix.
     */
    IT num_columns() {
        return num_columns_;
    }

    /**
     *  @return     a list of those row indices which belongs to a non-empty row.
     */
    std::vector<IT> get_post_rank() { return post_ranks_; }

    /**
     *  @details    get column indices
     *  @return     a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
     */
    std::vector<std::vector<IT>> get_pre_ranks() {
        std::vector<std::vector<IT>> lil_pre_ranks;

        if (row_major) {
            for (int i = 0; i < post_ranks_.size(); i++) {
                lil_pre_ranks.push_back(std::move(get_dendrite_pre_rank(i)));
            }            
        }else {
            std::cerr << "SELLMatrix::get_pre_ranks() is not implemented for column major" << std::endl;
        }
        return lil_pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    typename std::vector<IT> get_dendrite_pre_rank(IT lil_idx) {
        assert((lil_idx < post_ranks_.size()));

        IT row_idx = post_ranks_[lil_idx];
        IT block_idx = row_idx / block_size_;
        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT local_row = row_idx % block_size_;

        if (row_major) {            
            auto beg = col_idx_.begin() + row_ptr_[block_idx] + local_row * block_length;
            auto end = std::find(beg + 1, beg + block_length, 0);
            return std::vector<IT>(beg, end);
        } else {
            std::vector<IT> tmp;
            auto offset = row_ptr_[block_idx] + local_row;

            tmp.push_back(col_idx_[offset]); // push directly into the 0-th element

            for (int c = 1; c < block_length; c++) {
                auto col = col_idx_[c * block_size_ + offset];
                if (col == 0)break;
                tmp.push_back(col);
            }
            return tmp;
        }        
    }

    /**
     *  @brief      number of synapses in the complete matrix
     *  @returns    number of synapses for all rows
     */
    ST nb_synapses() {
        return this->num_non_zeros_;
    }

    /**
    *  @details    returns the number of stored rows. (i. e. each of these rows contains at least one connection).
    */
    IT nb_dendrites() {
        return static_cast<IT>(post_ranks_.size());
    }

    /*
    *  @details    return the number of non-zero in this matrix for a given row.
    */
    IT dendrite_size(IT lil_idx) {
        IT row_idx = post_ranks_[lil_idx];
        IT block_idx = row_idx / block_size_;

        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT local_row = row_idx % block_size_;

        if (row_major) {            
            auto beg = col_idx_.begin() + row_ptr_[block_idx] + local_row * block_length;
            auto end = std::find(beg + 1, beg + block_length, 0);
            return static_cast<IT>(std::distance(beg, end));
        }
        else {
            IT nnz_given_row = 0;
            auto offset = row_ptr_[block_idx] + local_row;

            for (int c = 0; c < block_length; c++) {
                if (mask_[c * block_size_ + offset])nnz_given_row++;                
            }

            return nnz_given_row;
        }    
    }

    /**
    *  @brief      get column indices
    *  @returns    the column indices as std::vector<IT>
    */
    std::vector<IT> column_indices() {
        return col_idx_;
    }

    /**
    *  @brief      get number of blocks
    *  @returns    number of blocks 
    */
    IT get_num_blocks() {
        return num_blocks_;
    }

    /**
    *  @brief      get the size of block
    *  @returns    the size of block
    */
    IT get_block_size() {
        return block_size_;
    }

    /**
    *  @brief      get row ptr
    *  @returns    the row ptr as std::vector<ST>
    */
    std::vector<ST> row_ptr() {
        return row_ptr_;
    }

    /**
    *  @brief      get mask
    *  @returns    the mask as std::vector<short int>
    */
    std::vector<char> get_mask() {
        return mask_;
    }

/************************************************************************************************************/
/*  Initialize the sparse matrix representation                                                             */
/************************************************************************************************************/

    /**
     *  @brief      initialize connectivity based on a provided LIL representation.        
     */
    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector<std::vector<IT>> column_indices) {
    #ifdef _DEBUG
        std::cout << "SELLMatrix::init_matrix_from_lil()" << std::endl;
    #endif

        post_ranks_ = row_indices;
        auto lil_row_idx = 0;        

        //compute number of blocks
        unsigned int num_blocks = num_rows_ / block_size_;
        if (num_rows_ % block_size_)num_blocks++;
        num_blocks_ = num_blocks;
        std::vector<unsigned int> blocklength(num_blocks, 0);

        std::vector<IT> row_length_(num_rows_, 0);
        //get row length
        for (int i = 0; i < num_rows_; i++) {
            if (i == row_indices[lil_row_idx]) {
                row_length_[i] = column_indices[lil_row_idx].size();

                // next row in LIL
                lil_row_idx++;
            }            
        }

        // for debug
        ST min_block_length = num_columns_;
        ST max_block_length = 0;

        //compute blocklength in each block
        for (int i = 0; i < num_blocks; i++) {
            unsigned int rowbegin = i * block_size_;
            blocklength[i] = row_length_[rowbegin];
            for (int j = 1; j < block_size_; j++) {
                int row_now = rowbegin + j;
                if ((row_now) >= num_rows_)break;
                if (blocklength[i] < row_length_[row_now]) blocklength[i] = row_length_[row_now];
            }

            // debug
            if ( blocklength[i] < min_block_length)
                min_block_length = blocklength[i];
            if ( blocklength[i] > max_block_length)
                max_block_length = blocklength[i];
        }
        
        lil_row_idx = 0;
        unsigned int sell_row_idx = 0;

        // start to convert LIL to SELL
        for (int i = 0; i < num_blocks; i++) {
            std::vector<IT> temp_block_col(block_size_ * blocklength[i], 0);
            std::vector<char> temp_block_mask(block_size_ * blocklength[i], false);
            row_ptr_.push_back(col_idx_.size());

            //in each block
            if (row_major) {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)
                        break;

                    if (sell_row_idx == row_indices[lil_row_idx]) {
                        std::copy(column_indices[lil_row_idx].begin(), column_indices[lil_row_idx].end(), temp_block_col.begin() + (j * blocklength[i]));
                        num_non_zeros_ += column_indices[lil_row_idx].size();
                        // next row in LIL
                        lil_row_idx++;
                    }
                    //encode mask
                    for (int k = 0; k < row_length_[sell_row_idx]; k++) {
                        temp_block_mask[j * blocklength[i] + k] = true;
                    }
                }                
            }
            else {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)
                        break;

                    if (sell_row_idx == row_indices[lil_row_idx]) {
                        for (int c = 0; c < column_indices[lil_row_idx].size(); c++) {
                            temp_block_col[c * block_size_ + j] = column_indices[lil_row_idx][c];
                            temp_block_mask[c * block_size_ + j] = true; //encode mask
                        }
                        num_non_zeros_ += column_indices[lil_row_idx].size();
                        // next row in LIL
                        lil_row_idx++;
                    }                    
                }
            }
            col_idx_.insert(col_idx_.end(), temp_block_col.begin(), temp_block_col.end());
            mask_.insert(mask_.end(), temp_block_mask.begin(), temp_block_mask.end());
            temp_block_col.clear();
            temp_block_mask.clear();
        }

        row_ptr_.push_back(col_idx_.size());

        // remove unneccessary allocated space
        row_ptr_.shrink_to_fit();
        col_idx_.shrink_to_fit();
        mask_.shrink_to_fit();

        // sanity check (did we allocate enough dense blocks?)
        if (lil_row_idx != row_indices.size()) {
            std::cerr << "SELLMatrix::init_matrix_from_lil() something went wrong ..." << std::endl;
            return false;
        }

    #ifdef _DEBUG
        print_matrix_statistics();
        std::cout << "  created " << num_blocks_ << " blocks with " << block_size_ << " rows each" << std::endl;
        std::cout << "    min. size: " << min_block_length << std::endl;
        std::cout << "    max. size: " << max_block_length << std::endl;
    #endif
        return true;
    }

/************************************************************************************************************/
/*  Initialize Matrix Variables                                                                             */
/************************************************************************************************************/

    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   the default value for all nonzeros in the matrix.
     *  @returns    A STL object filled with the default values according to LILMatrix::pre_rank
     */
    template <typename VT>
    std::vector<VT> init_matrix_variable(VT default_value) {
        auto variable = std::vector<VT>(mask_.size());
        for (ST idx = 0; idx < mask_.size(); idx++) {
            variable[idx] = (mask_[idx]) ? default_value : static_cast<VT>(0.0);
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
    std::vector<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif
        auto variable = std::vector<VT>(mask_.size());
        std::uniform_real_distribution<VT> dis (a,b);

        for (ST idx = 0; idx < mask_.size(); idx++) {
            variable[idx] = (mask_[idx]) ? dis(rng) : static_cast<VT>(0.0);
        }

        return variable;
    }

/************************************************************************************************************/
/*  Update Values of Matrix Variables                                                                       */
/************************************************************************************************************/

    /**
     *  @details    Updates a single *existing* entry within the matrix.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT>& variable, const IT lil_idx, const IT column_idx, const VT value) {
        IT row_idx = post_ranks_[lil_idx];

        if (row_major) {
            IT block_idx = row_idx / block_size_;
            //first should compute block length in this block
            IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
            IT offset_in_block = row_idx % block_size_;
            auto beg = col_idx_.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
            auto end = std::find(beg + 1, beg + block_length, 0);

            for (auto j = beg; j < end; j++) {
                if (*j == column_idx) {
                    variable[std::distance(col_idx_.begin(), j)] = value;
                    break;
                }
            }
        }
        else {
            std::cerr << "SELLMatrix::update_matrix_variable() is not implemented for column major" << std::endl;
        }
        
    }

    /**
     *  @details    Updates all *existing* entries of a matrix row.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT>& variable, const IT lil_idx, const std::vector<VT> data) {
        IT row_idx = post_ranks_[lil_idx];
        IT block_idx = row_idx / block_size_;
        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT local_row = row_idx % block_size_;

        if (row_major) {            
            std::copy(data.begin(), data.end(), variable.begin() + row_ptr_[block_idx] + local_row * block_length);
        }
        else {
            auto offset = row_ptr_[block_idx] + local_row;

            for (int c = 0; c < data.size(); c++) {
                variable[c * block_size_ + offset] = data[c];
            }
        }        
    }

    /**
     *  @details    Updates all *existing* entries of a matrix.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT>& variable, const std::vector< std::vector<VT> >& data)
    {
    #ifdef _DEBUG
        std::cout << "SELLMatrix::update_matrix_variable_all()" << std::endl;
    #endif
        if (data.size() != post_ranks_.size())
            std::cerr << "Update variable failed: mismatch of data field sizes." << std::endl;
        if (row_major) {
            for (auto i = 0; i < post_ranks_.size(); i++) {
                update_matrix_variable_row(variable, i, data[i]);
            }
        }
        else
        {
            IT data_row_idx = 0;
            ST data_offset = 0;
            for (int i = 0; i < num_blocks_; i++) {
                IT block_length = (row_ptr_[i + 1] - row_ptr_[i]) / block_size_;
                
                std::vector<VT> temp_block_value(block_size_ * block_length, 0.0);
                for (int j = 0; j < block_size_; j++) {
                    IT sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)break;
                    if (sell_row_idx == post_ranks_[data_row_idx]) {
                        for (int c = 0; c < data[data_row_idx].size(); c++) {
                            temp_block_value[c * block_size_ + j] = data[data_row_idx][c];
                        }
                        data_row_idx++;
                    }                    
                }

                std::copy(temp_block_value.begin(), temp_block_value.end(), variable.begin() + data_offset);
                data_offset += temp_block_value.size();
                temp_block_value.clear();                
            }
        }
        
    }

/************************************************************************************************************/
/*  Read-out Values of Matrix Variables                                                                     */
/************************************************************************************************************/

    /**
     *  @brief      retrieve a LIL representation for a given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline std::vector< std::vector <VT> > get_matrix_variable_all(const std::vector<VT>& variable) {
        auto values = std::vector< std::vector <VT> >();
        for (unsigned int lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            values.push_back(std::move(get_matrix_variable_row(variable, lil_idx)));
        }        
        return values;
    }

    /**
     *  @brief      retrieve a specific row from the given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector<VT>& variable, const IT lil_idx) {
        IT row_idx = post_ranks_[lil_idx];
        IT block_idx = row_idx / block_size_;
        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT local_row = row_idx % block_size_;
        if (row_major) {
            auto beg = variable.begin() + row_ptr_[block_idx] + local_row * block_length;
            auto end = std::find(beg + 1, beg + block_length, 0);
            return std::vector<VT>(beg, end);
        }
        else {
            auto row_size = dendrite_size(lil_idx);
            auto tmp = std::vector<VT>(row_size);
            auto offset = row_ptr_[block_idx] + local_row;

            for (IT c = 0; c < row_size; c++) {
                tmp[c] = variable[c * block_size_ + offset];
            }
            return tmp;
        }
        
    }

    /**
     *  @brief      retruns a single value from the given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT lil_idx, const IT column_idx) {
        IT row_idx = post_ranks_[lil_idx];
        IT block_idx = row_idx / block_size_;
        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT offset_in_block = row_idx % block_size_;
        auto beg = col_idx_.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
        auto end = std::find(beg + 1, beg + block_length, 0);

        for (auto j = beg; j < end; j++) {
            if (*j == column_idx)
                return variable[std::distance(col_idx_.begin(), j)];
        }            
        return 0; // should not happen ...
    }

/************************************************************************************************************/
/*  Initialization and Update of vector variables                                                           */
/************************************************************************************************************/

    /**
     *  \brief      Initialize a vector variable
     *  \details    Variables marked as 'semiglobal' stored in a vector of the size of LILMatrix::post_rank
     *  \tparam     VT              data type of the variable.
     *  \param[in]  default_value   value to initialize all elements in the vector
     *  \returns    the initialized vector containing DenseMatrix::num_rows_ elements.
     */
    template <typename VT>
    inline std::vector<VT> init_vector_variable(VT default_value) {
        return std::vector<VT>(post_ranks_.size(), default_value);
    }

/************************************************************************************************************/
/*  Other helpful functions                                                                                 */
/************************************************************************************************************/

    /**
     *  @brief      computes the size in bytes
     *  @details    contains also the required size of LILMatrix partition but not account allocated variables.
     *  @returns    size in bytes for stored connectivity
     *  @see        LILMatrix::size_in_bytes()
     */
    virtual size_t size_in_bytes() {

        size_t size = 4 * sizeof(IT);
        size += sizeof(ST);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += col_idx_.capacity() * sizeof(IT);

        size += sizeof(std::vector<ST>);
        size += row_ptr_.capacity() * sizeof(ST);

        size += sizeof(std::vector<char>);
        size += mask_.capacity() * sizeof(char); 

        return size;
    }

  protected:
    /**
     *  @brief      print some matrix characteristics to the standard out (i. e. command-line)
     *  @details    Intended for debug.
     */
    void print_matrix_statistics() {
        std::cout << "  #rows: " << static_cast<unsigned long>(num_rows_) << std::endl;
        std::cout << "  #columns: " << static_cast<unsigned long>(num_columns_) << std::endl;
        std::cout << "  #nnz: " << num_non_zeros_ << std::endl;
        std::cout << "  #blocksize: " << block_size_ << std::endl;
        std::cout << "  #num of blocks: " << num_blocks_ << std::endl;
        std::cout << "  #stored as " << ((row_major) ? "row_major" : "column_major") << std::endl;
    }

    /**
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. 
     */
    void print_data_representation() {
        std::cout << "SELLMatrix instance at " << this << std::endl;
        print_matrix_statistics();

        std::cout << "  post_ranks = [ " << std::endl;
        for (IT r = 0; r < post_ranks_.size(); r++) {
            std::cout << static_cast<unsigned long>(post_ranks_[r]) << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  row_ptr_ = [ " << std::endl;
        for (IT i = 0; i < row_ptr_.size(); i++) {
            std::cout << row_ptr_[i] << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  col_idx_ = [ " << std::endl;
        for (ST i = 0; i < col_idx_.size(); i++) {
            std::cout << static_cast<unsigned long>(col_idx_[i]) << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  mask_ = [ " << std::endl;
        for (ST i = 0; i < mask_.size(); i++) {
            if (mask_[i])std::cout << "T ";
            else std::cout << "F ";
        }
        std::cout << "]" << std::endl;
    }

    /**
     *  @brief      print col_idx_ of the matrix representation to console.
     */
    void print_data() {

        std::cout << "col indices " << std::endl;
        unsigned int rowbegin = 0;
        //i-th block
        for (int i = 0; i < num_blocks_; i++) {
            int len = (row_ptr_[i + 1] - row_ptr_[i]) / block_size_;
            //j-th row in i-th block
            for (int j = 0; j < block_size_; j++) {
                rowbegin = j + i * block_size_;
                std::cout << "(" << rowbegin << ") [";
                int location = row_ptr_[i] + j * len;
                //k-th element in j-th row 
                for (int k = 0; k < len; k++) {
                    std::cout << col_idx_[location + k] << " ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

  protected:
    IT num_rows_;                       ///< maximum number of rows 
    IT num_columns_;                    ///< maximum number of columns 
    IT block_size_;                     ///< size of each block (maximum number of rows in each block)
    IT num_blocks_;                     ///< number of blocks in the dense matrix
    ST num_non_zeros_;                  ///< number of nonzeros

    std::vector<IT> post_ranks_;        ///< which rows does contain entries
    std::vector<IT> col_idx_;           ///< column indices for accessing dense vector
    std::vector<ST> row_ptr_;           ///< points to first element in each block
    std::vector<char> mask_;            ///< mask of entries
};