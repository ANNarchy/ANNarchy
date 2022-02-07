/*
 *
 *    SELLRMatrix.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021-22  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                           Qi Tang <kevin2014tq@gmail.com>
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
#include "LILMatrix.hpp"

/*
 *   @brief     sliced ELLPACK sparse matrix representation according to Alexander Monakov et al. (2010) 
 *              and Moritz Kreutzer et al. (2013) with some minor modifications as described below.
 *
 *   @details   more details can be found in the paper:
 *
 *              - Automatically tuning sparse matrix-vector multiplication for GPU architectures by Alexander Monakov et al. (2010)
 *              - A unified sparse matrix data format for modern processors with wide SIMD units by Moritz Kreutzer et al. (2013)
 *  
 *              The sliced ELLPACK format encodes the nonzeros of a sparse matrix in dense matrices where one represent the column
 *              indices, one for values and  another one  for row_ptr. The sliced ELLPACK divides the sparse matrix into equal-sized 
 *              blocks, and each block is stored in ELLPACK format.
 * 
 *              Let's consider the following example matrix
 * 
 *                      | 0 1 0 |
 *               A =    | 2 0 3 |
 *                      | 0 0 0 |
 *                      | 0 0 4 |
 * 
 *              First we need to determine the block size of the matrix , in our example block_size = 2. This means that every two 
 *              Fows form a block. Then we need to determine the maximum number of column-entries per block (we call this *maxnzr*),
 *              in our case maxnzr = 2 for the first block and maxnzr = 1 for the second block.
 * 
 *                           | 1 0 |
 *               col_idx_ =  | 0 2 |
 *                           | 0 |		
 *                           | 2 |
 * 
 *              row_ptr points to the position of the first element of each block. The last element in row_ptr stores the total number
 *              of sliced ELLPACK elements. 
 * 
 * 	             row_ptr = [0, 4, 6]
 * 
 *              row_length array stores the number of elements in each row (it has the same function as in ELLPACK-R)
 *           	 
 *               row_length = [1, 2, 0, 1]
 * 
 *
 *  \tparam     IT      intended to be used for all values related to row/column
 *  \tparam     ST      intended to be used for running indices across the matrix (e.g. if the value would overflow for IT)
 */
template<typename IT = unsigned int, typename ST = unsigned long int, bool row_major = true>
class SELLRMatrix {
  protected:
    IT num_rows_;                       ///< maximum number of rows 
    IT num_columns_;                    ///< maximum number of columns 
    IT block_size_;                      ///< size of each block (maximum number of rows in each block)
    IT num_blocks_;                     ///< number of blocks in the dense matrix
    ST num_non_zeros_;                  ///< number of nonzeros

    std::vector<IT> post_ranks_;        ///< which rows does contain entries
    std::vector<IT> row_length_;        ///< number of nonzeros in each row
    std::vector<IT> col_idx_;           ///< column indices for accessing dense vector
    std::vector<ST> row_ptr_;           ///< points to first element in each block

  public:

    explicit SELLRMatrix(const IT num_rows, const IT num_columns, const IT blocksize):
        num_rows_(num_rows), num_columns_(num_columns), block_size_(blocksize) {
        num_blocks_ = 0;
        post_ranks_ = std::vector<IT>();
        row_length_ = std::vector<IT>(num_rows, 0);
        col_idx_ = std::vector<IT>();
        row_ptr_ = std::vector<ST>();
    }

    virtual ~SELLRMatrix() {
    #ifdef _DEBUG
        std::cout << "SELLRMatrix::~SELLRMatrix()" << std::endl;
    #endif
        clear();
    }
    
    void clear() {
    #ifdef _DEBUG
        std::cout << "SELLRMatrix::clear()" << std::endl;
    #endif
        std::fill(row_length_.begin(), row_length_.end(), 0);

        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        col_idx_.clear();
        col_idx_.shrink_to_fit();

        row_ptr_.clear();
        row_ptr_.shrink_to_fit();

        num_non_zeros_ = 0;
    }

    // Returns size in bytes for connectivity
    size_t size_in_bytes() {

        size_t size = 4 * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += post_ranks_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += row_length_.capacity() * sizeof(IT);

        size += sizeof(std::vector<IT>);
        size += col_idx_.capacity() * sizeof(IT);

        size += sizeof(std::vector<ST>);
        size += row_ptr_.capacity() * sizeof(ST);

        size += sizeof(ST);

        return size;
    }

    //
    // INITIALIZATION METHODS
    //
    /**
     *  @brief      initialize connectivity based on a provided LIL representation.        
     */
    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector<std::vector<IT>> column_indices) {

        post_ranks_ = row_indices;
        auto lil_row_idx = 0;        

        //compute number of blocks
        unsigned int num_blocks = num_rows_ / block_size_;
        if (num_rows_ % block_size_)num_blocks++;
        num_blocks_ = num_blocks;
        std::vector<unsigned int> blocklength(num_blocks, 0);

        //get row length
        for (int i = 0; i < num_rows_; i++) {
            if (i == row_indices[lil_row_idx]) {
                row_length_[i] = column_indices[lil_row_idx].size();

                // next row in LIL
                lil_row_idx++;
            }            
        }

        //compute blocklength in each block
        for (int i = 0; i < num_blocks; i++) {
            unsigned int rowbegin = i * block_size_;
            blocklength[i] = row_length_[rowbegin];
            for (int j = 1; j < block_size_; j++) {
                int row_now = rowbegin + j;
                if ((row_now) >= num_rows_)break;
                if (blocklength[i] < row_length_[row_now]) blocklength[i] = row_length_[row_now];
            }
        }

        
        lil_row_idx = 0;
        unsigned int sell_row_idx = 0;
        const IT nonvalue_idx = std::numeric_limits<IT>::max();

        // start to convert LIL to SELL
        for (int i = 0; i < num_blocks; i++) {
            std::vector<IT> temp_block_col(block_size_ * blocklength[i], nonvalue_idx);
            row_ptr_.push_back(col_idx_.size());

            //in each block
            if (row_major) {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)break;
                    if (sell_row_idx == row_indices[lil_row_idx]) {
                        std::copy(column_indices[lil_row_idx].begin(), column_indices[lil_row_idx].end(), temp_block_col.begin() + (j * blocklength[i]));
                        num_non_zeros_ += column_indices[lil_row_idx].size();
                        // next row in LIL
                        lil_row_idx++;
                    }
                }                
            } else {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)break;
                    if (sell_row_idx == row_indices[lil_row_idx]) {
                        for (int c = 0; c < column_indices[lil_row_idx].size(); c++) {
                            temp_block_col[c * block_size_ + j] = column_indices[lil_row_idx][c];
                        }
                        num_non_zeros_ += column_indices[lil_row_idx].size();
                        // next row in LIL
                        lil_row_idx++;
                    }                    
                }
            }
            col_idx_.insert(col_idx_.end(), temp_block_col.begin(), temp_block_col.end());
            temp_block_col.clear();
        }

        row_ptr_.push_back(col_idx_.size());

        // sanity check
        if (lil_row_idx != row_indices.size()) {
            std::cerr << "SELLRMatrix::init_matrix_from_lil() something went wrong ..." << std::endl;
            return false;
        }
        
        return true;
    }

    /**
    *  @brief      initialize connectivity using a fixed_number_pre pattern
    *  @details    for more details on this pattern see the ANNarchy Documentation.
    */
    bool fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, IT nnz_per_row, std::mt19937& rng) {
        // Generate post_to_pre LIL
        auto lil_mat = new LILMatrix<IT>(this->num_rows_, this->num_columns_);
        lil_mat->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng);

        // Generate SELL from this LIL
        auto success = init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

        // cleanup
        delete lil_mat;
        return success;
    }


    /**
     *  @brief      reads in a .csv file which contains the matrix stored as COO.
     *  @details    this function creates also the variable array, which is usually performed afterwards.
     *  @tparam     VT          value type of the nonzero
     *  @tparam     zero_based  set to true if the contained data in csv has as minimum possible index 0. If
     *                          set to false, the read-in indices will be decremented by 1.
     */
    template<typename VT, bool zero_based = true>
    std::vector<VT> init_from_csv(std::string filename, const char delimiter = ',') {
        auto tmp_col_idx = std::vector< std::vector < IT > >(num_rows_, std::vector<IT>());
        auto tmp_values = std::vector< std::vector < VT > >(num_rows_, std::vector<VT>());

        std::ifstream matrix_file(filename);
        if (!matrix_file.is_open()) {
            std::cerr << "Could not open the file: " << filename << std::endl;
        }
        else {
            std::string item;
            auto coo_triplet = std::vector<std::string>(3);

            std::string line = "";
            IT r_cast, c_cast;
            VT v_cast;

            //read each line and split the content using delimeter
            while (getline(matrix_file, line))
            {
                if (line.size() == 0) continue; //skip an empty line
                //std::cout << line.size() << std::endl;
                std::stringstream ss(line);
                for (int i = 0; i < 3; i++) {
                    std::getline(ss, item, delimiter);
                    coo_triplet[i] = std::move(item);
                }
                if (zero_based) {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()));
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()));
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                }
                else {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()) - 1);
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()) - 1);
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                }            

                tmp_col_idx[r_cast].push_back(c_cast);
                tmp_values[r_cast].push_back(v_cast);
            }
        }

        //compute number of blocks
        unsigned int num_blocks = num_rows_ / block_size_;
        if (num_rows_ % block_size_)num_blocks++;
        num_blocks_ = num_blocks;        

        //get row length
        for (int i = 0; i < num_rows_; i++) {
            row_length_[i] = tmp_col_idx[i].size();
            num_non_zeros_ += row_length_[i];
        }

        //compute blocklength in each block
        std::vector<unsigned int> blocklength(num_blocks, 0);
        for (int i = 0; i < num_blocks; i++) {
            int rowbegin = i * block_size_;
            blocklength[i] = row_length_[rowbegin];
            for (int j = 1; j < block_size_; j++) {
                int row_now = rowbegin + j;
                if ((row_now) >= num_rows_)break;
                if (blocklength[i] < row_length_[row_now]) blocklength[i] = row_length_[row_now];
            }
        }

        post_ranks_.clear();
        unsigned int sell_row_idx = 0;    //global row index in sell
        auto lil_values = std::vector<std::vector<VT>>();
        std::vector<VT> values_col_major;

        for (int i = 0; i < num_blocks; i++) {
            std::vector<IT> temp_block_col(block_size_ * blocklength[i], 0);
            std::vector<VT> temp_block_value(block_size_ * blocklength[i], 0.0);
            row_ptr_.push_back(col_idx_.size());

            if (row_major) {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)break;
                    std::copy(tmp_col_idx[sell_row_idx].begin(), tmp_col_idx[sell_row_idx].end(), temp_block_col.begin() + (j * blocklength[i]));

                    if (tmp_col_idx[sell_row_idx].size() > 0) {
                        post_ranks_.push_back(sell_row_idx);
                        lil_values.push_back(std::move(tmp_values[sell_row_idx]));
                    }
                }
            }
            else {
                for (int j = 0; j < block_size_; j++) {
                    sell_row_idx = j + i * block_size_;
                    if (sell_row_idx >= num_rows_)break;
                    for (int c = 0; c < tmp_col_idx[sell_row_idx].size(); c++) {
                        temp_block_col[c * block_size_ + j] = tmp_col_idx[sell_row_idx][c];
                        temp_block_value[c * block_size_ + j] = tmp_values[sell_row_idx][c];
                    }
                    
                    if (tmp_col_idx[sell_row_idx].size() > 0) {
                        post_ranks_.push_back(sell_row_idx);
                        lil_values.push_back(std::move(tmp_values[sell_row_idx]));
                    }
                }
            }
            
            col_idx_.insert(col_idx_.end(), temp_block_col.begin(), temp_block_col.end());
            values_col_major.insert(values_col_major.end(), temp_block_value.begin(), temp_block_value.end());
            temp_block_col.clear();
            temp_block_value.clear();
        }        

        row_ptr_.push_back(col_idx_.size());
        if (row_major) {
            auto values = init_matrix_variable<VT>(0.0);
            update_matrix_variable_all<VT>(values, lil_values);
            return values;
        }
        else {
            return values_col_major;
        }       
        
    }


    template<typename VT, bool zero_based = true>
    std::vector<VT> init_from_csv_to_lil(std::string filename, const char delimiter = ',') {
        auto tmp_col_idx = std::vector< std::vector < IT > >(num_rows_, std::vector<IT>());
        auto tmp_values = std::vector< std::vector < VT > >(num_rows_, std::vector<VT>());

        std::cout << "init_from_csv_to_lil" << std::endl;

        std::ifstream matrix_file(filename);
        if (!matrix_file.is_open()) {
            std::cerr << "Could not open the file: " << filename << std::endl;
        }
        else {
            std::string item;
            auto coo_triplet = std::vector<std::string>(3);

            std::string line = "";
            IT r_cast, c_cast;
            VT v_cast;

            //read each line and split the content using delimeter
            while (getline(matrix_file, line))
            {
                if (line.size() == 0) continue; //skip an empty line
                //std::cout << line.size() << std::endl;
                std::stringstream ss(line);
                for (int i = 0; i < 3; i++) {
                    std::getline(ss, item, delimiter);
                    coo_triplet[i] = std::move(item);
                }
                if (zero_based) {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()));
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()));
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                }
                else {
                    r_cast = static_cast<IT>(atoi(coo_triplet[0].data()) - 1);
                    c_cast = static_cast<IT>(atoi(coo_triplet[1].data()) - 1);
                    v_cast = static_cast<VT>(atof(coo_triplet[2].data()));
                }

                tmp_col_idx[r_cast].push_back(c_cast);
                tmp_values[r_cast].push_back(v_cast);
            }
        }
        // create a LIL from the read data
        auto lil_ranks = std::vector<IT>();
        auto lil_col_idx = std::vector<std::vector<IT>>();
        auto lil_values = std::vector<std::vector<VT>>();
        for (auto row = 0; row < num_rows_; row++) {

            row_length_[row] = tmp_col_idx[row].size();
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
     *  @brief      print the matrix representation to console.
     *  @details    All important fields are printed. 
     */
    virtual void print_data_representation() {
        std::cout << "SELLRMatrix instance at " << this << std::endl;
        std::cout << "  #rows: " << static_cast<unsigned long>(num_rows_) << std::endl;
        std::cout << "  #columns: " << static_cast<unsigned long>(num_columns_) << std::endl;
        std::cout << "  #nnz: " << num_non_zeros_ << std::endl;
        std::cout << "  #blocksize: " << block_size_ << std::endl;
        std::cout << "  #num of blocks: " << num_blocks_ << std::endl;
        std::cout << "  #stored as " << ((row_major) ? "row_major" : "column_major") << std::endl;
        std::cout << "  post_ranks = [ " << std::endl;
        for (IT r = 0; r < post_ranks_.size(); r++) {
            std::cout << static_cast<unsigned long>(post_ranks_[r]) << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  row_length_ = [ " << std::endl;
        for (IT i = 0; i < row_length_.size(); i++) {
            std::cout << row_length_[i] << " ";
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
    }

    /**
     *  @brief      print col_idx_ of the matrix representation to console.
     */
    void print_data() {
        std::cout << " #blocksize: " << block_size_ << std::endl;
        std::cout << " #num of blocks: " << num_blocks_ << std::endl;

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




    //
    // ACCESSOR METHODS
    //

    /*
     *  @return     a list of those row indices which belongs to a non-empty row.
     */
    std::vector<IT> get_post_rank() { return post_ranks_; }

    /**
    *  @details    get column indices
    *  @returns    a list-in-list of column indices for all rows comprising of at least one element sorted by rows.
       */
    std::vector<std::vector<IT>> get_pre_ranks() {
        std::vector<std::vector<IT>> lil_pre_ranks;

        if (row_major) {
            for (int i = 0; i < post_ranks_.size(); i++) {
                lil_pre_ranks.push_back(std::move(get_dendrite_pre_rank(i)));
            }            
        }else {
            std::cerr << "SELLRMatrix::get_pre_ranks() is not implemented for column major" << std::endl;
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

        if (row_major) {
            IT row_idx = post_ranks_[lil_idx];
            IT block_idx = row_idx / block_size_;
            //first should compute block length in this block
            IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
            IT offset_in_block = row_idx % block_size_;
            auto beg = col_idx_.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
            auto end = beg + row_length_[row_idx];
            return std::vector<IT>(beg, end);
        } else {
            std::cerr << "SELLRMatrix::get_dendrite_pre_rank() is not implemented for column major" << std::endl;
        }

        
    }
    

    IT dense_num_rows() {
        return num_rows_;
    }

    IT dense_num_columns() {
        return num_columns_;
    }

    /**
    *  @brief      number of elements in one row
    */
    IT nb_synapses(IT lil_idx) {
        int post_rank = post_ranks_[lil_idx];
        return row_length_[post_rank];
    }

    /**
     *  @brief      number of synapses in the complete matrix
     *  @returns    can easily exceed the number of columns and is therefore ST
     *              (which defaults in many implementations to ST)
     */
    ST nb_synapses() {
        return this->num_non_zeros_;
    }

    IT nb_dendrites() {
        return post_ranks_.size();
    }

    std::vector<IT> column_indices() {
        return col_idx_;
    }

    IT get_num_blocks() {
        return num_blocks_;
    }

    IT get_blocksize() {
        return block_size_;
    }

    ST get_num_nonzeros() {
        return num_non_zeros_;
    }

    std::vector<IT> row_length() {
        return row_length_;
    }

    std::vector<ST> row_ptr() {
        return row_ptr_;
    }

    
    IT dendrite_size(IT lil_idx) {
        IT post_rank = post_ranks_[lil_idx];

        // TODO: return the number of non-zero of row with index post_rank
        return 0;
    }

    //
    //  Initialize Variables
    //

    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     */
    template <typename VT>
    std::vector<VT> init_matrix_variable(VT default_value) {
        if (row_major) {
            ST variable_size = col_idx_.size();
            return std::vector<VT>(variable_size, default_value);
        }
        else {
            return std::vector<VT>();
        }
        
    }

    //
    //  Update Variables
    //

    /**
     *  @details    Updates a single *existing* entry within the matrix.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable(std::vector<VT>& variable, const IT row_idx, const IT column_idx, const VT value) {

        if (row_major) {
            IT block_idx = row_idx / block_size_;
            //first should compute block length in this block
            IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
            IT offset_in_block = row_idx % block_size_;
            auto beg = col_idx_.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
            auto end = beg + row_length_[row_idx];

            for (auto j = beg; j < end; j++) {
                if (*j == column_idx) {
                    variable[std::distance(col_idx_.begin(), j)] = value;
                    break;
                }
            }
        }
        else {
            std::cerr << "SELLRMatrix::update_matrix_variable() is not implemented for column major" << std::endl;
        }
        
    }

    /**
     *  @details    Updates all *existing* entries of a matrix row.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT>& variable, const IT lil_idx, const std::vector<VT> data) {
        if (row_major) {
            IT row_idx = post_ranks_[lil_idx];
            IT block_idx = row_idx / block_size_;
            //first should compute block length in this block
            IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
            IT offset_in_block = row_idx % block_size_;
            std::copy(data.begin(), data.end(), variable.begin() + row_ptr_[block_idx] + offset_in_block * block_length);
        }
        else {
            std::cerr << "SELLRMatrix::update_matrix_variable_row() is not implemented for column major" << std::endl;
        }
        
    }

    /**
     *  @details    Updates all *existing* entries of a matrix.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT>& variable, const std::vector< std::vector<VT> >& data)
    {
        if (data.size() != post_ranks_.size())
            std::cerr << "Update variable failed: mismatch of data field sizes." << std::endl;
        if (row_major) {
            for (auto i = 0; i < post_ranks_.size(); i++) {
                update_matrix_variable_row(variable, i, data[i]);
            }
        }
        else {
            IT data_row_idx = 0;
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
                variable.insert(variable.end(), temp_block_value.begin(), temp_block_value.end());
                temp_block_value.clear();
            }
        }
        
    }


    /**
     *  @brief      retruns a single value from the given variable.
     *  @details    this function is only called by the Python interface retrieve the current value of a *local* variable.
     *  @tparam     VT          data type of the variable.
     */
    template <typename VT>
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT row_idx, const IT column_idx) {
        IT block_idx = row_idx / block_size_;
        //first should compute block length in this block
        IT block_length = (row_ptr_[block_idx + 1] - row_ptr_[block_idx]) / block_size_;
        IT offset_in_block = row_idx % block_size_;
        auto beg = col_idx_.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
        auto end = beg + row_length_[row_idx];

        for (auto j = beg; j < end; j++) {
            if (*j == column_idx)
                return variable[std::distance(col_idx_.begin(), j)];
        }            
        return 0; // should not happen ...
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
        IT offset_in_block = row_idx % block_size_;

        auto beg = variable.begin() + row_ptr_[block_idx] + offset_in_block * block_length;
        auto end = beg + row_length_[row_idx];
        return std::vector<VT>(beg, end);
    }

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
};