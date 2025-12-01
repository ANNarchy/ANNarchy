/*
 *    BSRMatrixBitmask.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

/**
 *	\brief		Implementation of a blocked compressed sparse row (BSR) format.
 *	\details	A blocked variant of the classic compressed sparse row matrix format. It is basically
 *              the idea of a compressed sparse row, but instead of single values we encode a dense block.
 *              The format has been described in detail, e.g. in:
 * 
 *              * Eberhardt & Hoemmen (2016): Optimization of block sparse matrix-vector multiplication on shared-memory parallel architectures
 *
 *	\tparam 	IT		    index data type
 *	\tparam 	VT		    value data type
 *  \tparam     MT          As a zero can represent a non-existing entry or a existing entry, we need an additional array which encodes if a position is
 *                          a non-zero entry in the matrix. In this implementation each value of the mask corresponds to one position. The size of each
 *                          entry is determined by MT (we recommend char as it consumes only 1 byte).
 *  \tparam     row_major   row_major   determines the matrix storage for the dense sub matrices. If
 *                          set to true, the matrix will be stored as row major, otherwise
 *                          in column major.
 */
template<typename IT = unsigned int, typename ST = unsigned long int, typename MT = char, bool row_major=true>
class BSRMatrixBitmask {
 protected:
    const IT  num_rows_;
    const IT  num_columns_;
    const IT  tile_size_;
    const IT  bitmask_size_;

    // we take the variant of Benetia et al. (2018) to encode pointer and length in a CSR-like array.
    // Further block column index is not directly stored with the block as in Vershoor and Jalba (2012).
    std::vector<IT> block_row_pointer_;
    std::vector<IT> block_column_index_;
    std::vector<MT> tile_bitmask_;
    ST num_tile_elems_;                       ///< number of elements stored across all tiles. Note that must not be the number of nonzeros as padding zeros are counted too.

    // not typical for BSR, but helpful for the usage in ANNarchy
    std::vector<IT> post_ranks_;

    // Attention: this function returns the LIL indices, this easier for the following processing
    std::vector<std::vector<IT>> split_row_indices(std::vector<IT>& row_indices, IT nb_block_rows) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::split_row_indices()" << std::endl;
    #endif
        assert( (row_indices.size() <= this->num_rows_) );

        auto chunks = std::vector<std::vector<IT>>(nb_block_rows, std::vector<IT>());
        auto it = row_indices.begin();
        IT lil_idx = 0;
        for (; it != row_indices.end(); it++, lil_idx++) {
            IT chunk_idx = IT(double(*it)/double(tile_size_));
            chunks[chunk_idx].push_back(lil_idx);
        }

        return chunks;
    }

 public:
    /**
     *  \brief      Constructor of a BSRMatrixBitmask
     *  \details    which will throw an exception of the number of rows/columns is not divisable by tile_size
     */
    explicit BSRMatrixBitmask(const unsigned int num_rows, const unsigned int num_columns, const unsigned int tile_size):
        num_rows_(num_rows), num_columns_(num_columns), tile_size_(tile_size), bitmask_size_(sizeof(MT)*8) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::BSRMatrixBitmask(num_rows=" << num_rows << ", num_columns=" << num_columns << ", tile_size=" << tile_size << ", bitmask_=" << bitmask_size_ << ")" << std::endl;
    #endif
            // HD: this limitation is a consequence of the current implementation.
            assert(tile_size_ <= bitmask_size_);
    }

    ~BSRMatrixBitmask() {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::~BSRMatrixBitmask()" << std::endl;
    #endif
    }

    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::clear()" << std::endl;
    #endif
        post_ranks_.clear();
        post_ranks_.shrink_to_fit();

        block_row_pointer_.clear();
        block_row_pointer_.shrink_to_fit();

        block_column_index_.clear();
        block_column_index_.shrink_to_fit();

        block_row_pointer_.clear();
        block_row_pointer_.shrink_to_fit();

        tile_bitmask_.clear();
        tile_bitmask_.shrink_to_fit();
    }

    inline IT num_rows() {
        return this->num_rows_;
    }

    inline IT num_columns() {
        return this->num_columns_;
    }

    //
    //  Accessors for the computation
    //

    /**
     *  \brief      get access to block_ptr.
     *  \details    the i-th entry in row_ptr indicates the entries of this block. Is used to access the
     *              BCSRMatrix::row_begin_ and array.
     */
    inline IT* block_row_pointer() {
        return block_row_pointer_.data();
    }

    inline IT* block_column_index() {
        return block_column_index_.data();
    }

    inline IT block_row_size() {
        return block_row_pointer_.size() - 1;
    }

    inline IT get_tile_size() {
        return tile_size_;
    }

    //
    //  Initialization methods
    //

    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector<std::vector<IT>> column_indices) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::init_matrix_from_lil()" << std::endl;
    #endif

        // Construct the BSR format from LIL
        post_ranks_ = row_indices;

        // sanity checks
        assert( (row_indices.size() == column_indices.size()) );

        // data vector = vec[row_block][col_block]
        IT nb_block_rows = IT(ceil(double(this->num_rows_) / double(this->tile_size_)));
        IT nb_blocks_per_row = IT(ceil(double(this->num_columns_) / double(this->tile_size_)));

    #ifdef _DEBUG
        std::cout << "Theoretical max. dimension: " << nb_block_rows << " x " << nb_blocks_per_row << " with tile dimension: " << tile_size_ << " x " << tile_size_ << std::endl;
    #endif

        if ((size_t(nb_block_rows) * size_t(nb_blocks_per_row)) >= size_t(std::numeric_limits<ST>::max()))
            std::cout << "Warning theoretical number of blocks could exceed size_type of BSR matrix." << std::endl;

        this->block_row_pointer_ = std::vector<IT>(nb_block_rows+1, 0);
        this->block_column_index_ = std::vector<IT>();
        this->num_tile_elems_ = 0;

        // The variables are stored as a dense block if one tile has more than one nonzero
        auto current_tile_bitmask = std::vector<MT>(tile_size_, static_cast<MT>(0.0));

        // We need to transform the matrix in chunks of rows, otherwise the temporary lists get too large
        auto row_indices_chunked = split_row_indices(row_indices, nb_block_rows);

        ST total_blocks=0;
        IT r_cast, c_cast;
        for (IT b_r_idx = 0; b_r_idx < nb_block_rows; b_r_idx++) {
            // block row offset
            this->block_row_pointer_[b_r_idx] = this->block_column_index_.size();

            // scan the current chunk of rows for nonzeros and note there indices
        #ifdef _DEBUG_CONN
            std::cout << "  block row " << b_r_idx << " considers " << row_indices_chunked[b_r_idx].size() << " rows" << std::endl;
        #endif
            auto idx_pairs_per_block = std::vector<std::vector<std::pair<IT, IT>>>(nb_blocks_per_row, std::vector<std::pair<IT, IT>>());

            for (auto lil_it = row_indices_chunked[b_r_idx].begin(); lil_it != row_indices_chunked[b_r_idx].end(); lil_it++) {
                r_cast = row_indices[*lil_it];

                for (auto col_it = column_indices[*lil_it].begin(); col_it != column_indices[*lil_it].end(); col_it++) {
                    c_cast = *col_it;

                    IT b_c_idx = c_cast / tile_size_;
                    idx_pairs_per_block[b_c_idx].push_back(std::pair<IT, IT>(r_cast, c_cast));
                }
            }

            // We check once for all possible blocks instead each block individually
            check_free_memory(nb_blocks_per_row * tile_size_ * tile_size_);

            // Store the dense tiles
            IT total_blocks_in_row = 0;
            for (IT b_c_idx = 0; b_c_idx < nb_blocks_per_row; b_c_idx++ ) {
                if (idx_pairs_per_block[b_c_idx].size()>0) {
                #ifdef _DEBUG_CONN
                    std::string matrix_order = (row_major) ? "row-major" : "column-major";
                    std::cout << "  create dense block index " << b_c_idx << " with " << idx_pairs_per_block[b_c_idx].size() << " nonzeros using " << matrix_order << " storage." << std::endl;
                #endif

                    // fill the complete tile with zeros
                    std::fill(current_tile_bitmask.begin(), current_tile_bitmask.end(), static_cast<MT>(0.0));

                    // set the positions of nonzeros
                    for (auto it = idx_pairs_per_block[b_c_idx].begin(); it != idx_pairs_per_block[b_c_idx].end(); it++) {
                        IT tile_r_idx = it->first % tile_size_;
                        IT tile_c_idx = it->second % tile_size_;

                        if (row_major) {
                            current_tile_bitmask[tile_r_idx] |= (1<<tile_c_idx);
                        }else{
                            current_tile_bitmask[tile_c_idx] |= (1<<tile_r_idx);
                        }
                    }

                #ifdef _DEBUG_CONN
                    std::cout << "  encoded bitmask:" << std::endl;
                    if (row_major) {
                        for (IT row = 0; row < tile_size_; row++) {
                            std::cout << "    ";
                            for (IT col = 0; col < tile_size_; col++) {
                                std::cout << ((current_tile_bitmask[row] & (1<<col)) ? 1 : 0) << " ";
                            }
                            std::cout << std::endl;
                        }
                    } else {
                        for (IT row = 0; row < tile_size_; row++) {
                            std::cout << "    ";
                            for (IT col = 0; col < tile_size_; col++) {
                                std::cout << ((current_tile_bitmask[col] & (1<<row)) ? 1 : 0) << " ";
                            }
                            std::cout << std::endl;
                        }
                    }
                #endif

                    this->block_column_index_.push_back(b_c_idx);
                    this->tile_bitmask_.insert(this->tile_bitmask_.end(), current_tile_bitmask.begin(), current_tile_bitmask.end());
                    this->num_tile_elems_ += tile_size_ * tile_size_;

                    total_blocks_in_row++;
                }
            }

            // do we create an overflow?
            assert( (size_t(total_blocks+total_blocks_in_row) < size_t(std::numeric_limits<ST>::max())) );

            total_blocks += total_blocks_in_row;
        }

    #ifdef _DEBUG
        std::cout << "  Created " << total_blocks << " of " << nb_block_rows * nb_blocks_per_row << " possible." << std::endl;
        std::cout << "  i.e.," << total_blocks << " times " << tile_size_ << "x" << tile_size_ << "-> " << total_blocks * tile_size_ * tile_size_ << " elements allocated." << std::endl;
    #endif

        // last row
        this->block_row_pointer_[nb_block_rows] = this->block_column_index_.size();

        // sanity check (did we allocate enough dense blocks?)
        assert( this->num_tile_elems_ == (total_blocks * tile_size_ * tile_size_) );

        // remove unneccessary allocated space
        this->block_column_index_.shrink_to_fit();
        this->tile_bitmask_.shrink_to_fit();

        return true;
    }

    //
    //  Accessors for the Python ANNarchy interface
    //

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
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            pre_ranks.push_back(std::move(get_dendrite_pre_rank(lil_idx)));
        }
        return pre_ranks;
    }

    /**
     *  @details    get column indices of a specific row.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    a list of column indices of a specific row.
     */
    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        auto pre_ranks = std::vector<IT>();

        // sanity check
        assert( lil_idx < post_ranks_.size() );

        // decode which block_row we need
        IT b_r_idx = post_ranks_[lil_idx] / tile_size_;

        // decode the column indices for the corresponding block row
        for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
            IT row_in_tile = post_ranks_[lil_idx] % tile_size_; // selected row mapped to tile
            ST tile_offset = b_c_idx * tile_size_ * tile_size_; // where this tile begins

            // scan the row if it contains any nonzeros
            if (row_major) {

                // begin/end of the row in the dense array
                ST row_in_tile_begin = tile_offset + row_in_tile * tile_size_;
                ST row_in_tile_end = row_in_tile_begin + tile_size_;

                // Note the col_idx is running index over all tiles
                for (ST col_idx = row_in_tile_begin; col_idx < row_in_tile_end; col_idx++ ) {
                    // index relative to the single row
                    IT col_in_tile = col_idx%tile_size_;

                    // check if the position is a non-zero
                    if (tile_bitmask_[b_c_idx * tile_size_ + row_in_tile] & (1<<col_in_tile))
                        pre_ranks.push_back(block_column_index_[b_c_idx]*tile_size_+col_in_tile);
                }
            } else {
                auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];
                for (IT col = 0; col < tile_size_; col++) {
                    if ((current_tile_bitmask[col] & (1<<row_in_tile)) ? 1 : 0) {
                        pre_ranks.push_back(block_column_index_[b_c_idx]*tile_size_+col);
                    }
                }
            }
        }

    #ifdef _DEBUG_CONN
        std::cout << "decoded the following pre_ranks for post-neuron " << post_ranks_[lil_idx] << ": ";
        for (auto it = pre_ranks.begin(); it != pre_ranks.end(); it++)
            std::cout << *it << " ";
        std::cout << std::endl;
    #endif
        return pre_ranks;
    }

    /**
     *  @details    returns the stored connections in this matrix
     *  @returns    number of synapses across all rows
     */
    inline ST nb_synapses() {
        ST count = 0;

        for (auto it = tile_bitmask_.begin(); it != tile_bitmask_.end(); it++) {
            for (auto idx = 0; idx < this->bitmask_size_; idx++)
                if (*it & (1<<idx))
                    count++;
        }
        
        return count;
    }

    /**
     *  @details    returns the number of stored rows. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @returns    the number of stored rows (i. e. each of these rows contains at least one connection).
     */
    inline IT nb_dendrites() {
        return post_ranks_.size();
    }

    /**
     *  @details    returns the stored connections in this matrix for a given row. The return type is an unsigned int as the maximum of small data types used for IT could be exceeded.
     *  @param[in]  lil_idx     index of the selected row. To get the correct index use the post_rank array, e. g. lil_idx = post_ranks.find(row_idx).
     *  @returns    number of synapses across all rows of a given row.
     */
    IT dendrite_size(int lil_idx) {
        // sanity check
        assert( lil_idx < post_ranks_.size() );

        IT b_r_idx = post_ranks_[lil_idx] / tile_size_;
        IT count = 0;
        for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
            IT row_in_tile = post_ranks_[lil_idx] % tile_size_;
            auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];

            for (IT col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++) {
                if (row_major) {
                    if ((current_tile_bitmask[row_in_tile] & (1<<col_in_tile)) ? 1 : 0)
                        count++;
                } else {
                    if ((current_tile_bitmask[col_in_tile] & (1<<row_in_tile)) ? 1 : 0)
                        count++;
                }
            }
        }

        return count;
    }

    //
    //  Initialization and Update of variables.
    //
    /**
     *  @details    Initialize a num_rows_ by num_columns_ matrix based on the stored connectivity.
     *  @tparam     VT              data type of the variable.
     *  @param[in]  default_value   the default value for all nonzeros in the matrix.
     *  @returns    initialized STL container
     */
    template <typename VT>
    std::vector< VT > init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        if (!check_free_memory(this->num_tile_elems_ * sizeof(VT))) {
            std::cerr << "BSRMatrixBitmask::init_matrix_variable() allocation failed." << std::endl; 
            return std::vector<VT>();
        }

        auto variable = std::vector<VT>(this->num_tile_elems_, static_cast<VT>(0.0));

        for (IT b_r_idx = 0; b_r_idx < this->block_row_pointer_.size(); b_r_idx++) {
            for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
                for (IT row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++) {
                    if (row_major) {
                        ST row_in_tile_begin = b_c_idx * tile_size_ * tile_size_ + row_in_tile * tile_size_;
                        for (ST col_in_tile = row_in_tile_begin; col_in_tile < row_in_tile_begin + tile_size_; col_in_tile++ ) {
                            if (tile_bitmask_[b_c_idx * tile_size_ + row_in_tile] & (1<<(col_in_tile%tile_size_)))
                                variable[col_in_tile] = default_value;
                        }
                    } else {
                        ST col_in_tile_begin = b_c_idx * tile_size_ * tile_size_;
                        auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];
                        for (IT col = 0; col < tile_size_; col++) {
                            if ((current_tile_bitmask[col] & (1<<row_in_tile)) ? 1 : 0) {
                                variable[col_in_tile_begin + col * tile_size_ + row_in_tile] = default_value;
                            }
                        }
                    }
                }
            }
        }

        return variable;
    }

    template <typename VT>
    std::vector< VT > init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {

        std::uniform_real_distribution<VT> dis (a,b);
        auto variable = std::vector<VT>(this->num_tile_elems_, static_cast<VT>(0.0));

        for (IT b_r_idx = 0; b_r_idx < this->block_row_pointer_.size(); b_r_idx++) {
            for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
                for (IT row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++) {
                    if (row_major) {
                        ST row_in_tile_begin = b_c_idx * tile_size_ * tile_size_ + row_in_tile * tile_size_;
                        for (ST col_in_tile = row_in_tile_begin; col_in_tile < row_in_tile_begin + tile_size_; col_in_tile++ ) {
                            if (tile_bitmask_[b_c_idx * tile_size_ + row_in_tile] & (1<<(col_in_tile%tile_size_)))
                                variable[col_in_tile] = dis(rng);
                        }
                    } else {
                        ST col_in_tile_begin = b_c_idx * tile_size_ * tile_size_;
                        auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];
                        for (IT col = 0; col < tile_size_; col++) {
                            if ((current_tile_bitmask[col] & (1<<row_in_tile)) ? 1 : 0) {
                                variable[col_in_tile_begin + col * tile_size_ + row_in_tile] = dis(rng);
                            }
                        }
                    }
                }
            }
        }

        return variable;
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector<VT> &variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::update_matrix_variable_all()" << std::endl;
    #endif
        // Sanity checks
        assert( (post_ranks_.size() == data.size()) );
        assert( (variable.size() > 0));
        assert( (this->num_tile_elems_ == variable.size()));

        // update matrix row by row
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++ ) {
            update_matrix_variable_row(variable, lil_idx, data[lil_idx]);
        }
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::update_matrix_variable_row(lil_idx = " << lil_idx << ")" << std::endl;
    #endif
        IT b_r_idx = post_ranks_[lil_idx] / tile_size_;
        IT val_idx = 0;

        for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
            IT row_in_tile = post_ranks_[lil_idx] % tile_size_;

            if (row_major) {
                ST row_in_tile_begin = b_c_idx * tile_size_ * tile_size_ + row_in_tile * tile_size_;
                for (ST col_in_tile = row_in_tile_begin; col_in_tile < row_in_tile_begin + tile_size_; col_in_tile++ ) {
                    if (tile_bitmask_[b_c_idx * tile_size_ + row_in_tile] & (1<<(col_in_tile%tile_size_)))
                        variable[col_in_tile] = data[val_idx++];
                }
            } else {
                ST col_in_tile_begin = b_c_idx * tile_size_ * tile_size_;
                auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];
                for (IT col = 0; col < tile_size_; col++) {
                    if ((current_tile_bitmask[col] & (1<<row_in_tile)) ? 1 : 0) {
                        variable[col_in_tile_begin + col * tile_size_ + row_in_tile] = data[val_idx++];
                    }
                }
            }
        }
    }

    template <typename VT>
    inline void update_matrix_variable(std::vector<VT> &variable, const IT lil_idx, const IT column_idx, const VT value) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::update_matrix_variable(lil_idx = " << lil_idx << ", column_idx = " << column_idx << ")" << std::endl;
    #endif
        IT row_idx = post_ranks_[lil_idx];
        IT b_r_idx = row_idx / tile_size_;

        for (IT blk_col_idx = block_row_pointer_[b_r_idx]; blk_col_idx < block_row_pointer_[b_r_idx+1]; blk_col_idx++) {
            IT bcol_idx = block_column_index_[blk_col_idx];     // which column in row

            if ((column_idx >= bcol_idx * tile_size_) & (column_idx < (bcol_idx+1) * tile_size_)) {
                IT row_tile_offset = row_idx % tile_size_;
                IT col_tile_offset = column_idx % tile_size_;

                if (row_major) {
                    ST idx = blk_col_idx * tile_size_ * tile_size_ + row_tile_offset * tile_size_ + col_tile_offset;
                    if (tile_bitmask_[blk_col_idx * tile_size_ + row_tile_offset] & (1<<idx%tile_size_))
                        variable[idx] = value;

                    return; // early stop
                } else {
                    ST idx = blk_col_idx * tile_size_ * tile_size_ + column_idx%tile_size_ * tile_size_ + row_idx % tile_size_;
                    if (tile_bitmask_[blk_col_idx * tile_size_ + column_idx%tile_size_] & (1<<row_idx % tile_size_))
                        variable[idx] = value;

                    return; // early stop
                }
            }
        }

        // no tile was hit. should not happen ...
        std::cerr << "BSRMatrixBitmask::update_matrix_variable(): failed to update value ..." << std::endl;
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
        for (IT lil_idx = 0; lil_idx < post_ranks_.size(); lil_idx++) {
            lil_variable.push_back(std::move(get_matrix_variable_row<VT>(variable, lil_idx)));
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
        IT b_r_idx = post_ranks_[lil_idx] / tile_size_;

        auto values = std::vector<VT>();
        for (IT b_c_idx = block_row_pointer_[b_r_idx]; b_c_idx < block_row_pointer_[b_r_idx+1]; b_c_idx++) {
            IT row_in_tile = post_ranks_[lil_idx] % tile_size_;

            if (row_major) {
                ST row_in_tile_begin = b_c_idx * tile_size_ * tile_size_ + row_in_tile * tile_size_;
                for (ST col_in_tile = row_in_tile_begin; col_in_tile < row_in_tile_begin + tile_size_; col_in_tile++ ) {
                    if (tile_bitmask_[b_c_idx * tile_size_ + row_in_tile] & (1<<(col_in_tile%tile_size_)))
                        values.push_back(variable[col_in_tile]);
                }
            } else {
                ST col_in_tile_begin = b_c_idx * tile_size_ * tile_size_;
                auto current_tile_bitmask = &tile_bitmask_[b_c_idx * tile_size_];
                for (IT col = 0; col < tile_size_; col++) {
                    if ((current_tile_bitmask[col] & (1<<row_in_tile)) ? 1 : 0) {
                        values.push_back(variable[col_in_tile_begin + col * tile_size_ + row_in_tile]);
                    }
                }
            }
        }
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
    inline VT get_matrix_variable(const std::vector<VT>& variable, const IT &lil_idx, const IT &col_idx) {
    #ifdef _DEBUG
        std::cout << "BSRMatrixBitmask::get_matrix_variable(lil_idx = " << lil_idx << ", column_idx = " << col_idx << ")" << std::endl;
    #endif
        IT row_idx = post_ranks_[lil_idx];
        IT b_r_idx = row_idx / tile_size_;

        for (IT blk_col_idx = block_row_pointer_[b_r_idx]; blk_col_idx < block_row_pointer_[b_r_idx+1]; blk_col_idx++) {
            IT bcol_idx = block_column_index_[blk_col_idx];     // which column in row

            if ((col_idx >= bcol_idx * tile_size_) & (col_idx < (bcol_idx+1) * tile_size_)) {
                IT row_tile_offset = row_idx % tile_size_;
                IT col_tile_offset = col_idx % tile_size_;

                if (row_major) {
                    return variable[blk_col_idx * tile_size_ * tile_size_ + row_tile_offset * tile_size_ + col_tile_offset];
                } else {
                    return variable[blk_col_idx * tile_size_ * tile_size_  + col_tile_offset * tile_size_ + row_tile_offset];
                }
            }
        }

        return static_cast<VT>(0.0); // should not happen
    }

    //
    //  Initialization and Update of vector variables.
    //

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

    //
    //  Other helpful functions
    //

    virtual size_t size_in_bytes() {
        size_t size = 0;

        size += 4*sizeof(IT) + sizeof(ST);               // constants

        // STL container
        size += 3*sizeof(std::vector<IT>);  // block_row_pointer_, block_column_index_ and post_ranks_
        size += sizeof(std::vector<MT>);    // tile_bitmask_

        // Data
        size += block_row_pointer_.capacity() * sizeof(IT);
        size += block_column_index_.capacity() * sizeof(IT);
        size += tile_bitmask_.capacity() * sizeof(MT);
        size += post_ranks_.capacity() * sizeof(IT);

        return size;
    }

    void print_data_representation(bool print_memory_footprint=true) {
        std::cout << "BSR tile size:        " << this->tile_size_ << std::endl;
        std::cout << "Number of block rows: " << this->block_row_pointer_.size()-1 << std::endl;
        std::cout << "block column indices = [ ";
        for (IT block_row_idx = 0; block_row_idx < this->block_row_pointer_.size()-1; block_row_idx++ ) {
            std::cout << "[ ";
            for (IT blk_col_idx = block_row_pointer_[block_row_idx]; blk_col_idx < block_row_pointer_[block_row_idx+1]; blk_col_idx++) {
                std::cout << block_column_index_[blk_col_idx] << " ";
            }
            std::cout << "] ";
        }
        std::cout << "]" << std::endl;

        if (print_memory_footprint)
            std::cout << "Requires " << (this->size_in_bytes() / 1024.0 / 1024) << "MB (~" << this->size_in_bytes() / this->nb_synapses() << " bytes per non-zero)" << std::endl;
    }
};
