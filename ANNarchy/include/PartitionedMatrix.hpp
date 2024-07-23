/*
 *    PartitionedMatrix.hpp
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
 */
#pragma once

#include "helper_functions.hpp"

/**
 *  @brief      Wrapper class for handling multiple instances of SPARSE_MATRIX_TYPE.
 *  @details    In order to support the parallel evaluation of expecially spiking networks
 *              we divide the whole matrix into as many as threads parts.
 *  @tparam     SPARSE_MATRIX_TYPE  sparse matrix class, most likely a LILMatrix, LILInvMatrix or CSRCMatrix
 *  @tparam     IT                  index type which should be the same as used in the SPARSE_MATRIX_TYPE declaration
 *  @tparam     ST                  size type which should be the same as used in the SPARSE_MATRIX_TYPE declaration
 */
template<typename SPARSE_MATRIX_TYPE, typename IT = unsigned int, typename ST = unsigned long int>
class PartitionedMatrix {
protected:
    const IT num_rows_;     ///< number of rows in the original matrix
    const IT num_columns_;  ///< number of columns in the original matrix
    IT num_partitions_;     ///< stores the number of threads used for allocation. Should not change during runtime, or the data structure needs to be reinited.
    IT chunk_size_;         ///< number of rows computed by each thread

    /**
     *  @brief      Divide the matrix across rows in equally large partitions.
     *  @details    Sets the chunk_size_ as well as the slices_ attribute.
     */
    void divide_post_ranks(std::vector<IT> &row_indices, int num_partitions) {
        clear();

        num_partitions_ = num_partitions;
        chunk_size_ = static_cast<int>(ceil(static_cast<double>(num_rows_)/static_cast<double>(num_partitions)));

        for(int i = 0; i < num_partitions_; i++) {
            int beg = i * chunk_size_;
            int end = std::min(static_cast<int>((i+1) * chunk_size_), static_cast<int>(num_rows_));

            sub_matrices_.push_back(new SPARSE_MATRIX_TYPE(num_rows_, num_columns_));
        }

        // ATTENTION: this assumes that row_indices are sorted ascending
        int lower_bound = 0;
        for (int part_idx = 0; part_idx < num_partitions_; part_idx++) {
            int part_border = (part_idx+1) * chunk_size_;

            auto it = std::partition(row_indices.begin(), row_indices.end(), [&part_border](int i){return i < part_border;});
            int size = std::distance(row_indices.begin(), it);

            slices_.push_back(std::pair<int,int>(lower_bound, size));
            lower_bound = size;
        }
    }

public:
    std::vector<SPARSE_MATRIX_TYPE*> sub_matrices_;         // container which hold the partitions
    std::vector<std::pair<IT, IT>> slices_;                 // Encodde begin and end of each partition

    explicit PartitionedMatrix(const unsigned int num_rows, const unsigned int num_columns) :
        num_rows_(num_rows), num_columns_(num_columns) {
    }

    ~PartitionedMatrix() {
    }

    int chunk_size() {
        return this->chunk_size_;
    }

    void clear() {
        // delete old stuff
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++)
            delete *it;
        sub_matrices_.clear();
        slices_.clear();
    }

    //
    //  Connectivity accessors ( for Cython )
    //
    std::vector<IT> get_post_rank() {
        auto complete_post_ranks = std::vector<IT>();

        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            auto post_rank_slice = (*it)->get_post_rank();
            complete_post_ranks.insert(complete_post_ranks.end(), post_rank_slice.begin(), post_rank_slice.end());
        }

        return complete_post_ranks;
    }

    std::vector<std::vector<IT>> get_pre_ranks() {
        auto complete_pre_ranks = std::vector<std::vector<IT>>();

        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            auto tmp = (*it)->get_pre_ranks();

            complete_pre_ranks.insert(complete_pre_ranks.end(), tmp.begin(), tmp.end());
        }

        return complete_pre_ranks;
    }

    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        // find the correct partition
        auto it = slices_.begin();
        int part = 0;

        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                return sub_matrices_[part]->get_dendrite_pre_rank(lil_idx-it->first);
            }
        }

        // should not happen
        return std::vector<IT>();
    }

    int dendrite_size(int lil_idx) {
        auto it = slices_.begin();
        int part = 0;

        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                return sub_matrices_[part]->dendrite_size(lil_idx-it->first);
            }
        }
        return 0;
    }

    int nb_synapses() {
        int size = 0;
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            size += (*it)->nb_synapses();
        }
        return size;
    }

    int nb_dendrites() {
        int size = 0;
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            size += (*it)->nb_dendrites();
        }
        return size;
    }

    std::map<IT, IT> nb_efferent_synapses() {
        std::map<IT, IT> efferents;
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            auto sliced_efferents = (*it)->nb_efferent_synapses();
            efferents.insert(sliced_efferents.begin(), sliced_efferents.end());
        }
        return efferents;
    }

    bool init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks, const IT num_partitions) {
    #ifdef _DEBUG
        std::cout << "PartitionedMatrix::init_matrix_from_lil():" << std::endl;
    #endif

        // Sanity check
        assert ( (post_ranks.size() == pre_ranks.size()) );

        // determine partitions
        divide_post_ranks(post_ranks, num_partitions);

        auto slice_it = slices_.begin();
        int part_idx = 0;
        for(; slice_it != slices_.end(); slice_it++, part_idx++) {
            // create sub matrix with the previously determined slices
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+slice_it->first, post_ranks.begin()+slice_it->second);
            auto pre_rank_slice = std::vector< std::vector<IT> >(pre_ranks.begin()+slice_it->first, pre_ranks.begin()+slice_it->second);

            // initialize the sub-matrices
            bool success = sub_matrices_[part_idx]->init_matrix_from_lil(post_rank_slice, pre_rank_slice);
            if (!success) {
                std::cerr << "Failed to initialize partition " << part_idx << std::endl;
                return false;
            }
        }

    #ifdef _DEBUG
        std::cout << "PartitionedMatrix: created " << num_partitions << " partitions with partition borders:" << std::endl;
        for (int t = 0; t < num_partitions; t++) {
            std::cout << "  partition " << t << ": " << std::distance(post_ranks.begin(), post_ranks.begin()+slices_[t].first) <<
            " to " << std::distance(post_ranks.begin(), post_ranks.begin()+slices_[t].second) << 
            " ( " << slices_[t].second - slices_[t].first << " rows, "<< sub_matrices_[t]->nb_synapses() <<" nnz )" << std::endl;
        }
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
    template<typename VT, typename PART_TYPE, bool zero_based=true>
    std::vector<PART_TYPE> init_matrix_from_csv(const std::string filename, const IT num_partitions, const char delimiter=',') {
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
        init_matrix_from_lil(lil_ranks, lil_col_idx, num_partitions);

        // create the value matrix
        auto value = init_matrix_variable<VT, PART_TYPE>(0.0);
        update_matrix_variable_all<VT, PART_TYPE>(value, lil_values);

        return value;
    }

    //
    //  ANNarchy connectivity patterns
    //
    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::vector<std::mt19937>& rng, const unsigned int num_partitions) {
    #ifdef _DEBUG
        std::cout << "ParallelLIL::fixed_number_pre_pattern():" << std::endl;
    #endif
        // determine partitions
        divide_post_ranks(post_ranks, num_partitions);

    #if !defined(_DISABLE_PARALLEL_RNG) and defined(_OPENMP)
        #pragma omp parallel num_threads(num_partitions)
        {
            int tid = omp_get_thread_num();
            auto slice = slices_[tid];
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+slice.first, post_ranks.begin()+slice.second);
            sub_matrices_[tid]->fixed_number_pre_pattern(post_rank_slice, pre_ranks, nnz_per_row, rng[tid]);
        }
    #else
        // Create the matrix as in single thread
        auto lil_mat = new SPARSE_MATRIX_TYPE(num_rows_, num_columns_);
        lil_mat->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng[0]);

        // distribute towards threads
        auto slice_it = slices_.begin();
        int part_idx = 0;
        for(; slice_it != slices_.end(); slice_it++, part_idx++) {
            auto sliced_lil = lil_mat->slice_across_rows(slice_it->first, slice_it->second);
            sub_matrices_[part_idx]->init_matrix_from_lil(sliced_lil->post_rank, sliced_lil->pre_rank);
        }
    #endif
    }

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::vector<std::mt19937>& rng, const unsigned int num_partitions) {
    #ifdef _DEBUG
        std::cout << "ParallelLIL::fixed_probability_pattern():" << std::endl;
    #endif
        // determine partitions
        divide_post_ranks(post_ranks, num_partitions);

    #if !defined(_DISABLE_PARALLEL_RNG) and defined(_OPENMP)
        #pragma omp parallel num_threads(num_partitions)
        {
            int tid = omp_get_thread_num();
            auto slice = slices_[tid];
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+slice.first, post_ranks.begin()+slice.second);
            sub_matrices_[tid]->fixed_probability_pattern(post_rank_slice , pre_ranks, p, allow_self_connections, rng[tid]);
        }
    #else
        // Create the matrix as in single thread
        auto single_matrix = new SPARSE_MATRIX_TYPE(num_rows_, num_columns_);
        single_matrix->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng[0]);

        // distribute towards threads
        auto slice_it = slices_.begin();
        int part_idx = 0;
        for (; slice_it != slices_.end(); slice_it++, part_idx++) {
            auto sliced_lil = single_matrix->slice_across_rows(slice_it->first, slice_it->second);
            sub_matrices_[part_idx]->init_matrix_from_lil(sliced_lil->post_rank, sliced_lil->pre_rank);
        }
    #endif
    }

    //
    //  Initialize matrix variables ( post-size times pre-size divided into num_partitions chunks)
    //
    /**
     *  @brief      Initialize a matrix variable
     *  @tparam     PART_TYPE   as the slice type depends on the format, e. g. LIL vector<vector<VT>> and CSRC vector<VT>
     *                          but I wanted to have an unified interface, the PART_TYPE provides the necessary information.
     */
    template <typename VT, typename PART_TYPE>
    std::vector< PART_TYPE > init_matrix_variable(VT default_value) {
        auto new_variable = std::vector< PART_TYPE >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back((*it)->init_matrix_variable(default_value));
        }

        return new_variable;
    }

    template <typename VT, typename PART_TYPE>
    std::vector< PART_TYPE > init_matrix_variable_uniform(VT a, VT b, std::vector<std::mt19937>& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif

    #if !defined(_DISABLE_PARALLEL_RNG) and defined(_OPENMP)
        auto new_variable = std::vector< PART_TYPE >(num_partitions_);
        #pragma omp parallel num_threads(num_partitions_)
        {
            int tid = omp_get_thread_num();
            new_variable[tid] = std::move(sub_matrices_[tid]->init_matrix_variable_uniform(a, b, rng[tid]));
        }
        return new_variable;
    #else
        auto new_variable = std::vector< PART_TYPE >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_matrix_variable_uniform(a, b, rng[0])));
        }
        return new_variable;
    #endif
    }

    template <typename VT, typename PART_TYPE>
    std::vector< PART_TYPE > init_matrix_variable_normal(VT mean, VT sigma, std::vector<std::mt19937>& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Normal(" << mean << ", " << sigma << ")" << std::endl;
    #endif

    #if !defined(_DISABLE_PARALLEL_RNG) and defined(_OPENMP)
        auto new_variable = std::vector< PART_TYPE >(num_partitions_);
        #pragma omp parallel num_threads(num_partitions_)
        {
            int tid = omp_get_thread_num();
            new_variable[tid] = std::move(sub_matrices_[tid]->init_matrix_variable_normal(mean, sigma, rng[tid]));
        }
        return new_variable;
    #else
        auto new_variable = std::vector< PART_TYPE >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_matrix_variable_normal(mean, sigma, rng[0])));
        }
        return new_variable;
    #endif
    }

    //
    //  Update matrix variables
    //
    template <typename VT, typename PART_TYPE>
    inline void update_matrix_variable(std::vector< PART_TYPE > &variable, const IT lil_idx, const IT col_idx, const VT value)
    {
        assert ( (variable.size() == num_partitions_) );
        assert ( (slices_.size() == num_partitions_) );

        auto it = slices_.begin();
        int part = 0;
        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                sub_matrices_[part]->update_matrix_variable(variable[part], lil_idx-it->first, col_idx, value);
            }
        }        
    }

    template <typename VT, typename PART_TYPE>
    inline void update_matrix_variable_row(std::vector< PART_TYPE > &variable, const IT lil_idx, const std::vector<VT> data)
    {
        assert ( (variable.size() == num_partitions_) );
        assert ( (slices_.size() == num_partitions_) );

        auto it = slices_.begin();
        int part = 0;
        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                sub_matrices_[part]->update_matrix_variable_row(variable[part], lil_idx-it->first, data);
            }
        }
    }

    template <typename VT, typename PART_TYPE>
    inline void update_matrix_variable_all(std::vector< PART_TYPE > &variable, const std::vector< std::vector<VT> > &data)
    {
        assert ( (variable.size() == num_partitions_) );
        assert ( (slices_.size() == num_partitions_) );

        auto it = slices_.begin();
        int part_idx = 0;
        for(; it != slices_.end(); it++, part_idx++) {
            auto data_slice = std::vector< std::vector<VT> >(data.begin()+it->first, data.begin()+it->second);

            sub_matrices_[part_idx]->update_matrix_variable_all(variable[part_idx], data_slice);
        }
    }

    //
    //  Access matrix variables
    //
    template <typename VT, typename PART_TYPE>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector< PART_TYPE > &variable) {
        auto new_variable = std::vector< std::vector < VT > >();

        auto post_ranks = get_post_rank();
        for( int lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            new_variable.push_back(std::move(this->template get_matrix_variable_row<VT, PART_TYPE>(variable, lil_idx)));
        }
        return new_variable;
    }

    template <typename VT, typename PART_TYPE>
    inline std::vector< VT > get_matrix_variable_row(const std::vector< PART_TYPE > &variable, const IT &lil_idx) {
        // find the correct partition
        auto it = slices_.begin();
        int part_idx = 0;

        for(; it != slices_.end(); it++, part_idx++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                return sub_matrices_[part_idx]->get_matrix_variable_row(variable[part_idx], lil_idx-it->first);
            }
        }

        // should not happen
        return std::vector< VT >();
    }

    template <typename VT, typename PART_TYPE>
    inline VT get_matrix_variable(std::vector< PART_TYPE > &variable, const IT &row_idx, const IT &col_idx) {
        std::cerr << "Not implemented ..." << std::endl;
        return static_cast<VT>(0.0); // should not happen
    }

    //
    //  Initialize matrix variables ( post-size divided into num_partitions chunks)
    //
    template <typename VT>
    std::vector< std::vector< VT > > init_vector_variable(VT default_value) {
        auto new_variable = std::vector< std::vector<VT> >();

        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_vector_variable(default_value)));
        }

        return new_variable;
    }

    template <typename VT>
    inline std::vector<VT> get_vector_variable_all(std::vector< std::vector<VT> > variable) {
        auto new_variable = std::vector<VT>();

        int part = 0;
        auto it = sub_matrices_.begin();

        for(; it != sub_matrices_.end(); it++, part++) {
            auto tmp = std::move((*it)->get_vector_variable_all(variable[part]));
            new_variable.insert(new_variable.end(), tmp.begin(), tmp.end());
        }

        return new_variable;
    }

    template<typename VT>
    inline VT get_vector_variable(std::vector< std::vector<VT> > variable, int lil_idx) {
        // find the correct partition
        auto it = slices_.begin();
        int part_idx = 0;

        for(; it != slices_.end(); it++, part_idx++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                return sub_matrices_[part_idx]->get_vector_variable(variable[part_idx], lil_idx - it->first);
            }
        }
    }

    template<typename VT>
    inline void update_vector_variable_all(std::vector< std::vector<VT> > &variable, std::vector<VT> data) {
        assert ( (variable.size() == num_partitions_) );
        assert ( (slices_.size() == num_partitions_) );

        auto it = slices_.begin();
        int part = 0;
        int off = 0;
        auto beg = data.begin();
        for(; it != slices_.end(); it++, part++) {
            int part_size = sub_matrices_[part]->nb_dendrites();

            auto end = beg + part_size;
            sub_matrices_[part]->update_vector_variable_all(variable[part], std::vector<VT>(beg, end));

            beg = end;
        }
    }

    template<typename VT>
    inline void update_vector_variable(std::vector< std::vector<VT> > &variable, int lil_idx, VT data) {
        assert ( (variable.size() == num_partitions_) );
        assert ( (slices_.size() == num_partitions_) );

        auto it = slices_.begin();
        int part = 0;
        for(; it != slices_.end(); it++, part++) {
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                sub_matrices_[part]->update_vector_variable(variable[part], lil_idx-it->first, data);
            }
        }
    }

    void print_data_representation() {
        auto post_ranks = get_post_rank();
        std::cout << "ParallelLIL representation: "<< std::endl;
        std::cout << "   dimensions: " << num_rows_ << " x " << num_columns_ << std::endl;
        std::cout << "   post_rank.size(): " << post_ranks.size() << std::endl;
        std::cout << "   number threads(): " << num_partitions_ << std::endl;
        std::cout << "divided into:" << std::endl;
        for(auto it = slices_.begin(); it != slices_.end(); it++) {
            std::cout << "   (" << it->first << ", " << it->second << ")" << std::endl;
        }
        std::cout << "using chunk_size = " << chunk_size_ << std::endl;

        std::cout << "Partitioned matrices ..." << std::endl;
        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++)
            (*it)->print_data_representation();
  
    }

    // Returns size in bytes for connectivity
    size_t size_in_bytes() {
        // constants
        size_t size = 4 * sizeof(IT);

        // partitions
        size += sizeof(std::vector<SPARSE_MATRIX_TYPE*>);
        size += sub_matrices_.capacity() * sizeof(SPARSE_MATRIX_TYPE*);
        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++)
            size += static_cast<SPARSE_MATRIX_TYPE*>(*it)->size_in_bytes();

        // ranges
        size += sizeof(std::vector<std::pair<IT, IT>>);
        size += slices_.capacity() * sizeof(std::pair<IT, IT>);

        return size;
    };
};
