/*
 *
 *    ParallelLIL.hpp
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
 *  @brief      Wrapper class for handling multiple instances of LIL.
 *  @details    In order to support the parallel evaluation of expecially spiking networks
 *              we divide the whole matrix into as many as threads parts.
 *  @tparam     LIL_TYPE    LIL class, either LILMatrix or LILInvMatrix
 *  @tparam     IT          index type which should be the same as used in the LIL_TYPE declaration
 *  @todo       LIL_TYPE could be technically extended to CSRMatrix and others, as most of
 *              the functions call the corresponding functions from the sub_matrices_ instances.
 */
template<typename LIL_TYPE, typename IT = unsigned int>
class ParallelLIL {
public:
    std::vector<LIL_TYPE*> sub_matrices_;
    std::vector<std::pair<IT, IT>> slices_;       // Encodde begin and end of each partition

    const unsigned int num_rows_;
    const unsigned int num_columns_;
    unsigned int num_threads_;      ///< stores the number of threads used for allocation. Should not change during runtime, or the data structure needs to be reinited.
    unsigned int chunk_size_;       ///< number of rows computed by each thread

    void clear() {
        // delete old stuff
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++)
            delete *it;
        sub_matrices_.clear();
        slices_.clear();
    }

    /**
     *  @brief      Divide the matrix across rows in equally large partitions.
     *  @details    Sets the chunk_size_ as well as the slices_ attribute.
     */
    void divide_post_ranks(std::vector<IT> &row_indices, int num_threads) {
        clear();

        num_threads_ = num_threads;
        chunk_size_ = static_cast<int>(ceil(static_cast<double>(num_rows_)/static_cast<double>(num_threads)));

        for(int i = 0; i < num_threads_; i++) {
            int beg = i * chunk_size_;
            int end = std::min(static_cast<int>((i+1) * chunk_size_), static_cast<int>(num_rows_));

            sub_matrices_.push_back(new LIL_TYPE(num_rows_, num_columns_));
        }

        // ATTENTION: this assumes that row_indices are sorted ascending
        int lower_bound = 0;
        for(int part_idx = 0; part_idx < num_threads_; part_idx++) {
            int part_border = (part_idx+1) * chunk_size_;

            auto it = std::partition(row_indices.begin(), row_indices.end(), [&part_border](int i){return i < part_border;});
            int size = std::distance(row_indices.begin(), it);

            slices_.push_back(std::pair<int,int>(lower_bound, size));
            lower_bound = size;
        }
    }

    ParallelLIL(const unsigned int num_rows, const unsigned int num_columns) :
        num_rows_(num_rows), num_columns_(num_columns) {
    }

    //
    //  Connectivity accessors ( for Cython )
    //
    std::vector<IT> get_post_rank() { 
        auto complete_post_ranks = std::vector<IT>();

        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            complete_post_ranks.insert(complete_post_ranks.end(), (*it)->post_rank.begin(), (*it)->post_rank.end());
        }

        return complete_post_ranks;
    }

    std::vector<std::vector<IT>> get_pre_ranks() { 
        auto complete_pre_ranks = std::vector<std::vector<IT>>(); 

        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            auto tmp = it->get_pre_ranks();

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

    int nb_synapses(int lil_idx) {
        auto it = slices_.begin();
        int part = 0;

        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                return sub_matrices_[part]->nb_synapses(lil_idx-it->first);
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

    //
    //  Connectivity patterns
    //
    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks, const unsigned int num_threads) {
        assert ( (post_ranks.size() == pre_ranks.size()) );
    #ifdef _DEBUG
        std::cout << "ParallelLIL::init_matrix_from_lil():" << std::endl;
    #endif
        // determine partitions
        divide_post_ranks(post_ranks, num_threads);

        auto slice_it = slices_.begin();
        int part_idx = 0;
        for(; slice_it != slices_.end(); slice_it++, part_idx++) {
            // create sub matrix with the previously determined slices
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+slice_it->first, post_ranks.begin()+slice_it->second);
            auto pre_rank_slice = std::vector< std::vector<IT> >(pre_ranks.begin()+slice_it->first, pre_ranks.begin()+slice_it->second);

            sub_matrices_[part_idx]->init_matrix_from_lil(post_rank_slice, pre_rank_slice);
        }
    }

    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::vector<std::mt19937>& rng, const unsigned int num_threads) {
    #ifdef _DEBUG
        std::cout << "ParallelLIL::fixed_number_pre_pattern():" << std::endl;
    #endif
        // determine partitions
        divide_post_ranks(post_ranks, num_threads);

    #ifndef _DISABLE_PARALLEL_RNG
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto slice = slices_[tid];
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+slice.first, post_ranks.begin()+slice.second);
            sub_matrices_[tid]->fixed_number_pre_pattern(post_rank_slice, pre_ranks, nnz_per_row, rng[tid]);
        }
    #else
        // Create the matrix as in single thread
        auto lil_mat = new LIL_TYPE(num_rows_, num_columns_);
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

    void fixed_probability_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, double p, bool allow_self_connections, std::vector<std::mt19937>& rng, const unsigned int num_threads) {
    #ifdef _DEBUG
        std::cout << "ParallelLIL::fixed_probability_pattern():" << std::endl;
    #endif
        // determine partitions
        divide_post_ranks(post_ranks, num_threads);

        auto it = slices_.begin();
        int part_idx = 0;
        for(; it != slices_.end(); it++, part_idx++) {
            // create sub matrix with the previously determined slices
            auto post_rank_slice = std::vector<IT>(post_ranks.begin()+it->first, post_ranks.begin()+it->second);

            sub_matrices_[part_idx]->fixed_probability_pattern(post_rank_slice , pre_ranks, p, allow_self_connections, rng[0]);
        }
    }

    //
    //  Initialize matrix variables ( post-size times pre-size divided into num_threads chunks)
    //
    template <typename VT>
    std::vector< std::vector< std::vector<VT> > > init_matrix_variable(VT default_value) {    
        auto new_variable = std::vector< std::vector< std::vector<VT> > >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back((*it)->init_matrix_variable(default_value));
        }

        return new_variable;
    }

    template <typename VT>
    std::vector< std::vector< std::vector<VT> > > init_matrix_variable_uniform(VT a, VT b, std::vector<std::mt19937>& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif

    #ifndef _DISABLE_PARALLEL_RNG
        auto new_variable = std::vector< std::vector< std::vector<VT> > >(num_threads_);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            new_variable[tid] = std::move(sub_matrices_[tid]->init_matrix_variable_uniform(a, b, rng[tid]));
        }
        return new_variable;
    #else
        auto new_variable = std::vector< std::vector< std::vector<VT> > >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_matrix_variable_uniform(a, b, rng[0])));
        }
        return new_variable;
    #endif        
    }

    template <typename VT>
    std::vector< std::vector< std::vector<VT> > > init_matrix_variable_normal(VT mean, VT sigma, std::vector<std::mt19937>& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Normal(" << mean << ", " << sigma << ")" << std::endl;
    #endif

    #ifndef _DISABLE_PARALLEL_RNG
        auto new_variable = std::vector< std::vector< std::vector<VT> > >(num_threads_);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            new_variable[tid] = std::move(sub_matrices_[tid]->init_matrix_variable_normal(mean, sigma, rng[tid]));
        }
        return new_variable;
    #else
        auto new_variable = std::vector< std::vector< std::vector<VT> > >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_matrix_variable_normal(mean, sigma, rng[0])));
        }
        return new_variable;
    #endif        
    }

    //
    //  Update matrix variables
    //
    template <typename VT>
    inline void update_matrix_variable(std::vector< std::vector< std::vector<VT> > > &variable, const IT lil_idx, const IT col_idx, const VT value) {
        assert ( (variable.size() == num_threads_) );
        assert ( (slices_.size() == num_threads_) );

        auto it = slices_.begin();
        int part = 0;
        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                sub_matrices_[part]->update_matrix_variable(variable[part], lil_idx-it->first, col_idx, value);
            }
        }        
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector< std::vector< std::vector<VT> > > &variable,
                             const IT lil_idx,
                             const std::vector<VT> data)
    {
        assert ( (variable.size() == num_threads_) );
        assert ( (slices_.size() == num_threads_) );

        auto it = slices_.begin();
        int part = 0;
        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                sub_matrices_[part]->update_matrix_variable_row(variable[part], lil_idx-it->first, data);
            }
        }
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector< std::vector< std::vector<VT> > > &variable,
                             const std::vector< std::vector<VT> > &data)
    {
        assert ( (variable.size() == num_threads_) );
        assert ( (slices_.size() == num_threads_) );

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
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector< std::vector< std::vector<VT> > > &variable) {
        auto new_variable = std::vector< std::vector < VT > >();

        auto post_ranks = get_post_rank();
        for( int lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            new_variable.push_back(std::move(get_matrix_variable_row(variable, lil_idx)));
        }
        return new_variable;
    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const std::vector< std::vector< std::vector<VT> > > &variable, const IT &lil_idx) {
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

    template <typename VT>
    inline VT get_matrix_variable(std::vector< std::vector< std::vector<VT> > > &variable, const IT &row_idx, const IT &col_idx) {

        return static_cast<VT>(0.0); // should not happen
    }

    //
    //  Initialize matrix variables ( post-size divided into num_threads chunks)
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
    inline void update_vector_variable_all(std::vector< std::vector<VT> > variable, std::vector<VT> data) {
        assert ( (variable.size() == num_threads_) );
        assert ( (slices_.size() == num_threads_) );

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
    inline void update_vector_variable(std::vector< std::vector<VT> > variable, int lil_idx, VT data) {
        assert ( (variable.size() == num_threads_) );
        assert ( (slices_.size() == num_threads_) );

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
        std::cout << "   number threads(): " << num_threads_ << std::endl;
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
        return 0;
    };
};