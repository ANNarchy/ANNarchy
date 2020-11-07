/*
 *
 *    CSRCMatrixTOMP.hpp
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

template<typename IT = unsigned int>
class CSRCMatrixTOMP{
  protected:
    std::vector<CSRCMatrixT<IT>*> sub_matrices_;
    std::vector<std::pair<int, int>> slices_;       // Encodde begin and end of each partition

    const unsigned int num_rows_;
    const unsigned int num_columns_;
    unsigned int num_threads_;
    unsigned int chunk_size_;

    void clear() {
        // delete old stuff
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++)
            delete *it;
        sub_matrices_.clear();
        slices_.clear();
    }

  public:
    CSRCMatrixTOMP(const unsigned int num_rows, const unsigned int num_columns) :
        num_rows_(num_rows), num_columns_(num_columns) {
    }

    //
    //  Connectivity accessors ( for Cython )
    //
    std::vector<IT> get_post_rank() { 
        auto complete_post_ranks = std::vector<IT>();

        for (auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            auto tmp = (*it)->get_post_rank();

            complete_post_ranks.insert(complete_post_ranks.end(), tmp.begin(), tmp.end());
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

        // should not happen ...
        return 0;
    }

    //
    //  Connectivity patterns
    //
    void init_matrix_from_lil(std::vector<IT> &post_ranks, std::vector< std::vector<IT> > &pre_ranks, const unsigned int num_threads) {
        clear();

        assert ( (post_ranks.size() == pre_ranks.size()) );
    #ifdef _DEBUG
        std::cout << "CSRCMatrixTOMP::init_matrix_from_lil():" << std::endl;
    #endif
        auto lil_omp_matrix = new ParallelLIL<LILMatrix<IT>, IT>(num_rows_, num_columns_);
        lil_omp_matrix->init_matrix_from_lil(post_ranks, pre_ranks, num_threads);

        for (unsigned int i = 0; i < lil_omp_matrix->num_threads_; i++) {
        #ifdef _DEBUG
            std::cout << "Start to initialize " << i << " of " << num_threads << " partitions ... " << std::endl;
        #endif
            auto lil_mat = lil_omp_matrix->sub_matrices_[i];
            auto csrc_t_mat = new CSRCMatrixT<IT>(lil_mat->num_rows_, lil_mat->num_columns_);

            if (lil_mat->nb_synapses() > 0)
                csrc_t_mat->init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());

            sub_matrices_.push_back(csrc_t_mat);
        }
        
        // slices from orgin LIL matrix
        slices_ = lil_omp_matrix->slices_;
        num_threads_ = num_threads;

        delete lil_omp_matrix;
    }

    void fixed_number_pre_pattern(std::vector<IT> post_ranks, std::vector<IT> pre_ranks, unsigned int nnz_per_row, std::vector<std::mt19937>& rng, const unsigned int num_threads) {
        clear();

        // The T-matrices are initialized with flipped pre and post dimensions
        auto lil_omp_matrix = new ParallelLIL<LILMatrix<IT>, IT>(num_rows_, num_columns_);
        lil_omp_matrix->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng, num_threads);
    #ifdef _DEBUG
        std::cout << "PostToPre matrix (LIL): " << lil_omp_matrix->num_rows_ << " times " << lil_omp_matrix->num_columns_ << " and " << lil_omp_matrix->nb_synapses() << " entries." << std::endl;
    #endif

        for (unsigned int i = 0; i < lil_omp_matrix->num_threads_; i++) {
            auto lil_mat = lil_omp_matrix->sub_matrices_[i];

            auto csrc_t_mat = new CSRCMatrixT<IT>(lil_mat->num_rows_, lil_mat->num_columns_);
            csrc_t_mat->init_matrix_from_lil(lil_mat->get_post_rank(), lil_mat->get_pre_ranks());
            sub_matrices_.push_back(csrc_t_mat);
        }
        
        // slices from orgin LIL matrix
        slices_ = lil_omp_matrix->slices_;
        num_threads_ = num_threads;

        delete lil_omp_matrix;
    }

    //
    //  Initialize variables
    //
    template <typename VT>
    std::vector< std::vector<VT> > init_matrix_variable(VT default_value) {    
        auto new_variable = std::vector< std::vector<VT> >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back((*it)->init_matrix_variable(default_value));
        }

        return new_variable;
    }

    template <typename VT>
    std::vector< std::vector<VT> > init_matrix_variable_uniform(VT a, VT b, std::vector<std::mt19937>& rng) {
    #ifdef _DEBUG
        std::cout << "Initialize variable with Uniform(" << a << ", " << b << ")" << std::endl;
    #endif

    #ifndef _DISABLE_PARALLEL_RNG
        auto new_variable = std::vector< std::vector<VT> >(num_threads_);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            new_variable[tid] = std::move(sub_matrices_[tid]->init_matrix_variable_uniform(a, b, rng[tid]));
        }
        return new_variable;
    #else
        auto new_variable = std::vector< std::vector<VT> >();
        for(auto it = sub_matrices_.begin(); it != sub_matrices_.end(); it++) {
            new_variable.push_back(std::move((*it)->init_matrix_variable_uniform(a, b, rng[0])));
        }
        return new_variable;
    #endif        
    }

    //
    //  Update variables
    //
    template <typename VT>
    inline void update_matrix_variable(std::vector< std::vector<VT> > &variable, const IT lil_idx, const IT col_idx, const VT value) {
    }

    template <typename VT>
    inline void update_matrix_variable_row(std::vector< std::vector<VT> > &variable,
                             const IT lil_idx,
                             const std::vector<VT> data)
    {
    }

    template <typename VT>
    inline void update_matrix_variable_all(std::vector< std::vector<VT> > &variable,
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
    //  Access variables
    //
    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const std::vector< std::vector<VT> > &variable) {
        auto new_variable = std::vector< std::vector < VT > >();

        /*
        auto post_ranks = get_post_rank();
        for( int lil_idx = 0; lil_idx < post_ranks.size(); lil_idx++) {
            new_variable.push_back(std::move(get_matrix_variable_row(variable, lil_idx)));
        }
        */
        return new_variable;

    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(std::vector< std::vector<VT> > &variable, const IT &lil_idx) {
        // find the correct partition
        auto it = slices_.begin();
        int part = 0;

        for(; it != slices_.end(); it++, part++){
            if ((lil_idx >= it->first) && (lil_idx < it->second)) {
                auto csrc_mat = sub_matrices_[part];
                int rel_lil_idx = lil_idx-it->first;
                return csrc_mat->get_matrix_variable_row(variable[part], rel_lil_idx);
            }
        }

        // should not happen
        return std::vector< VT >();
    }

    template <typename VT>
    inline VT get_matrix_variable(std::vector< std::vector<VT> > &variable, const IT &row_idx, const IT &col_idx) {

        return static_cast<VT>(0.0); // should not happen
    }

    // Returns size in bytes for connectivity
    size_t size_in_bytes() {
        return 0;
    };
};