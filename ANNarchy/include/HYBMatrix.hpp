/*
 * COOMatrix.hpp
 *
 * Copyright (c) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 *  As the hybrid format is a combination of two formats, one can not simply
 *  express a variable as one single container.
 */
template<typename VT>
struct hyb_local {
    std::vector< VT > ell;
    std::vector< VT > coo;

    hyb_local() {
    }

    ~hyb_local() {
        ell.clear();
        ell.shrink_to_fit();

        coo.clear();
        coo.shrink_to_fit();
    }
};

/**
 *  @brief      Implementation of the hybrid (HYB) sparse matrix format.
 *  @details    The hybrid format, originally proposed by Bell and Garland 2009 for GPUs, combines
 *              the ELLPACK format and the coordinate format. The first is impaired if the mean 
 *              average of nonzeros per row differs too much from the maximum average nonzeros per row.
 */
template<typename IT = unsigned int, bool row_major=true>
class HYBMatrix {
  protected:
    ELLMatrix<IT, row_major> *ell_matrix_;
    COOMatrix<IT> *coo_matrix_;
    unsigned int ell_size_;

  public:
    HYBMatrix(const IT num_rows, const IT num_columns){
        ell_matrix_ = new ELLMatrix<IT, row_major>(num_rows, num_columns);
        coo_matrix_ = new COOMatrix<IT>(num_rows, num_columns);
    }

    std::vector<IT> get_post_rank() {
        return std::move(ell_matrix_->get_post_rank());
    }

    std::vector<std::vector<IT>> get_pre_ranks() { 
        auto pre_ranks = ell_matrix_->get_pre_ranks();        

        for (int lil_idx = 0; lil_idx < pre_ranks.size(); lil_idx++) {
            auto coo_pre_ranks = coo_matrix_->get_dendrite_pre_rank(lil_idx);
            pre_ranks[lil_idx].insert(pre_ranks[lil_idx].end(), coo_pre_ranks.begin(), coo_pre_ranks.end());
        }

        return pre_ranks; 
    }

    std::vector<IT> get_dendrite_pre_rank(int lil_idx) {
        auto ell_pre_rank = ell_matrix_->get_dendrite_pre_rank(lil_idx);
        auto coo_pre_ranks = coo_matrix_->get_dendrite_pre_rank(lil_idx);

        ell_pre_rank.insert(ell_pre_rank.end(), coo_pre_ranks.begin(), coo_pre_ranks.end());
        return ell_pre_rank;
    }

    void init_matrix_from_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices, unsigned int ell_size) {
        if (ell_size != std::numeric_limits<unsigned int>::max()) {
            ell_size_ = ell_size;
        } else {
            unsigned int nnz = 0;
            for(auto it = column_indices.begin(); it != column_indices.end(); it++)
                nnz += it->size();
            
            ell_size_ = static_cast<unsigned int>(ceil(static_cast<double>(nnz)/static_cast<double>(row_indices.size())));
        }
    #ifdef _DEBUG
        std::cout << "HYBMatrix::init_matrix_from_lil()" << std::endl;
        std::cout << "  ell_size = " << ell_size_ << std::endl;
    #endif

        std::vector< std::vector<IT> > ell_part;
        std::vector< std::vector<IT> > coo_part;

        for(auto it = column_indices.begin(); it != column_indices.end(); it++) {
            if (it->size() <= ell_size_) {
                ell_part.push_back(std::vector<IT>(it->begin(), it->end()));
                coo_part.push_back(std::vector<IT>());
            }else{
                ell_part.push_back(std::vector<IT>(it->begin(), it->begin()+ell_size_));
                coo_part.push_back(std::vector<IT>(it->begin()+ell_size_, it->end()));
            }
        }

        ell_matrix_->init_matrix_from_lil(row_indices, ell_part);
        coo_matrix_->init_matrix_from_lil(row_indices, coo_part);

    #ifdef _DEBUG
        std::cout << "HYBMatrix::init_matrix_from_lil()" << std::endl;
        std::cout << "  nnz in ell = " << ell_matrix_->nb_synapses() << " (" << static_cast<double>(ell_matrix_->nb_synapses()) / static_cast<double>(nb_synapses()) * 100.0 << "%)" << std::endl;
        std::cout << "  nnz in coo = " << coo_matrix_->nb_synapses() << " (" << static_cast<double>(coo_matrix_->nb_synapses()) / static_cast<double>(nb_synapses()) * 100.0 << "%)" << std::endl;
    #endif
    }

    unsigned int nb_synapses() {
        return ell_matrix_->nb_synapses() + coo_matrix_->nb_synapses();
    }

    unsigned int nb_synapses(int lil_idx) {
        return ell_matrix_->nb_synapses(lil_idx) + coo_matrix_->nb_synapses(lil_idx);
    }

    unsigned int nb_dendrites() {
        return ell_matrix_->nb_dendrites();
    }

    template<typename VT>
    hyb_local<VT> init_matrix_variable(VT default_value) {
        hyb_local<VT> new_variable;

        new_variable.coo = std::move(coo_matrix_->init_matrix_variable(default_value));
        new_variable.ell = std::move(ell_matrix_->init_matrix_variable(default_value));

        return new_variable;
    }

    template <typename VT>
    hyb_local<VT> init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        std::cout << "CSRMatrix::initialize_variable_uniform(): arguments = (" << a << ", " << b << ") and num_non_zeros_ = " << num_non_zeros_ << std::endl;
    #endif
        hyb_local<VT> new_variable;

        new_variable.coo = std::move(coo_matrix_->init_matrix_variable_uniform(a, b, rng));
        new_variable.ell = std::move(ell_matrix_->init_matrix_variable_uniform(a, b, rng));

        return new_variable;
    }

    template <typename VT>
    inline void update_matrix_variable_all(hyb_local<VT> &variable, const std::vector< std::vector<VT> > &data) {
        std::vector< std::vector<VT> > ell_part;
        std::vector< std::vector<VT> > coo_part;

        for(auto it = data.begin(); it != data.end(); it++) {
            if (it->size() <= ell_size_) {
                ell_part.push_back(std::vector<VT>(it->begin(), it->end()));
                coo_part.push_back(std::vector<VT>());
            }else{
                ell_part.push_back(std::vector<VT>(it->begin(), it->begin()+ell_size_));
                coo_part.push_back(std::vector<VT>(it->begin()+ell_size_, it->end()));
            }
        }

        ell_matrix_->update_matrix_variable_all(variable.ell, ell_part);
        coo_matrix_->update_matrix_variable_all(variable.coo, coo_part);
    }

    template <typename VT>
    inline void update_matrix_variable_row(hyb_local<VT> &variable, const IT lil_idx, const std::vector<VT> data) {
        std::cerr << "Not implemented" << std::endl;
    }

    template <typename VT>
    inline void update_matrix_variable(hyb_local<VT> &variable, const IT row_idx, const IT column_idx, const VT value) {
        std::cerr << "Not implemented" << std::endl;
    }

    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const hyb_local<VT> &variable) {
        auto lil_variable = std::vector< std::vector < VT > >();

        return lil_variable;
    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const hyb_local<VT>& variable, const IT &lil_idx) {
        auto result = std::vector<VT>();

        result = ell_matrix_->get_matrix_variable_row(variable.ell, lil_idx);

        if (result.size() == ell_size_) {
            auto coo_result = coo_matrix_->get_matrix_variable_row(variable.coo, lil_idx);
            result.insert(result.end(), coo_result.begin(), coo_result.end());
        }

        return result;
    }

    template <typename VT>
    inline VT get_matrix_variable(const hyb_local<VT>& variable, const IT &lil_idx, const IT &col_idx) {

        return static_cast<VT>(0.0); // should not happen
    }

    size_t size_in_bytes() {
        size_t size = 0;

        size += ell_matrix_->size_in_bytes();
        size += coo_matrix_->size_in_bytes();
        size += sizeof(ell_size_);
        size += 2*sizeof(void*);        // 2 class references

        return size;
    }

};
