/*
 *    HYBMatrix.hpp
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

#include "COOMatrix.hpp"
#include "ELLMatrix.hpp"

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

    void clear() {
        ell.clear();
        ell.shrink_to_fit();

        coo.clear();
        coo.shrink_to_fit();
    }

    size_t size_in_bytes() {
        size_t size=0;

        size += 2*sizeof(std::vector<VT>);

        size += ell.capacity() * sizeof(VT);
        size += coo.capacity() * sizeof(VT);

        return size;
    }

    ~hyb_local() {
    }
};

/**
 *  @brief      Implementation of the hybrid (HYB) sparse matrix format.
 *  @details    The hybrid format, originally proposed by Bell and Garland (2009) for GPUs, combines
 *              the ELLPACK format and the coordinate format. The hybrid format should improve the
 *              performance of SpMV on matrices where the average of nonzeros per row differs
 *              too much from the maximum average nonzeros per row.
 */
template<typename IT = unsigned int, typename ST=unsigned long int, bool row_major=true>
class HYBMatrix {
  protected:
    ELLMatrix<IT, ST, row_major> *ell_matrix_;  ///< partition of the matrix represented as ELLPACK
    COOMatrix<IT, ST> *coo_matrix_;             ///< partition of the matrix represented as COOrdinate format
    unsigned int ell_size_;                     ///< row-length of the ELLPACK partition (either provided by the user or determined within determine_ell_size() )
    const IT num_rows_;
    const IT num_columns_;

    /**
     *  @brief      Determine size of ELLPACK partition
     *  @details    We need to define the cut between ELL and COO which can be determined in multiple ways.
     *              This version determines a histogram on the row-lengths. And take the highest one as threshold.
     *              This version was suggested by: 
     *              Bell & Garland (2009) Implementing sparse matrix-vector multiplication on throughput-oriented processors
     */
    IT determine_ell_size_hist(const std::vector<IT> row_indices, const std::vector<std::vector<IT>> column_indices) {
        std::map<IT, int> row_length_hist;
        for(auto it = column_indices.begin(); it != column_indices.end(); it++) {
            row_length_hist[it->size()]++;
        }

    #ifdef _DEBUG
        std::cout << "Row-length distribution: " << std::endl;
        for (auto it = row_length_hist.begin(); it != row_length_hist.end(); it++) {
            std::cout << "  " << static_cast<long>(it->first) << ": " << static_cast<long>(it->second) << std::endl;
        }
    #endif

        int max = row_length_hist.begin()->second;
        int key = -1;
        for (auto it = row_length_hist.begin(); it != row_length_hist.end(); it++) {
            if (it->second > max) {
                max = it->second;
                key = it->first;
            }
        }

    #ifdef _DEBUG
        std::cout << "selected " << key << " based on row-length distribution." << std::endl;
    #endif
        return key;
    }

    double compute_ell_partition_size(const unsigned int ell_size_candidate, const std::vector<IT> &row_sizes) {
        unsigned int hits = 0;

        for(auto it = row_sizes.begin(); it != row_sizes.end(); it++)
            if (*it == ell_size_candidate) hits++;

        return double(hits)/double(row_sizes.size());
    }

    /**
     *  @brief      Determine size of ELLPACK partition
     *  @details    We need to define the cut between ELL and COO. In my oppinion this can be either the minimum
     *              row-length or the average row-length. But as we use ELLPACK as partition, we should ensure
     *              that all rows are completely filled. But also enough rows must be then in the ELLPACK partition.
     */
    IT determine_ell_size_avg(const std::vector<IT> row_indices, const std::vector<std::vector<IT>> column_indices) {
    #ifdef _DEBUG
        std::cout << "HYBMatrix::determine_ell_size() - try to determine a good partition size" << std::endl;
    #endif
        auto row_sizes = std::vector<IT>();
        for(auto it = column_indices.begin(); it != column_indices.end(); it++) {
            row_sizes.push_back(it->size());
        }

        IT min_nnz = column_indices[0].size();
        IT max_nnz = column_indices[0].size();
        for(auto it = row_sizes.begin(); it != row_sizes.end(); it++) {
            if ( (*it < min_nnz) && (*it > 0) )
                min_nnz = *it;
            if ( *it > max_nnz )
                max_nnz = *it;
        }

        unsigned int nnz = 0;
        for(auto it = column_indices.begin(); it != column_indices.end(); it++) {
            nnz += it->size();
        }
        double avg_nnz = static_cast<unsigned int>(ceil(static_cast<double>(nnz)/static_cast<double>(row_indices.size())));

    #ifdef _DEBUG
        std::cout << "   possible range (min, avg, max): " << min_nnz << ", " << avg_nnz << ", " << max_nnz << std::endl;
    #endif

        double max_cov = 0.0;
        double selected_size = -1;
        for (int i = min_nnz; i <= avg_nnz; i++) {
            auto ell_part_size = compute_ell_partition_size(i, row_sizes);
            if (ell_part_size > max_cov) {
                max_cov = ell_part_size;
                selected_size = i;
            }
        }
    #ifdef _DEBUG
        std::cout << "   selected size: "<< selected_size << " which will cover " << max_cov*100.0 << " percent of rows." << std::endl;
    #endif
        return selected_size;
    }

  public:
    /**
     *  @brief  constructor
     */
    explicit HYBMatrix(const IT num_rows, const IT num_columns):
        num_rows_(num_rows), num_columns_(num_columns) {
        ell_matrix_ = new ELLMatrix<IT, ST, row_major>(num_rows, num_columns);
        coo_matrix_ = new COOMatrix<IT, ST>(num_rows, num_columns);
    }

    ~HYBMatrix() {
    #ifdef _DEBUG
        std::cout << "HYBMatrix::~HYBMatrix()" << std::endl;
    #endif
        delete ell_matrix_;
        delete coo_matrix_;
    }

    /**
     *  @brief      Replace the ELLPACK, COOrdinate pointers
     *  @details    This function is only intended for the usage within the HYBMatrixCUDA class.
     */
    void replace_pointer(ELLMatrix<IT, ST, row_major>* ell_ptr, COOMatrix<IT, ST>* coo_ptr) {
    #ifdef _DEBUG
        std::cout << "HYBMatrix::replace_pointer() - destroy 'old' content" << std::endl;
    #endif
        delete ell_matrix_;
        delete coo_matrix_;

    #ifdef _DEBUG
        std::cout << "HYBMatrix::replace_pointer() - set new pointer" << std::endl;
    #endif
        ell_matrix_ = ell_ptr;
        coo_matrix_ = coo_ptr;
    }

    virtual void clear() {
    #ifdef _DEBUG
        std::cout << "HYBMatrix::clear()" << std::endl;
    #endif
        // clean up partial matrices
        ell_matrix_->clear();
        coo_matrix_->clear();
    }

    IT num_rows() {
        return num_rows_;
    }

    IT num_columns() {
        return num_columns_;
    }

    std::vector<IT> get_post_rank() {
        return ell_matrix_->get_post_rank();
    }

    ELLMatrix<IT, ST, row_major>* get_ell_instance() {
        return ell_matrix_;
    }

    COOMatrix<IT, ST>* get_coo_instance() {
        return coo_matrix_;
    }

    unsigned int ell_part_size() {
        return ell_size_;
    }

    size_t nb_synapses() {
        return ell_matrix_->nb_synapses() + coo_matrix_->nb_synapses();
    }

    IT dendrite_size(int lil_idx) {
        IT ell_dendrite_size = 0;
        IT coo_dendrite_size = 0;

        ell_dendrite_size = ell_matrix_->dendrite_size(lil_idx);
        coo_dendrite_size = coo_matrix_->dendrite_size(lil_idx);
        
        return ell_dendrite_size + coo_dendrite_size;
    }

    unsigned int nb_dendrites() {
        return ell_matrix_->nb_dendrites();
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
        auto pre_ranks = ell_matrix_->get_dendrite_pre_rank(lil_idx);

        auto coo_pre_ranks = coo_matrix_->get_dendrite_pre_rank(lil_idx);
        pre_ranks.insert(pre_ranks.end(), coo_pre_ranks.begin(), coo_pre_ranks.end());
        return pre_ranks;
    }

    /*
     *
     */
    bool init_matrix_from_lil(std::vector<IT> row_indices, std::vector< std::vector<IT> > column_indices, unsigned int ell_size=std::numeric_limits<unsigned int>::max()) {
        if (ell_size != std::numeric_limits<unsigned int>::max()) {
            ell_size_ = ell_size;
        } else {
            // Algorithmic search for the ELLPACK partition size
            //ell_size_ = determine_ell_size_avg(row_indices, column_indices);
            ell_size_ = determine_ell_size_hist(row_indices, column_indices);
        }
    #ifdef _DEBUG
        std::cout << "HYBMatrix::init_matrix_from_lil()" << std::endl;
        std::cout << "  ell_size = " << ell_size_ << std::endl;
    #endif

        std::vector< std::vector<IT> > ell_part_column_indices;
        std::vector< std::vector<IT> > coo_part_column_indices;

        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();
        for(; col_it != column_indices.end(); col_it++, row_it++) {
            if (col_it->size() < ell_size_) {
                ell_part_column_indices.push_back(std::vector<IT>(col_it->begin(), col_it->end()));
                coo_part_column_indices.push_back(std::vector<IT>());
            } else {
                ell_part_column_indices.push_back(std::vector<IT>(col_it->begin(), col_it->begin()+ell_size_));
                coo_part_column_indices.push_back(std::vector<IT>(col_it->begin()+ell_size_, col_it->end()));
            }
        }
    #ifdef _DEBUG
        std::cout << "  ELL-sizes = [";
        for (auto it = ell_part_column_indices.begin(); it != ell_part_column_indices.end(); it++) {
            std::cout << it->size() << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  COO-sizes = [";
        for (auto it = coo_part_column_indices.begin(); it != coo_part_column_indices.end(); it++) {
            std::cout << it->size() << ", ";
        }
        std::cout << "]" << std::endl;
    #endif

        bool ell_success = ell_matrix_->init_matrix_from_lil(row_indices, ell_part_column_indices);
        bool coo_success = coo_matrix_->init_matrix_from_lil(row_indices, coo_part_column_indices);
        if (!ell_success || !coo_success)
            return false;

    #ifdef _DEBUG
        std::cout << "HYBMatrix::init_matrix_from_lil()" << std::endl;
        std::cout << "  nnz in ell = " << ell_matrix_->nb_synapses() << " (" << static_cast<double>(ell_matrix_->nb_synapses()) / static_cast<double>(nb_synapses()) * 100.0 << "%)" << std::endl;
        std::cout << "  nnz in coo = " << coo_matrix_->nb_synapses() << " (" << static_cast<double>(coo_matrix_->nb_synapses()) / static_cast<double>(nb_synapses()) * 100.0 << "%)" << std::endl;
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
    hyb_local<VT>* init_matrix_from_csv(const std::string filename, const char delimiter=',', unsigned int ell_size=std::numeric_limits<unsigned int>::max()) {
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
        init_matrix_from_lil(lil_ranks, lil_col_idx, ell_size);

        // create the values matrix
        auto value = init_matrix_variable<VT>(0.0);
        update_matrix_variable_all<VT>(value, lil_values);

        return value;
    }

    template<typename VT>
    hyb_local<VT>* init_matrix_variable(VT default_value) {
    #ifdef _DEBUG
        std::cout << "HYBMatrix::init_matrix_variable(" << default_value << ")" << std::endl;
    #endif
        auto new_variable = new hyb_local<VT>();

        new_variable->coo = std::move(coo_matrix_->init_matrix_variable(default_value));
        new_variable->ell = std::move(ell_matrix_->init_matrix_variable(default_value));

        return new_variable;
    }

    template <typename VT>
    hyb_local<VT>* init_matrix_variable_uniform(VT a, VT b, std::mt19937& rng) {
    #ifdef _DEBUG
        int num_non_zeros = ell_matrix_->nb_synapses() + coo_matrix_->nb_synapses();
        std::cout << "HYBMatrix::initialize_variable_uniform(): arguments = (" << a << ", " << b << ") and num_non_zeros_ = " << num_non_zeros << std::endl;
    #endif
        auto new_variable = new hyb_local<VT>();

        new_variable.coo = std::move(coo_matrix_->init_matrix_variable_uniform(a, b, rng));
        new_variable.ell = std::move(ell_matrix_->init_matrix_variable_uniform(a, b, rng));

        return new_variable;
    }

    template <typename VT>
    inline void update_matrix_variable_all(hyb_local<VT>* variable, const std::vector< std::vector<VT> > &data) {
    #ifdef _DEBUG
        std::cout << "HYBMatrix()::update_matrix_variable_all()" << std::endl;
    #endif

        for (int lil_idx = 0; lil_idx < data.size(); lil_idx++) {
            if (data[lil_idx].size() <= ell_size_) {
                ell_matrix_->update_matrix_variable_row(variable->ell, lil_idx, std::vector<VT>(data[lil_idx].begin(), data[lil_idx].end()));
            } else {
                ell_matrix_->update_matrix_variable_row(variable->ell, lil_idx, std::vector<VT>(data[lil_idx].begin(), data[lil_idx].begin()+ell_size_));
                coo_matrix_->update_matrix_variable_row(variable->coo, lil_idx, std::vector<VT>(data[lil_idx].begin()+ell_size_, data[lil_idx].end()));
            }
        }
    }

    template <typename VT>
    inline void update_matrix_variable_row(hyb_local<VT>* variable, const IT lil_idx, const std::vector<VT> data) {
        std::cerr << "Not implemented" << std::endl;
    }

    template <typename VT>
    inline void update_matrix_variable(hyb_local<VT>* variable, const IT row_idx, const IT column_idx, const VT value) {
        std::cerr << "Not implemented" << std::endl;
    }

    template <typename VT>
    inline std::vector< std::vector < VT > > get_matrix_variable_all(const hyb_local<VT>* variable) {
        auto lil_variable = std::vector< std::vector < VT > >();

        for(int r = 0; r < ell_matrix_->nb_dendrites(); r++) {
            lil_variable.push_back(get_matrix_variable_row(variable, r));
        }

        return lil_variable;
    }

    template <typename VT>
    inline std::vector< VT > get_matrix_variable_row(const hyb_local<VT>* variable, const IT &lil_idx) {
        auto result = ell_matrix_->get_matrix_variable_row(variable->ell, lil_idx);

        auto coo_result = coo_matrix_->get_matrix_variable_row(variable->coo, lil_idx);
        result.insert(result.end(), coo_result.begin(), coo_result.end());

        return result;
    }

    template <typename VT>
    inline VT get_matrix_variable(const hyb_local<VT>* variable, const IT &lil_idx, const IT &col_idx) {

        return static_cast<VT>(0.0); // should not happen
    }

    virtual size_t size_in_bytes() {
        size_t size = 0;

        size += static_cast<ELLMatrix<IT, ST, row_major>*>(ell_matrix_)->size_in_bytes();
        size += static_cast<COOMatrix<IT, ST>*>(coo_matrix_)->size_in_bytes();

        size += sizeof(ell_size_);
        size += 2*sizeof(void*);        // 2 class references

        return size;
    }

};