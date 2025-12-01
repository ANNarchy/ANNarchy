"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector<%(type)s> %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        w = init_matrix_variable<%(type)s>(%(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
"""
}

attribute_cpp_size = {
    'local': """
        // Local %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
""",
    'global': """
        // Global %(attr_type)s %(name)s
"""
}

attribute_cpp_delete = {
    'local': """
        // %(name)s
""",
    'semiglobal': """
        // %(name)s
""",
    'global': ""
}

#############################################
##  Synaptic delay
#############################################
delay = {
    # A single value for all synapses
    'uniform': None,
    # An individual value for each synapse
    'nonuniform_rate_coded': None,
    # An individual value for each synapse and a
    # buffer for spike events
    'nonuniform_spiking': None
}

###############################################################
# Rate-coded continuous transmission (default implementation)
###############################################################
continuous_transmission = {
    'sum' : """
    %(idx_type)s row_ptr_size = block_row_size();
    %(idx_type)s row_nr_off = 0;
    %(idx_type)s row_max = this->num_rows();
    %(idx_type)s tile_size = get_tile_size();
    %(idx_type)s tile_size2 = tile_size*tile_size;
    auto block_ptr = block_row_pointer();
    auto block_col_idx = block_column_index();

    %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();
    %(float_prec)s* __restrict__ pre_r = %(pre_prefix)sr.data();

    for (%(idx_type)s blk_row = 0; blk_row < row_ptr_size; blk_row++) {
        %(float_prec)s* loc_psp = target_ptr + blk_row * tile_size;

        for (%(idx_type)s blk_col_idx = block_ptr[blk_row]; blk_col_idx < block_ptr[blk_row+1]; blk_col_idx++) {
            %(idx_type)s bcol_idx = block_col_idx[blk_col_idx];     // which column in row

            %(float_prec)s* __restrict__ values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
            %(float_prec)s* __restrict__ loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

            // process the dense matrix row by row
            %(idx_type)s row_off = 0;
            %(idx_type)s r = 0;
            for (; r < tile_size; r++, row_off+=tile_size) {
                if (row_nr_off+r >= row_max)
                    continue;

                %(float_prec)s sum = 0.0;

                // process all columns in the row
                for (%(idx_type)s c = 0; c < tile_size; c++) {
                    sum += values[row_off+c] * loc_pr[c];
                }
                loc_psp[r] += sum;
            }
        }

        row_nr_off += tile_size;
    }
"""
}

# The inner dense SpMV is partially unrolled (2x2 kernel)
continuous_transmission_unroll_2x2 = {
    'sum' : """
        %(idx_type)s row_ptr_size = block_row_size();
        %(idx_type)s row_nr_off = 0;
        %(idx_type)s row_max = this->num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = %(pre_prefix)sr.data();

        for (%(idx_type)s blk_row = 0; blk_row < row_ptr_size; blk_row++) {
            %(float_prec)s* loc_psp = target_ptr + blk_row * tile_size;
            %(float_prec)s sum1=0.0;
            %(float_prec)s sum2=0.0;

            /* Unrolled inner 2x2 SpMV kernel */
            for (%(idx_type)s blk_col_idx = block_ptr[blk_row]; blk_col_idx < block_ptr[blk_row+1]; blk_col_idx++) {
                %(idx_type)s bcol_idx = block_col_idx[blk_col_idx];     // which column in row

                %(float_prec)s* __restrict__ values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

                sum1 += values[0] * loc_pr[0];
                sum1 += values[1] * loc_pr[1];

                sum2 += values[2] * loc_pr[0];
                sum2 += values[3] * loc_pr[1];
            }

            // prevent wrong write access
            if (row_nr_off < row_max)
                loc_psp[0] += sum1;
            if (row_nr_off+1 < row_max)
                loc_psp[1] += sum2;

            row_nr_off += tile_size;
        }
"""
}

# The inner dense SpMV is partially unrolled (3x3 kernel)
continuous_transmission_unroll_3x3 = {
    'sum' : """
        %(idx_type)s row_ptr_size = block_row_size();
        %(idx_type)s row_nr_off = 0;
        %(idx_type)s row_max = this->num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = %(pre_prefix)sr.data();

        for (%(idx_type)s blk_row = 0; blk_row < row_ptr_size; blk_row++) {
            %(float_prec)s* loc_psp = target_ptr + blk_row * tile_size;

            for (%(idx_type)s blk_col_idx = block_ptr[blk_row]; blk_col_idx < block_ptr[blk_row+1]; blk_col_idx++) {
                %(idx_type)s bcol_idx = block_col_idx[blk_col_idx];     // which column in row

                %(float_prec)s* __restrict__ values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

                %(float_prec)s sum1_1, sum1_2, sum1_3, sum2_1, sum2_2, sum2_3, sum3_1, sum3_2, sum3_3;
                sum1_1 = values[0] * loc_pr[0];
                sum1_2 = values[1] * loc_pr[1];
                sum1_3 = values[2] * loc_pr[2];

                sum2_1 = values[4] * loc_pr[0];
                sum2_2 = values[5] * loc_pr[1];
                sum2_3 = values[6] * loc_pr[2];

                sum3_1 = values[7] * loc_pr[0];
                sum3_2 = values[8] * loc_pr[1];
                sum3_3 = values[9] * loc_pr[2];

                if (row_nr_off < row_max)
                    loc_psp[0] += sum1_1 + sum1_2 + sum1_3;
                if (row_nr_off+1 < row_max)
                    loc_psp[1] += sum2_1 + sum2_2 + sum2_3;
                if (row_nr_off+2 < row_max)
                    loc_psp[2] += sum3_1 + sum3_2 + sum3_3;
            }

            row_nr_off += tile_size;
        }
"""
}

# The inner dense SpMV is partially unrolled (2x2 kernel)
# and the resulting code is vectorized.
continuous_transmission_unroll_2x2_avx512 = {
    'sum' : {
        'double': """
    #ifdef __AVX512F__
        %(idx_type)s row_ptr_size = block_row_size();
        %(idx_type)s row_nr_off = 0;
        %(idx_type)s row_max = this->num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = %(pre_prefix)sr.data();
        double _tmp_sum[8];

        // AVX512: 2 blocks times 4 elements each
        for (%(idx_type)s blk_row = 0; blk_row < row_ptr_size; blk_row++) {
            %(float_prec)s* loc_psp = target_ptr + blk_row * tile_size;

            __m512d _tmp_reg_sum = _mm512_setzero_pd();

            // block begin
            %(idx_type)s blk_col_idx = block_ptr[blk_row];

            /* Unrolled inner 2x2 SpMV kernel, process two blocks at once */
            for (; blk_col_idx+2 < block_ptr[blk_row+1]; blk_col_idx+=2) {
                %(idx_type)s bcol_idx_1 = block_col_idx[blk_col_idx];     // which column in row
                %(idx_type)s bcol_idx_2 = block_col_idx[blk_col_idx+1];     // which column in row

                %(float_prec)s* __restrict__ values_1 = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ values_2 = w.data() + (blk_col_idx+1) * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ loc_pr_1 = pre_r + bcol_idx_1 * tile_size;            // select the correct part in pre vector
                %(float_prec)s* __restrict__ loc_pr_2 = pre_r + bcol_idx_2 * tile_size;

                __m512d _tmp_r = _mm512_set_pd(
                    loc_pr_2[1], loc_pr_2[0], loc_pr_2[1], loc_pr_2[0],
                    loc_pr_1[1], loc_pr_1[0], loc_pr_1[1], loc_pr_1[0]
                );
                __m512d _tmp_w = _mm512_set_pd(
                    values_2[3], values_2[2], values_2[1], values_2[0],
                    values_1[3], values_1[2], values_1[1], values_1[0]
                );

                _tmp_reg_sum = _mm512_add_pd(_tmp_reg_sum, _mm512_mul_pd(_tmp_r, _tmp_w));
            }

            // remainder step ...
            for (; blk_col_idx < block_ptr[blk_row+1]; blk_col_idx++) {
                %(idx_type)s bcol_idx = block_col_idx[blk_col_idx];     // which column in row

                %(float_prec)s* __restrict__ values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

                __m512d _tmp_r = _mm512_set_pd(
                    0.0, 0.0, 0.0, 0.0,
                    loc_pr[1], loc_pr[0], loc_pr[1], loc_pr[0]
                );
                __m512d _tmp_w = _mm512_set_pd(
                    0.0, 0.0, 0.0, 0.0,
                    values[3], values[2], values[1], values[0]
                );

                _tmp_reg_sum = _mm512_add_pd(_tmp_reg_sum, _mm512_mul_pd(_tmp_r, _tmp_w));
            }

            // write back to memory
            _mm512_storeu_pd(_tmp_sum, _tmp_reg_sum);

            // prevent wrong write access
            if (row_nr_off < row_max)
                loc_psp[0] += _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[4] + _tmp_sum[5];
            if (row_nr_off+1 < row_max)
                loc_psp[1] += _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[6] + _tmp_sum[7];

            row_nr_off += tile_size;
        }
    #else
         std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
""",
    'float': """
    #ifdef __AVX512F__
        %(idx_type)s row_ptr_size = block_row_size();
        %(idx_type)s row_nr_off = 0;
        %(idx_type)s row_max = this->num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = %(pre_prefix)sr.data();
        float _tmp_sum[16];

        // AVX512: 4 blocks times 4 elements each
        for (%(idx_type)s blk_row = 0; blk_row < row_ptr_size; blk_row++) {
            %(float_prec)s* loc_psp = target_ptr + blk_row * tile_size;
            %(float_prec)s sum1 = 0.0;
            %(float_prec)s sum2 = 0.0;

            __m512 _tmp_reg_sum = _mm512_setzero_ps();

            // block begin
            %(idx_type)s blk_col_idx = block_ptr[blk_row];

            /* Unrolled inner 2x2 SpMV kernel, process four blocks at once */
            for (; blk_col_idx+4 < block_ptr[blk_row+1]; blk_col_idx += 4) {
                // which column in row
                %(idx_type)s bcol_idx_1 = block_col_idx[blk_col_idx];
                %(idx_type)s bcol_idx_2 = block_col_idx[blk_col_idx+1];
                %(idx_type)s bcol_idx_3 = block_col_idx[blk_col_idx+2];
                %(idx_type)s bcol_idx_4 = block_col_idx[blk_col_idx+3];

                // find the correct dense tile
                %(float_prec)s* __restrict__ values_1 = w.data() + blk_col_idx * tile_size2;
                %(float_prec)s* __restrict__ values_2 = w.data() + (blk_col_idx+1) * tile_size2;
                %(float_prec)s* __restrict__ values_3 = w.data() + (blk_col_idx+2) * tile_size2;
                %(float_prec)s* __restrict__ values_4 = w.data() + (blk_col_idx+3) * tile_size2;

                // select the correct part in pre vector
                %(float_prec)s* __restrict__ loc_pr_1 = pre_r + block_col_idx[blk_col_idx] * tile_size;
                %(float_prec)s* __restrict__ loc_pr_2 = pre_r + bcol_idx_2 * tile_size;
                %(float_prec)s* __restrict__ loc_pr_3 = pre_r + bcol_idx_3 * tile_size;
                %(float_prec)s* __restrict__ loc_pr_4 = pre_r + bcol_idx_4 * tile_size;

                __m512 _tmp_r = _mm512_set_ps(
                    loc_pr_4[1], loc_pr_4[0], loc_pr_4[1], loc_pr_4[0],
                    loc_pr_3[1], loc_pr_3[0], loc_pr_3[1], loc_pr_3[0],
                    loc_pr_2[1], loc_pr_2[0], loc_pr_2[1], loc_pr_2[0],
                    loc_pr_1[1], loc_pr_1[0], loc_pr_1[1], loc_pr_1[0]
                );
                __m512 _tmp_w = _mm512_set_ps(
                    values_4[3], values_4[2], values_4[1], values_4[0],
                    values_3[3], values_3[2], values_3[1], values_3[0],
                    values_2[3], values_2[2], values_2[1], values_2[0],
                    values_1[3], values_1[2], values_1[1], values_1[0]
                );

                _tmp_reg_sum = _mm512_add_ps(_tmp_reg_sum, _mm512_mul_ps(_tmp_r, _tmp_w));
            }

            // remainder step(s) ... (TODO: maybe switch to SSE4, might be better than filling the registers with nothing ...)
            for (; blk_col_idx < block_ptr[blk_row+1]; blk_col_idx++) {
                %(idx_type)s bcol_idx = block_col_idx[blk_col_idx];     // which column in row

                %(float_prec)s* __restrict__ values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
                %(float_prec)s* __restrict__ loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

                __m512 _tmp_r = _mm512_set_ps(
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    loc_pr[1], loc_pr[0], loc_pr[1], loc_pr[0]
                );
                __m512 _tmp_w = _mm512_loadu_ps(values);

                _tmp_reg_sum = _mm512_add_ps(_tmp_reg_sum, _mm512_mul_ps(_tmp_r, _tmp_w));
            }

            // write back to memory
            _mm512_storeu_ps(_tmp_sum, _tmp_reg_sum);

            // prevent wrong write access
            if (row_nr_off < row_max)
                loc_psp[0] += _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[8] + _tmp_sum[9] + _tmp_sum[12] + _tmp_sum[13];
            if (row_nr_off+1 < row_max)
                loc_psp[1] += _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[6] + _tmp_sum[7] + _tmp_sum[10] + _tmp_sum[11] + _tmp_sum[14] + _tmp_sum[15];

            row_nr_off += tile_size;
        }
    #else
         std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables = {
    'local': ""
}

###############################################################
# Event-driven updates
###############################################################
spiking_sum_fixed_delay = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    auto g_%(target)s_ext = std::vector<%(float_prec)s>( static_cast<int>(ceil(static_cast<%(float_prec)s>(%(post_prefix)ssize)/static_cast<%(float_prec)s>(this->tile_size_))*this->tile_size_));
    memcpy(g_%(target)s_ext.data(), %(post_prefix)sg_%(target)s.data(), %(post_prefix)ssize * sizeof(%(float_prec)s));

    // Iterate over all spiking neurons
    for (int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        int _pre = %(pre_array)s[_idx];

        %(idx_type)s block_column_idx = _pre / this->tile_size_;        // which column
        %(idx_type)s block_column_tile_idx = _pre %% this->tile_size_;  // position in the column

        for (%(idx_type)s blk_row_idx = block_column_pointer_[block_column_idx]; blk_row_idx < block_column_pointer_[block_column_idx+1]; blk_row_idx++) {
            %(idx_type)s tile_idx = block_inv_index_[blk_row_idx];
            %(idx_type)s row_idx = block_row_index_[blk_row_idx];

            // target determined by column
            %(float_prec)s* __restrict__ target_psp = g_%(target)s_ext.data() + row_idx * this->tile_size_;
            // dense tiles are stored in colum-major ordering
            %(float_prec)s* __restrict__ values = w.data() + tile_idx * this->tile_size_ * this->tile_size_ + block_column_tile_idx * this->tile_size_;

            for (%(idx_type)s r = 0; r < this->tile_size_; r++) {
                target_psp[r] += values[r];
            }
        }
    }

    memcpy(%(post_prefix)sg_%(target)s.data(), g_%(target)s_ext.data(), %(post_prefix)ssize * sizeof(%(float_prec)s));
}
"""

conn_templates = {
    # accessors
    'delay': delay,
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,

    'rate_coded_sum': continuous_transmission,
    # optimized kernels sorted by block size and SIMD
    'unrolled_default_psp': {
        2: {
            'none': {
                'multi_w': continuous_transmission_unroll_2x2
            },
            #'sse',
            #'avx',
            'avx512': {
                'multi_w': continuous_transmission_unroll_2x2_avx512
            }
        },
        3: {
            'none': {
                'multi_w': continuous_transmission_unroll_3x3,
            }
            #'sse','avx'
        }
    },
    'update_variables': update_variables,
    'spiking_sum_fixed_delay': spiking_sum_fixed_delay
}

conn_ids = {
    'local_index': '',
    'pre_index': '',
    'post_index': '',
}
