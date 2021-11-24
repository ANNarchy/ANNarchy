#===============================================================================
#
#     BSR.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
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
        w = init_matrix_variable(%(init)s);
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

###############################################################
# Rate-coded continuous transmission (default implementation)
###############################################################
continuous_transmission = {
    'sum' : """
    %(idx_type)s row_ptr_size = block_row_size();
    %(idx_type)s row_nr_off = 0;
    %(idx_type)s row_max = this->get_num_rows();
    %(idx_type)s tile_size = get_tile_size();
    %(idx_type)s tile_size2 = tile_size*tile_size;
    auto block_ptr = block_row_pointer();
    auto block_col_idx = block_column_index();

    %(float_prec)s* __restrict__ target_ptr = pop%(id_post)s._sum_%(target)s.data();
    %(float_prec)s* __restrict__ pre_r = pop%(id_pre)s.r.data();

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
        %(idx_type)s row_max = this->get_num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = pop%(id_post)s._sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = pop%(id_pre)s.r.data();

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
        %(idx_type)s row_max = this->get_num_rows();
        %(idx_type)s tile_size = get_tile_size();
        %(idx_type)s tile_size2 = tile_size*tile_size;
        auto block_ptr = block_row_pointer();
        auto block_col_idx = block_column_index();

        %(float_prec)s* __restrict__ target_ptr = pop%(id_post)s._sum_%(target)s.data();
        %(float_prec)s* __restrict__ pre_r = pop%(id_pre)s.r.data();

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

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables = {
    'local': ""
}

conn_templates = {
    # accessors
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
            #'sse', 'avx'
        },
        3: {
            'none': {
                'multi_w': continuous_transmission_unroll_3x3,
            }
            #'sse','avx'
        }
    },
    'update_variables': update_variables
}

conn_ids = {
    'local_index': '',
    'pre_index': '',
    'post_index': '',
}