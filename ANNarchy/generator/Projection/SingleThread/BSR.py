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
# Rate-coded continuous transmission
###############################################################
continuous_transmission = {
    'sum' : """
    %(idx_type)s row_ptr_size = block_row_size();
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

            %(float_prec)s* values = w.data() + blk_col_idx * tile_size2;       // find the correct dense tile
            %(float_prec)s* loc_pr = pre_r + bcol_idx * tile_size;              // select the correct part in pre vector

            // process the row by row
            for (%(idx_type)s r = 0; r < tile_size; r++) {
                %(float_prec)s sum = 0.0;
                %(idx_type)s row_off = r*tile_size;

                // process all columns in the row
                for (%(idx_type)s c = 0; c < tile_size; c++) {
                    sum += values[row_off+c] * loc_pr[c];
                }
                loc_psp[r] += sum;
            }
        }
    }
""",
    'max': "",
    'min': "",
    'mean': "",
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
    'update_variables': update_variables
}

conn_ids = {
    'local_index': '',
    'pre_index': '',
    'post_index': '',
}