#===============================================================================
#
#     SELLR.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#                         Qi Tang <kevin2014tq@gmail.com>
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
    std::vector< %(type)s > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
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
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global
        size_in_bytes += sizeof(%(ctype)s);
"""
}

attribute_cpp_delete = {
    'local': """
        // %(name)s
        %(name)s.clear();
        %(name)s.shrink_to_fit();
""",
    'semiglobal': """
        // %(name)s
        %(name)s.clear();
        %(name)s.shrink_to_fit();
""",
    'global': ""
}

###############################################################
# Rate-coded continuous transmission
###############################################################
continuous_transmission = {
    'sum' : """
// iterate across all blocks
for (%(idx_type)s i = 0; i < num_blocks_; i++) {
    
    //compute maxlength (maxnzr) in each block
    %(size_type)s maxlength = (row_ptr_[i + 1] - row_ptr_[i]) / block_size_;

    // iterate over all values within the block
    for (%(idx_type)s j = 0; j < block_size_; j++) {
        %(size_type)s pos = row_ptr_[i] + j * maxlength;
        //row_now: compute global row idx
        %(size_type)s row_now = i * block_size_ + j;

        if (row_now < num_rows_) {
            %(float_prec)s sum = 0.0;
            for (int k = 0; k < maxlength; k++) {
                %(idx_type)s rk_pre = col_idx_[pos+k];

                sum += %(psp)s;
            }
            pop%(id_post)s._sum_%(target)s[row_now] += sum;
        }
        
    }
}
""",
    'max': "",
    'min': "",
    'mean': "",
}

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': None,
    
    # operations
    'rate_coded_sum': continuous_transmission,
    'vectorized_default_psp': {},
    'update_variables': None
}

conn_ids = {
    'local_index': '[pos+k]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}