#===============================================================================
#
#     Dense.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021-22  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#                            Julien Vitay <julien.vitay@gmail.com>
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
        if(%(name)s.empty())
            return false;
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
        // Global %(attr_type)s %(name)s
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

delay = {
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;""",
        
        'pyx_struct':
"""
        # Uniform delay
        int delay""",
        'init': """
    delay = delays[0][0];
""",
        'pyx_wrapper_init':
"""
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
    }
}

spiking_summation_fixed_delay = """// Event-based summation
if (_transmission && %(post_prefix)s_active){

    // Iterate over all spiking neurons
    for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
        %(size_type)s beg = (*it) * this->num_rows_;
        %(size_type)s end = (*it+1) * this->num_rows_;

        // Post-synaptic potential
        %(idx_type)s rk_post = 0;
        for (%(size_type)s j = beg; j < end; j++, rk_post++) {
            %(g_target)s
        }

        // 'on-pre' events
        rk_post = 0;
        for (%(size_type)s j = beg; j < end; j++, rk_post++) {
            if (mask_[j]) {
                %(event_driven)s
                %(pre_event)s
            }
        }
    }
} // active
"""

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,

    #operations
    'rate_coded_sum': "",
    'vectorized_default_psp': {},
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'update_variables': "",
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[rk_post]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}