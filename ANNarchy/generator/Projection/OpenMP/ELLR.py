#===============================================================================
#
#     ELLR.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

###############################################################
# Rate-coded continuous transmission
###############################################################
ellr_summation_operation = {
    'sum' : """
%(pre_copy)s

%(float_prec)s* __restrict__ target = pop%(id_post)s._sum_%(target)s.data();
%(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());

#pragma omp for firstprivate(maxnzr_)
for(%(idx_type)s i = 0; i < nb_post; i++) {
    rk_post = post_ranks_[i]; // Get postsynaptic rank

    sum = 0.0;
    for(%(size_type)s j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++) {
        rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }
    
    target%(post_index)s += sum;
}"""
}

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables = {
    'local': """
// Check periodicity
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    // Global variables
    %(global)s

    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
    // Local variables
    #pragma omp for
    for(%(idx_type)s i = 0; i < nb_post; i++){
        rk_post = post_ranks_[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s
        // Local variables
        for(%(size_type)s j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++){
            rk_pre = col_idx_[j]; // Get presynaptic rank
    %(local)s
        }
    }
}
"""
}

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,
    
    'rate_coded_sum': ellr_summation_operation,
    'update_variables': update_variables
}