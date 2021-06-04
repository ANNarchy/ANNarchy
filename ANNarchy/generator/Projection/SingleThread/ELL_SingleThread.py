#===============================================================================
#
#     ELL_SingleThread.py
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

#############################################
##  Synaptic delay
#############################################
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
"""},
    'nonuniform_rate_coded': {
        'declare': """
    std::vector<int> delay;
    int max_delay;

    std::vector<std::vector<int>> get_delay() { return get_matrix_variable_all<int>(delay); }
    void set_delay(std::vector<std::vector<int>> value) { update_matrix_variable_all<int>(delay, value); }
    std::vector<int> get_dendrite_delay(int lil_idx) { return get_matrix_variable_row<int>(delay, lil_idx); }
""",
        'init': """
    delay = init_variable<int>(1);
    update_variable_all<int>(delay, delays);
""",
        'reset': "",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] get_delay()
        void set_delay(vector[vector[int]])
        vector[int] get_dendrite_delay(int)
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.get_delay()
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.get_dendrite_delay(idx)
    def set_delay(self, value):
        proj%(id_proj)s.set_delay(value)
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    },
    'nonuniform_spiking': {
        'declare': """
    std::vector<int> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes;
""",
        'init': """
    delay = init_variable<int>(1);
    update_variable_all<int>(delay, delays);

    idx_delay = 0;
    max_delay = pop%(id_pre)s.max_delay;
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay =  pop%(id_pre)s.max_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );        
""",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    }    
}

###############################################################
# Rate-coded continuous transmission
###############################################################
ell_summation_operation = {
    'sum' : """
%(pre_copy)s

for(int i = 0; i < post_ranks_.size(); i++) {
    rk_post = post_ranks_[i]; // Get postsynaptic rank

    sum = 0.0;
    for(int j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++) {
        rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}""",
    'max': "",
    'min': "",
    'mean': "",
}

ell_summation_operation_simd = {
    'sum' : """
%(pre_copy)s

for(int i = 0; i < post_ranks_.size(); i++) {
    rk_post = post_ranks_[i]; // Get postsynaptic rank
    sum = 0.0;

    double pre_r[%(simd_len)s];
    int j = i*maxnzr_;

    // SIMD partition
    for(; j + %(simd_len)s < i*maxnzr_+rl_[i]; j += %(simd_len)s) {
        int* __restrict__ col_idx_ptr = &col_idx_[j];
        double *__restrict__ w_ptr = &w[j];

        // pre-load uncoalesced values
        for(int j2 = 0; j2 < %(simd_len)s; j2++)
            pre_r[j2] = pop%(id_pre)s.r[col_idx_ptr[j2]];

        // sum up
        for(int j2 = 0; j2 < %(simd_len)s; j2++)
            sum += w_ptr[j2] * pre_r[j2];
    }

    // remainder loop
    for(; j < i*maxnzr_+rl_[i]; j++) {
        rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }

    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}""",
    'max': "",
    'min': "",
    'mean': "",
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

    // Local variables
    for(int i = 0; i < post_ranks_.size(); i++){
        rk_post = post_ranks_[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s
        // Local variables
        for(int j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++){
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
    'delay': delay,
    
    'rate_coded_sum': ell_summation_operation,
    'update_variables': update_variables
}