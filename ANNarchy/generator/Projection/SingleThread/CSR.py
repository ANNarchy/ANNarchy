#===============================================================================
#
#     CSR.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2020  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

event_driven = {
    'declare': """
    std::vector<long> _last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

csr_summation_operation = {
    'sum' : """
%(pre_copy)s

const %(size_type)s * __restrict__ row_ptr = row_begin_.data();
const %(idx_type)s * __restrict__ col_idx = col_idx_.data();

for(auto it = post_ranks_.cbegin(); it != post_ranks_.cend(); it++) {
    %(idx_type)s rk_post = *it;

    sum = 0.0;
    for(%(size_type)s j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++) {
        sum += %(psp)s;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
} 
""",
    'max': """
%(pre_copy)s

const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
const %(idx_type)s* __restrict__ col_idx = col_idx_.data();

for(auto it = post_ranks_.cbegin(); it != post_ranks_.cend(); it++) {
    %(idx_type)s rk_post = *it;

    %(size_type)s j = _row_ptr[rk_post];
    sum = %(psp)s ;
    for(j = _row_ptr[rk_post]+1; j < _row_ptr[rk_post+1]; j++){
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'min': """
%(pre_copy)s

const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
const %(idx_type)s* __restrict__ col_idx = col_idx_.data();

for(auto it = post_ranks_.cbegin(); it != post_ranks_.cend(); it++) {
    %(idx_type)s rk_post = *it;
    
    %(size_type)s j= _row_ptr[rk_post];
    sum = %(psp)s ;
    for(j = _row_ptr[rk_post]+1; j < _row_ptr[rk_post+1]; j++){
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'mean': """
%(pre_copy)s

const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
const %(idx_type)s* __restrict__ col_idx = col_idx_.data();

for(auto it = post_ranks_.cbegin(); it != post_ranks_.cend(); it++) {
    %(idx_type)s rk_post = *it;

    sum = 0.0 ;
    for(%(size_type)s j = _row_ptr[rk_post]; j < _row_ptr[rk_post+1]; j++){
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum / static_cast<%(float_prec)s>(pre_rank[i].size());
}
"""
}

update_variables = {
    'local': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    // global variables
    %(global)s

    const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
    const %(idx_type)s* __restrict__ col_idx = col_idx_.data();

    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
    for (%(idx_type)s i = 0; i < nb_post; i++) {
        rk_post = post_ranks_[i];

        // semiglobal variables
    %(semiglobal)s
    
        // local variables
        for(%(size_type)s j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++){
            rk_pre = col_idx[j];
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s
    
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
    for (%(idx_type)s i = 0; i < nb_post; i++) {
        %(idx_type)s rk_post = post_ranks_[i];

    %(semiglobal)s
    }
}
"""
}

spiking_summation_fixed_delay_csr = """// Event-based summation
if (_transmission && pop%(id_post)s._active){
    // w as CSR
    const int * __restrict__ col_ptr = _col_ptr.data();

    // Iterate over all spiking neurons
    for (int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        int _pre = %(pre_array)s[_idx];

        // slice in CSRC
        int beg = col_ptr[_pre];
        int end = col_ptr[_pre+1];

        // Iterate over connected post neurons
        for (int syn = beg; syn < end; syn++) {
            %(event_driven)s
            %(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_post_event = """
// w as CSR
const int * __restrict__ row_ptr = row_begin_.data();

if(_transmission && pop%(id_post)s._active){
    for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = pop%(id_post)s.spiked[_idx_i];

        // slice in CSRC
        int beg = row_ptr[rk_post];
        int end = row_ptr[rk_post+1];

        // Iterate over all synapse to this neuron
        for (int j = beg; j < end; j++) {
%(event_driven)s
%(post_event)s
        }
    }
}
"""

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,
    'event_driven': event_driven,

    # operations
    'rate_coded_sum': csr_summation_operation,
    'update_variables': update_variables,
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay_csr,
    'spiking_sum_variable_delay': None,
    'post_event': spiking_post_event
}
