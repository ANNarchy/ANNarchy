#===============================================================================
#
#     CSR_T.py
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
        %(name)s = init_matrix_variable< %(type)s >(%(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable< %(type)s >(%(init)s);
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
        // Event-driven
        _last_event = init_matrix_variable<long>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

csr_summation_operation = {
    'sum' : """
%(pre_copy)s

for(int i = 0; i < row_ptr_.size()-1; i++) {
    sum = 0.0;
    for(int j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
"""
}

update_variables = {
    'local': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s

    int nb_post = static_cast<int>(post_ranks_.size());
    for (int i = 0; i < nb_post; i++) {
        rk_post = post_ranks_[i];

        // semiglobal variables
    %(semiglobal)s

        // local variables
        for (int j = col_ptr_[rk_post]; j < col_ptr_[rk_post+1]; j++) {
            rk_pre = row_idx_[j];
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s

    // semiglobal variables
    for(int i = 0; i < post_ranks.size(); i++){
        rk_post = post_ranks[i];
    %(semiglobal)s
    }
}
"""
}

spiking_summation_fixed_delay = """// Event-based summation
if (_transmission && pop%(id_post)s._active){

    // Iterate over all spiking neurons
    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        // Rank of the presynaptic neuron
        int _pre = %(pre_array)s[_idx];

        // slice in CSRC
        int beg = row_ptr_[_pre];
        int end = row_ptr_[_pre+1];

        // Iterate over connected post neurons
        for (int syn = beg; syn < end; syn++) {

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""

spiking_post_event =  """
if(_transmission && pop%(id_post)s._active){
    for (int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++) {
        // Rank of the postsynaptic neuron which fired
        rk_post = post_ranks_[pop%(id_post)s.spiked[_idx_i]];

        // slice in CSRC
        int beg = col_ptr_[rk_post];
        int end = col_ptr_[rk_post+1];

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
    'delay': delay,
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'event_driven': event_driven,

    #operations
    'update_variables': update_variables,
    'rate_coded_sum': csr_summation_operation,
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'spiking_sum_variable_delay': None,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[inv_idx_[j]]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'pre_index': '[row_idx_[j]]',
    'post_index': '[i]',
}