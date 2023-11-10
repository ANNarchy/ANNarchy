"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s ;
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
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    idx_delay = 0;
    max_delay = %(pre_prefix)smax_delay;
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay =  %(pre_prefix)smax_delay ;
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

%(omp_code)s
for(int i = 0; i < _col_ptr.size()-1; i++) {
    sum = 0.0;
    for(int j = _col_ptr[i]; j < _col_ptr[i+1]; j++) {
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
"""
}

update_variables = {
    'local': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s

    int nb_post = static_cast<int>(post_ranks_.size());
    #pragma omp for
    for (int i = 0; i < nb_post; i++) {
        rk_post = post_ranks_[i];
    %(semiglobal)s
        for(int j = col_ptr_[rk_post]; j < col_ptr_[rk_post+1]; j++){
            rk_pre = row_idx_[j];
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s

    int nb_post = static_cast<int>(post_ranks_.size());
    #pragma omp for
    for (int i = 0; i < nb_post; i++) {
        rk_post = post_ranks[i];
    %(semiglobal)s
    }
}
"""
}

spiking_summation_fixed_delay_inner_loop = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {
    // Iterate over all spiking neurons
    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        // Rank of the presynaptic neuron
        int _pre = %(pre_array)s[_idx];

        // slice in CSRC
        int beg = row_ptr_[_pre];
        int end = row_ptr_[_pre+1];

        // Iterate over connected post neurons
        #pragma omp for
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

spiking_summation_fixed_delay_outer_loop = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {
    // Iterate over all spiking neurons
    for (int _idx = tid; _idx < %(pre_array)s.size(); _idx += nt) {
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
            #pragma omp atomic%(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""

spiking_post_event =  """
if(_transmission && %(post_prefix)s_active){
    #pragma omp for
    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = post_ranks_[%(post_prefix)sspiked[_idx_i]];

        // Iterate over all synapse to this neuron
        for(int j = col_ptr_[rk_post]; j < col_ptr_[rk_post+1]; j++){
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
    'rate_coded_sum': csr_summation_operation,
    'spiking_sum_fixed_delay': {
        'inner_loop': spiking_summation_fixed_delay_inner_loop,
        'outer_loop': spiking_summation_fixed_delay_outer_loop
    },
    'spiking_sum_variable_delay': None,
    'update_variables': update_variables,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[i]',
    'pre_index': '[row_idx_[j]]',
}
