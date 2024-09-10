"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s> > %(name)s;
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
        %(name)s = init_matrix_variable< %(type)s, std::vector<%(type)s> >(%(init)s);
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
        size_in_bytes += sizeof(std::vector<std::vector<%(ctype)s>>);
        size_in_bytes += sizeof(std::vector<%(ctype)s>) * %(name)s.capacity();
        for(auto it = %(name)s.cbegin(); it != %(name)s.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(%(ctype)s);
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
        for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
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

update_variables = {
    'local': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s

    auto post_ranks = sub_matrices_[tid]->get_post_rank();
    auto col_ptr = sub_matrices_[tid]->col_ptr();
    auto row_idx = sub_matrices_[tid]->row_idx();
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks.size());

    for (int i = 0; i < nb_post; i++) {
        rk_post = post_ranks[i];
    %(semiglobal)s

        for(int j = col_ptr[rk_post]; j < col_ptr[rk_post+1]; j++){
            rk_pre = row_idx[j];
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

event_driven = {
    'declare': """
    std::vector<std::vector<long>> _last_event;
""",
    'cpp_init': """
        // Event-driven
        _last_event = init_matrix_variable<long, std::vector<long>>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

spiking_summation_fixed_delay = """// Event-based summation
if (_transmission && %(post_prefix)s_active){
    auto row_ptr_ = sub_matrices_[tid]->row_ptr();
    auto col_idx_ = sub_matrices_[tid]->col_idx();

    // Iterate over all spiking neurons
    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        // Rank of the presynaptic neuron
        int _pre = %(pre_array)s[_idx];

        // Iterate over connected post neurons
        for(int syn = row_ptr_[_pre]; syn < row_ptr_[_pre + 1]; syn++) {

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
if (_transmission && %(post_prefix)s_active) {
    auto col_ptr_ = sub_matrices_[tid]->col_ptr();
    auto row_idx_ = sub_matrices_[tid]->row_idx();
    auto inv_idx_ = sub_matrices_[tid]->inverse_indices();
    int part_beg = tid *  chunk_size_;
    int part_end = (tid+1) *  chunk_size_;

    #pragma omp for
    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++) {
        // Rank of the postsynaptic neuron which fired
        rk_post = %(post_prefix)sspiked[_idx_i];

        // not in the partition of the thread
        if (rk_post < part_beg)
            continue;
        if (rk_post >= part_end)
            continue;

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
    'spiking_sum_fixed_delay': {
        'outer_loop': spiking_summation_fixed_delay,
    },
    'spiking_sum_variable_delay': None,
    'update_variables': update_variables,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[tid][j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[i]',
    'pre_index': '[row_idx_[j]]',
}
