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
        %(name)s = init_matrix_variable< %(type)s, std::vector<%(type)s> >(%(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>( post_ranks_.size(), %(init)s);
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
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);
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
    delay = init_matrix_variable<int>(1);
    update_variable_all<int>(delay, delays);

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
    std::vector<std::vector<long>> _last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long, std::vector<long>>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

update_variables = {
    'local': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s

    auto row_ptr = sub_matrices_[tid]->row_ptr();
    auto col_idx = sub_matrices_[tid]->row_indices();
    auto post_ranks = sub_matrices_[tid]->get_post_rank();
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks.size());

    for(int i = 0; i < nb_post; i++){
        rk_post = post_ranks[i];
    %(semiglobal)s
        for(int j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++){
            rk_pre = col_idx[j];
    %(local)s
        }
    }
}
""",
        'global': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)) {
    %(global)s

    auto post_ranks = sub_matrices_[tid]->get_post_rank();
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks.size());

    for(%(idx_type)s i = 0; i < nb_post; i++){
        rk_post = post_ranks[i];
    %(semiglobal)s
    }
}
"""
}

spiking_summation_fixed_delay_outer_loop = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {
    auto _col_ptr = sub_matrices_[tid]->col_ptr();
    auto _row_idx = sub_matrices_[tid]->row_indices();
    auto _inv_idx = sub_matrices_[tid]->inverse_indices();

    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        int _pre = %(pre_array)s[_idx];

        // Iterate over connected post neurons
        for (int syn = _col_ptr[_pre]; syn < _col_ptr[_pre + 1]; syn++) {
            %(event_driven)s
            %(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_post_event = """
// w as CSR
auto row_ptr = sub_matrices_[tid]->row_ptr();
auto _col_idx = sub_matrices_[tid]->column_indices();

if(_transmission && %(post_prefix)s_active){
    #pragma omp for
    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = %(post_prefix)sspiked[_idx_i];

        // Iterate over all synapse to this neuron
        for (int j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++) {

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
    'rate_coded_sum': None,
    'vectorized_default_psp': None,
    'spiking_sum_fixed_delay': {
        'outer_loop': spiking_summation_fixed_delay_outer_loop
    },
    'spiking_sum_variable_delay': None,
    'update_variables': update_variables,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[tid][j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'pre_index': '[col_idx[j]]',
    'post_index': '[post_ranks_[i]]',
    'delay_nu' : '[delay[j]-1]', # non-uniform delay
    'delay_u' : '[delay-1]' # uniform delay
}
