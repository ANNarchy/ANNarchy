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
    # A single value for all synapses
    'uniform': {
        'declare': """
    // Uniform delay
    int delay;

    int get_delay() { return delay; }
    int get_dendrite_delay(int idx) { return delay; }
    void set_delay(int delay) { this->delay = delay; }
""",
        'init': """
    delay = delays[0][0];
"""
    },
    # An individual value for each synapse
    'nonuniform_rate_coded': None,
    # An individual value for each synapse and a
    # buffer for spike events
    'nonuniform_spiking': None
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
"""
}

spiking_summation_fixed_delay = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    // Iterate over all spiking neurons
    for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
        %(idx_type)s rk_pre = *it;
        %(size_type)s beg = rk_pre * this->num_rows_;
        %(size_type)s end = (rk_pre+1) * this->num_rows_;

        // 'on-pre' events
        %(idx_type)s rk_post = 0;
        for (%(size_type)s j = beg; j < end; j++, rk_post++) {
            if (mask_[j]) {
                %(event_driven)s
                %(g_target)s
                %(pre_event)s
            }
        }
    }
} // active
"""

# Specialized variant of the above kernel. By default, w is initialized with 0.0 which is neutral in the
# psp operation (such as g_target += w).
spiking_summation_fixed_delay_only_psp = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    // Iterate over all spiking neurons
    for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
        %(idx_type)s rk_pre = *it;
        %(size_type)s beg = rk_pre * this->num_rows_;
        %(size_type)s end = (rk_pre+1) * this->num_rows_;

        %(idx_type)s rk_post = 0;
        for (%(size_type)s j = beg; j < end; j++, rk_post++) {
            %(g_target)s
        }
    }
} // active
"""

dense_update_variables = {
    'local': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s

    // Semi-global variables
    for(rk_post = 0; rk_post < %(post_prefix)ssize; rk_post++) {
        %(semiglobal)s
    }

    // Local variables
    for (rk_pre = 0; rk_pre < %(pre_prefix)ssize; rk_pre++) {
        rk_post = 0;

        %(size_type)s beg = rk_pre*%(post_prefix)ssize;
        %(size_type)s end = (rk_pre+1)*%(post_prefix)ssize;

        // Local variables are updated to boolean flag
        #pragma omp simd
        for (%(size_type)s j = beg; j < end; j++) {
%(local)s
            rk_post++;
        }
    }
}
""",
    'global': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s

    // Semi-global variables
    for(int i = 0; i < %(post_prefix)ssize; i++){
        rk_post = i;
    %(semiglobal)s
    }
}
"""
}

spiking_post_event = """
if (_transmission && %(post_prefix)s_active) {

    %(idx_type)s rows = pop%(id_pre)s->size;
    %(idx_type)s columns = pop%(id_post)s->size;

    for (%(idx_type)s _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++) {
        %(idx_type)s post_rank = %(post_prefix)sspiked[_idx_i];
        %(idx_type)s rk_pre = 0;

        for (%(size_type)s j = post_rank; j < this->num_rows_ * this->num_columns_; j += this->num_rows_, rk_pre++) {
            if(mask_[j]) {
%(event_driven)s
%(post_event)s
            }
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

    #operations
    'rate_coded_sum': None,
    'vectorized_default_psp': {},
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'spiking_sum_fixed_delay_only_psp': spiking_summation_fixed_delay_only_psp,
    'update_variables': dense_update_variables,
    'post_event': spiking_post_event,
    'event_driven': event_driven
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[rk_post]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}
