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
if (_transmission && %(post_prefix)s_active) {
    const int post_slice_beg_ = post_slices_[tid];
    const int post_slice_end_ = post_slices_[tid+1];
#ifdef _DEBUG_OMP_DETAIL
    #pragma omp critical
    {
        std::cout << "thread " << tid << ": " << post_slice_beg_ << " - " << post_slice_end_ << std::endl;
    }
#endif

    // Iterate over all spiking neurons
    for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
        %(idx_type)s rk_pre = (*it);
        %(size_type)s beg = rk_pre * this->num_rows_;
        %(size_type)s end = (rk_pre+1) * this->num_rows_;

        // Iterate over columns
        for (%(idx_type)s rk_post = post_slice_beg_; rk_post < post_slice_end_; rk_post++) {
            %(size_type)s j = beg + rk_post;
            
            %(g_target)s

            if (mask_[j]) {
                %(event_driven)s
                %(pre_event)s
            }
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

    // Local variables
    #pragma omp for
    for(%(idx_type)s i = 0; i < %(post_prefix)ssize; i++){
        rk_post = i; // dense: ranks are indices

        // Semi-global variables
    %(semiglobal)s

        // Local variables are updated to boolean flag
        %(size_type)s j = i*%(pre_prefix)ssize;
        for(rk_pre = 0; rk_pre < %(pre_prefix)ssize; rk_pre++, j++) {
            if(mask_[j]) {
%(local)s
            }
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
    #pragma omp for
    for(int i = 0; i < %(post_prefix)ssize; i++){
        rk_post = i;
    %(semiglobal)s
    }
}
"""
}

spiking_post_event = """
if (_transmission && %(post_prefix)s_active) {

    %(idx_type)s rows = pop%(id_pre)s.size;
    %(idx_type)s columns = pop%(id_post)s.size;

    for (%(idx_type)s _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++) {
        %(idx_type)s post_rank = %(post_prefix)sspiked[_idx_i];

        for (%(size_type)s j = post_rank; j < this->num_rows_ * this->num_columns_; j += this->num_rows_) {
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

    #operations
    'rate_coded_sum': "",
    'vectorized_default_psp': {},
    'spiking_sum_fixed_delay': {
        # HD: need to distinguish ????
        'inner_loop': spiking_summation_fixed_delay,
        'outer_loop': spiking_summation_fixed_delay
    },
    'update_variables': dense_update_variables,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}
