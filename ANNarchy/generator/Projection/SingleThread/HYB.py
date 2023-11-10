"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    hyb_local<%(type)s>* %(name)s;
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector<%(type)s> %(name)s;
""",
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
"""
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
""",
    'global': """
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
}

attribute_cpp_size = {
    'local': """
        // Local %(attr_type)s %(name)s
        size_in_bytes += sizeof(hyb_local<%(ctype)s>);
        size_in_bytes += (%(name)s->ell.capacity()) * sizeof(%(ctype)s);
        size_in_bytes += (%(name)s->coo.capacity()) * sizeof(%(ctype)s);
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<%(ctype)s>());
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global
        size_in_bytes += sizeof(%(ctype)s);
"""
}

attribute_cpp_delete = {
    'local':  """
        // %(name)s
        %(name)s->clear();
""",
    'semiglobal': "",
    'global': "",
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
hyb_summation_operation = {
    'sum' : """
%(pre_copy)s

// ELLPACK partition
auto post_ranks = get_post_rank();
auto maxnzr_ = ell_matrix_->get_maxnzr();
auto col_idx_ = ell_matrix_->get_column_indices();
const %(idx_type)s nonvalue_idx = std::numeric_limits<%(idx_type)s>::max();

%(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks.size());
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s rk_post = post_ranks[i]; // Get postsynaptic rank

    sum = 0.0;
    %(size_type)s beg = i*maxnzr_;
    %(size_type)s end = (i+1)*maxnzr_;
    for (%(size_type)s j = beg; j < end; j++) {
        %(idx_type)s rk_pre = col_idx_[j];
        if (rk_pre == nonvalue_idx)
            break;

        sum += %(ell_psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(ell_post_index)s += sum;
}

// Coordinate partition
auto nnz = coo_matrix_->nb_synapses();
auto row_it = coo_matrix_->get_row_indices();
auto col_it = coo_matrix_->get_column_indices();

for(int j = 0; j < nnz; j++, row_it++, col_it++) {
    %(post_prefix)s_sum_%(target)s%(coo_post_index)s += %(coo_psp)s;
}
""",
    'max': "",
    'min': "",
    'mean': "",
}

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,
    
    'rate_coded_sum': hyb_summation_operation,
    'update_variables': ""
}
