attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s >> %(name)s;
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
        // Global
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
    },
    'nonuniform_rate_coded': {
    },
    'nonuniform_spiking': {
    }
}

continuous_transmission = {
    'sum': """
%(pre_copy)s

%(idx_type)s off, i, is, ie, j;
for (auto map_it = offsets_.begin(); map_it != offsets_.end(); map_it++) {

    off = map_it->first;
    is = std::max<int>(0, -off);
    ie = std::min<int>(num_columns_, num_columns_-off);

    %(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();

    %(omp_simd)s
    for(i=is; i < ie; i++) {
        target_ptr%(post_index)s += %(psp)s;
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

    'rate_coded_sum': continuous_transmission,
    'vectorized_default_psp': {},
    'update_variables': ""
}

conn_ids = {
    'local_index': '[map_it->second][i]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[i]',
    'pre_index': '[i+off]',
    'delay_u' : '[delay-1]', # uniform delay
    'delay_nu' : '[delay[j]-1]' # nonuniform delay
}
