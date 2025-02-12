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

#############################################
##  Synaptic delay
#############################################
delay = {
    # A single value for all synapses
    'uniform': None,
    # An individual value for each synapse
    'nonuniform_rate_coded': None,
    # An individual value for each synapse and a
    # buffer for spike events
    'nonuniform_spiking': None
}

###############################################################
# Rate-coded continuous transmission
###############################################################
coo_summation_operation = {
    'sum' : """
%(pre_copy)s

auto row_it = row_indices_.begin();
auto col_it = column_indices_.begin();
%(size_type)s nnz = nb_synapses();
%(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();

%(omp_code)s %(omp_clause)s %(omp_schedule)s
for(int j = 0; j < nnz; j++) {
    #pragma omp atomic
    target_ptr%(post_index)s += %(psp)s;
}
""",
    'max': "",
    'min': "",
    'mean': "",
}

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables = {
    'local': ""
}

conn_templates = {
    # accessors
    'delay': delay,
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    
    'rate_coded_sum': coo_summation_operation,
    'update_variables': update_variables
}

conn_ids = {
    'local_index': '[j]',
    'pre_index': '[*(col_it+j)]',
    'post_index': '[*(row_it+j)]',
}
