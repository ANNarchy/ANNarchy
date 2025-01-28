"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

copy_proj_template = {
    # Declare the connectivity matrix
    'declare_connectivity_matrix': "",
    # Accessors for the connectivity matrix
    'access_connectivity_matrix': "",
    # No initiaization of the connectivity matrix
    'init_connectivity_matrix': "",
    # No need for exporting access
    'declare_parameters_variables': "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;""",

    # Export the connectivity matrix
    'export_connectivity': "",

    # Initialize the wrapper connectivity matrix
    'wrapper': """
    // Copy ProjStruct%(id_proj)s
    nanobind::class_<ProjStruct%(id_proj)s>(m, "proj%(id_proj)s_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct%(id_proj)s::_transmission)
        .def_rw("_axon_transmission", &ProjStruct%(id_proj)s::_axon_transmission)
        .def_rw("_update", &ProjStruct%(id_proj)s::_update)
        .def_rw("_plasticity", &ProjStruct%(id_proj)s::_plasticity)

        // Other methods
        .def("clear", &ProjStruct%(id_proj)s::clear);    
    
    """,
}

copy_sum_template = {
    'sum': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s->_active){
        %(omp_code)s
        for(int i = 0; i < proj%(id)s->post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < proj%(id)s->pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s->_sum_%(target)s[proj%(id)s->post_rank[i]] += sum;
        }
    }
""",
    'max': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s->_active){
        %(omp_code)s
        for(int i = 0; i < proj%(id)s->post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < proj%(id)s->pre_rank[i].size(); j++){
                if(%(psp)s > sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s->_sum_%(target)s[proj%(id)s->post_rank[i]] += sum;
        }
    }
""",
    'min': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s->_active){
        %(omp_code)s
        for(int i = 0; i < proj%(id)s->post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < proj%(id)s->pre_rank[i].size(); j++){
                if(%(psp)s < sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s->_sum_%(target)s[proj%(id)s->post_rank[i]] += sum;
        }
    }
""",
    'mean': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s->_active){
        %(omp_code)s
        for(int i = 0; i < proj%(id)s->post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < proj%(id)s->pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s->_sum_%(target)s[proj%(id)s->post_rank[i]] += sum/ (%(float_prec)s)(proj%(id)s->pre_rank[i].size());
        }
    }
"""
}
