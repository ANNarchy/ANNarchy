"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

proj_struct_wrapper = """
    // ProjStruct%(id)s
    nanobind::class_<ProjStruct%(id)s>(m, "proj%(id)s_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct%(id)s::_transmission)
        .def_rw("_axon_transmission", &ProjStruct%(id)s::_axon_transmission)
        .def_rw("_update", &ProjStruct%(id)s::_update)
        .def_rw("_update_period", &ProjStruct%(id)s::_update_period)
        .def_rw("_update_offset", &ProjStruct%(id)s::_update_offset)
        .def_rw("_plasticity", &ProjStruct%(id)s::_plasticity)

        // Connectivity
        .def("init_from_lil", &ProjStruct%(id)s::init_from_lil)
        .def("post_rank", &ProjStruct%(id)s::get_post_rank)
        .def("dendrite_size", &ProjStruct%(id)s::dendrite_size)
        .def("nb_dendrites", &ProjStruct%(id)s::nb_dendrites)
        .def("pre_ranks", &ProjStruct%(id)s::get_pre_ranks)
        .def("pre_rank", &ProjStruct%(id)s::get_dendrite_pre_rank)
        .def("nb_synapses", &ProjStruct%(id)s::nb_synapses)

        // Methods
%(methods)s

        // Attributes
%(attributes)s

        // Other methods
%(additional)s
        .def("clear", &ProjStruct%(id)s::clear);
"""

proj_mon_wrapper = """
    // Monitor for Projection %(id)s
    nanobind::class_<ProjRecorder%(id)s>(m, "ProjRecorder%(id)s_wrapper")
        // Record flag
%(record_flag)s

        // Target container
%(record_container)s

        // Clear container
%(clear_container)s

        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &ProjRecorder%(id)s::clear)
        .def("size_in_bytes", &ProjRecorder%(id)s::size_in_bytes);
"""

proj_local_attr = """
        // local attributes
        .def("get_local_attribute_all_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_all_%(ctype)s)
        .def("get_local_attribute_row_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_row_%(ctype)s)
        .def("get_local_attribute_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_%(ctype)s)

        .def("set_local_attribute_all_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_all_%(ctype)s)
        .def("set_local_attribute_row_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_row_%(ctype)s)
        .def("set_local_attribute_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_%(ctype)s)
"""

proj_semiglobal_attr = """
        // semiglobal attributes
        .def("get_semiglobal_attribute_all_%(ctype)s", &ProjStruct%(id)s::get_semiglobal_attribute_all_%(ctype)s)
        .def("get_semiglobal_attribute_%(ctype)s", &ProjStruct%(id)s::get_semiglobal_attribute_%(ctype)s)

        .def("set_semiglobal_attribute_all_%(ctype)s", &ProjStruct%(id)s::set_semiglobal_attribute_all_%(ctype)s)
        .def("set_semiglobal_attribute_%(ctype)s", &ProjStruct%(id)s::set_semiglobal_attribute_%(ctype)s)
"""

proj_global_attr = """
        // global attributes
        .def("get_global_attribute_%(ctype)s", &ProjStruct%(id)s::get_global_attribute_%(ctype)s)
        .def("set_global_attribute_%(ctype)s", &ProjStruct%(id)s::set_global_attribute_%(ctype)s)
"""

proj_delays = {
    'uniform': """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)
""",
    'nonuniform_rate_coded': """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)

        .def("get_max_delay", &ProjStruct%(id)s::get_max_delay)
        .def("set_max_delay", &ProjStruct%(id)s::set_max_delay)
""",
    'nonuniform_spiking': """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)

        .def("get_max_delay", &ProjStruct%(id)s::get_max_delay)
        .def("set_max_delay", &ProjStruct%(id)s::set_max_delay)

        .def("update_max_delay", &ProjStruct%(id)s::update_max_delay)
        .def("reset_ring_buffer", &ProjStruct%(id)s::reset_ring_buffer)
"""
}