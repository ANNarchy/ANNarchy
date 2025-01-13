"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

proj_struct_wrapper = """// ProjStruct%(id)s
    nanobind::class_<ProjStruct%(id)s>(m, "proj%(id)s_wrapper")
        .def(nanobind::init<>())

        // flags
        .def_rw("_transmission", &ProjStruct%(id)s::_transmission)
        .def_rw("_axon_transmission", &ProjStruct%(id)s::_axon_transmission)
        .def_rw("_update", &ProjStruct%(id)s::_update)
        .def_rw("_plasticity", &ProjStruct%(id)s::_plasticity)

        // connectivity
        .def("init_from_lil", &ProjStruct%(id)s::init_from_lil)
        .def("post_rank", &ProjStruct%(id)s::get_post_rank)
        .def("dendrite_size", &ProjStruct%(id)s::dendrite_size)
        .def("pre_ranks", &ProjStruct%(id)s::get_pre_ranks)
        .def("pre_rank", &ProjStruct%(id)s::get_dendrite_pre_rank)

        // getter/setter
%(attributes)s

        // others
        .def("clear", &ProjStruct%(id)s::clear);
"""

proj_local_attr = """// local attributes
        .def("get_local_attribute_all_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_all_%(ctype)s)
        .def("get_local_attribute_row_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_row_%(ctype)s)
        .def("get_local_attribute_%(ctype)s", &ProjStruct%(id)s::get_local_attribute_%(ctype)s)

        .def("set_local_attribute_all_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_all_%(ctype)s)
        .def("set_local_attribute_row_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_row_%(ctype)s)
        .def("set_local_attribute_%(ctype)s", &ProjStruct%(id)s::set_local_attribute_%(ctype)s)
"""

proj_semiglobal_attr = """// semiglobal attributes
        .def("get_semiglobal_attribute_all_%(ctype)s", &ProjStruct%(id)s::get_semiglobal_attribute_all_%(ctype)s)
        .def("get_semiglobal_attribute_%(ctype)s", &ProjStruct%(id)s::get_semiglobal_attribute_%(ctype)s)

        .def("set_semiglobal_attribute_all_%(ctype)s", &ProjStruct%(id)s::set_semiglobal_attribute_all_%(ctype)s)
        .def("set_semiglobal_attribute_%(ctype)s", &ProjStruct%(id)s::set_semiglobal_attribute_%(ctype)s)
"""

proj_global_attr = """// global attributes
        .def("get_global_attribute_%(ctype)s", &ProjStruct%(id)s::get_global_attribute_%(ctype)s)
        .def("set_global_attribute_%(ctype)s", &ProjStruct%(id)s::set_global_attribute_%(ctype)s)
"""