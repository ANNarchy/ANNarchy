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

        // getter/setter
%(attributes)s

        // others
        .def("clear", &ProjStruct%(id)s::clear);
"""
