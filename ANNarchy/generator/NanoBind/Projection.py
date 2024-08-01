"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

proj_struct_wrapper = """// ProjStruct%(id)s
    nanobind::class_<ProjStruct%(id)s>(m, "proj%(id)s_wrapper")
        .def(nanobind::init<>())
        // init
        .def("init_from_lil", &ProjStruct%(id)s::init_from_lil)
        // getter/setter

        // others
        .def("clear", &ProjStruct%(id)s::clear);
"""
