"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

pop_struct_wrapper = """// PopStruct%(id)s
    nanobind::class_<PopStruct%(id)s>(m, "pop%(id)s_wrapper")
        // init
        .def(nanobind::init<>())
        // attributes
        .def_rw("size", &PopStruct%(id)s::size)
        .def_rw("active", &PopStruct%(id)s::_active)
        .def_rw("max_delay", &PopStruct%(id)s::max_delay)
        //getter/setter
        .def("set_local_attribute_all_double", &PopStruct%(id)s::set_local_attribute_all_double)
        // others
        .def("reset", &PopStruct%(id)s::reset)
        .def("compute_firing_rate", &PopStruct%(id)s::compute_firing_rate)
        .def("clear", &PopStruct%(id)s::clear);
"""

pop_mon_wrapper = """// Monitor for Population %(id)s
    nanobind::class_<PopRecorder%(id)s>(m, "PopRecorder%(id)s_wrapper")
        // Record flag
%(record_flag)s
        // Target container
%(record_container)s
        // Clear container
%(clear_container)s
        // Functions
        .def(nanobind::init<std::vector<int>, int, int, long>())
        .def("clear", &PopRecorder%(id)s::clear)
        .def("size_in_bytes", &PopRecorder%(id)s::size_in_bytes);
"""