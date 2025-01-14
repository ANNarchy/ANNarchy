"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

pop_struct_wrapper = """
    // PopStruct%(id)s
    nanobind::class_<PopStruct%(id)s>(m, "pop%(id)s_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct%(id)s::size)
        .def_rw("max_delay", &PopStruct%(id)s::max_delay)

        // Attributes
%(attributes)s

        // Other methods
%(additional)s
        .def("activate", &PopStruct%(id)s::set_active)
        .def("reset", &PopStruct%(id)s::reset)
        .def("clear", &PopStruct%(id)s::clear);
"""

pop_mon_wrapper = """
    // Monitor for Population %(id)s
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
