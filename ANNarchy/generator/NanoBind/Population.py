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
        .def("size_in_bytes", &PopStruct%(id)s::size_in_bytes)
        .def("clear", &PopStruct%(id)s::clear);
"""

pop_attr_accessor_global = """
.def_prop_rw(
    "%(name)s",
    [](const PopStruct%(id)s& obj) {
        return obj.%(name)s;
    },
    [](PopStruct%(id)s& obj, %(py_float_prec)s value) {
        obj.%(name)s = static_cast<%(cpp_float_prec)s>(value);
    }
)"""

pop_attr_accessor_local = """
.def_prop_rw(
    "%(name)s",
    [](const PopStruct%(id)s& obj) {
        std::vector<%(py_float_prec)s> res;
        res.reserve(obj.%(name)s.size());
        for (%(cpp_float_prec)s tmp : obj.%(name)s)
            res.push_back(%(py_float_prec)s{tmp});
        return res;
    },
    [](PopStruct%(id)s& obj, std::vector<%(py_float_prec)s> values) {
        assert(obj.%(name)s.size() == values.size());

        for (size_t i = 0; i < obj.%(name)s.size(); ++i)
            obj.%(name)s[i] = %(cpp_float_prec)s{values[i]};
    }
)"""

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
