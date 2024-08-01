"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

basetemplate = """#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include "ANNarchy.hpp"

NB_MODULE(ANNarchyCore%(net_id)s, m) {
// global functions
    m.def("set_number_threads", &setNumberThreads);
    m.def("set_seed", &setSeed);
    m.def("pyx_create", &create_cpp_instances);
    m.def("pyx_initialize", &initialize);
    m.def("run", &run);

%(pop_struct_wrapper)s
%(proj_struct_wrapper)s
%(pop_mon_wrapper)s
}
"""
