"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

basetemplate = """#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/bind_vector.h>

#include "ANNarchy.hpp"

NB_MODULE(ANNarchyCore%(net_id)s, m) {

    // Global functions
    m.def("set_seed", &setSeed);
    m.def("pyx_create", &create_cpp_instances);
    m.def("pyx_initialize", &initialize);
    m.def("run", &run);
    m.def("run_until", &run_until);
    m.def("step", &step);
    m.def("set_time", &setTime);
    m.def("get_time", &getTime);

    // Target device specific
%(device_specific)s

    // Simulation-related objects
%(functions_wrapper)s

%(constant_wrapper)s

%(pop_struct_wrapper)s

%(proj_struct_wrapper)s

%(pop_mon_wrapper)s

%(proj_mon_wrapper)s

%(profiling_wrapper)s
}
"""
