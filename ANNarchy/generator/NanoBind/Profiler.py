"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

profiler_template = """
    // Simulation kernel profiling
    nanobind::class_<Profiling>(m, "Profiling_wrapper")
        // Constructor
        .def(nanobind::init<>())
        .def("get_avg_time", &Profiling::get_avg_time)
        .def("get_std_time", &Profiling::get_std_time);
"""