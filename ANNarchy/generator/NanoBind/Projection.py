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
%(connectivity)s

        // Methods
%(methods)s

        // Attributes
%(attributes)s

        // Other methods
%(additional)s
        .def("size_in_bytes", &ProjStruct%(id)s::size_in_bytes)
        .def("clear", &ProjStruct%(id)s::clear);
"""

proj_lil_connectivity = """
        .def("init_from_lil", []( const ProjStruct%(id)s& obj,
                                  nanobind::list nb_row_indices,
                                  nanobind::list nb_column_indices,
                                  nanobind::list nb_values,
                                  nanobind::list nb_delays,
                                  bool requires_sorting) {
                /*
                 *  Transform packed data containers into STL
                 */
                auto cpp_row_indices = nanobind::cast<std::vector<int>>(nb_row_indices);

                auto cpp_column_indices = nanobind::cast<std::vector<std::vector<int>>>(nb_column_indices);

                std::vector<std::vector<%(cpp_float_prec)s>> cpp_values;
                if constexpr (std::is_same_v<%(py_float_prec)s, %(cpp_float_prec)s>) {
                    // same type can be directly copied/casted
                    cpp_values = nanobind::cast<std::vector<std::vector<%(cpp_float_prec)s>>>(nb_values);
                } else {
                    // direct cast from double to lower precision types like float/fp16/bf16 ends up in
                    // narrow-down warning or even std::bad_cast. Therefore, we need interim vector which
                    // implies a conversion call
                    for (size_t i = 0; i < nb_values.size(); ++i) {
                        // convert python -> c++ without changing size
                        auto tmp_arr = nanobind::cast<std::vector<%(py_float_prec)s>>(nb_values[i]);
                        // convert to low precision type
                        auto conv_arr = std::vector<%(cpp_float_prec)s>(tmp_arr.begin(), tmp_arr.end());
                        // store data
                        cpp_values.emplace_back(conv_arr.data(), conv_arr.data() + conv_arr.size());
                    }
                }

                auto cpp_delays = nanobind::cast<std::vector<std::vector<int>>>(nb_delays);

                // perform initialization ...
                return proj%(id)s->init_from_lil(cpp_row_indices, cpp_column_indices, cpp_values, cpp_delays, requires_sorting);
            })
        /* HD (18th Aug. 2025):  The C++ template library offers in some cases a const- and non-const accessor.
         *                       To ensure that Python accesses only using the non-const accessor an additional
         *                       "nanobind::overload_cast<>" is needed. Otherwise, its compiler dependent which
         *                       version is bound consequently resulting in strange side-effects ...
         */
        .def("post_rank", nanobind::overload_cast<>(&ProjStruct%(id)s::get_post_rank))
        .def("dendrite_size", &ProjStruct%(id)s::dendrite_size)
        .def("nb_dendrites", &ProjStruct%(id)s::nb_dendrites)
        .def("pre_ranks", &ProjStruct%(id)s::get_pre_ranks)
        .def("pre_rank", &ProjStruct%(id)s::get_dendrite_pre_rank)
        .def("nb_synapses", &ProjStruct%(id)s::nb_synapses)
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
    "uniform": """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)
""",
    "nonuniform_rate_coded": """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)

        .def("get_max_delay", &ProjStruct%(id)s::get_max_delay)
        .def("set_max_delay", &ProjStruct%(id)s::set_max_delay)
""",
    "nonuniform_spiking": """
        // Synaptic delays
        .def("get_delay", &ProjStruct%(id)s::get_delay)
        .def("get_dendrite_delay", &ProjStruct%(id)s::get_dendrite_delay)
        .def("set_delay", &ProjStruct%(id)s::set_delay)

        .def("get_max_delay", &ProjStruct%(id)s::get_max_delay)
        .def("set_max_delay", &ProjStruct%(id)s::set_max_delay)

        .def("update_max_delay", &ProjStruct%(id)s::update_max_delay)
        .def("reset_ring_buffer", &ProjStruct%(id)s::reset_ring_buffer)
""",
}
