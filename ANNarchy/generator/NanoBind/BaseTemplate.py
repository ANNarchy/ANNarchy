"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

basetemplate = """#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/ndarray.h>

#include "ANNarchyCore%(net_id)s.hpp"

NB_MODULE(ANNarchyWrapper%(net_id)s, m) {

    // Global functions
    m.def("set_seed", &setSeed);
    m.def("pyx_create", &create_cpp_instances);
    m.def("pyx_initialize", [](%(py_float_prec)s _dt) {
        %(cpp_float_prec)s _conv_dt = static_cast<%(cpp_float_prec)s>(_dt);
        initialize(_conv_dt);
    });
    m.def("estimate_record_size", &estimate_record_size);
    m.def("run", &run);
    m.def("run_until", &run_until);
    m.def("step", &step);
    m.def("set_time", &set_sim_step);
    m.def("get_time", &get_sim_step);
    m.def("get_sim_dt", []() -> %(py_float_prec)s {
        return static_cast<%(py_float_prec)s>(getDt());
    });

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

function_template = """
    m.def("%(name)s_wrapper", [](nanobind::object x) -> nanobind::object {

        // Single value?
        auto numpy = nanobind::module_::import_("numpy");

        if (nanobind::isinstance<nanobind::float_>(x) ||
            nanobind::isinstance<nanobind::int_>(x) ||
            nanobind::cast<bool>(numpy.attr("isscalar")(x))) {

            double value = nanobind::cast<double>(x);
            return nanobind::cast(%(name)s(value));
        }

        // Python-List
        if (nanobind::isinstance<nanobind::list>(x)) {

            std::vector<double> values =
                nanobind::cast<std::vector<double>>(x);

            for (auto &v : values)
                v = %(name)s(v);

            return nanobind::cast(values);
        }

        // NumPy-Array?
        if (nanobind::isinstance<nanobind::ndarray<>>(x)) {

            auto arr = nanobind::cast<
                nanobind::ndarray<const double, nanobind::ndim<1>>
            >(x);

            size_t n = arr.shape(0);

            auto *data = new std::vector<double>(n);

            for (size_t i = 0; i < n; ++i)
                (*data)[i] = %(name)s(arr(i));

            nanobind::capsule owner(data, [](void *p) noexcept {
                delete static_cast<std::vector<double> *>(p);
            });

            return nanobind::cast(
                nanobind::ndarray<double, nanobind::numpy>(
                    data->data(), {n}, owner
                )
            );
        }

        throw nanobind::type_error("Expected single value, list of scalars, or NumPy array");
    });
"""