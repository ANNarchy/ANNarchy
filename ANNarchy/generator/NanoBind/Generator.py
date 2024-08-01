"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.generator.NanoBind.BaseTemplate import basetemplate
from ANNarchy.generator.NanoBind.Population import *
from ANNarchy.generator.NanoBind.Projection import *

class NanoBindGenerator:
    """
    Create a Python-wrapper for the ANNarchy simulation core using the nanobind package (https://nanobind.readthedocs.io/en/latest/index.html)
    """
    def __init__(self, annarchy_dir, populations, projections, net_id):
        """
        Store a list of population und projection objects for later processing.
        """
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._net_id = net_id

    def generate(self):
        """
        Generate the code stored in *ANNarchyCore[net_id].cpp*.
        """
        pop_struct_code = ""
        proj_struct_code = ""
        pop_mon_code = ""

        for pop in self._populations:

            pop_struct_code += self._generate_pop_wrapper(pop)

            pop_mon_code += self._generate_pop_mon_wrapper(pop)

        for proj in self._projections:
            proj_struct_code += proj_struct_wrapper % {
                'id': proj.id
            }
            proj_struct_code += '\n'

        return basetemplate % {
            'net_id': self._net_id,
            'pop_struct_wrapper': pop_struct_code,
            'proj_struct_wrapper': proj_struct_code,
            'pop_mon_wrapper': pop_mon_code
        }

    def _generate_pop_wrapper(self, pop: "Population") -> str:
        """
        Generate wrapper for the C++ *PopStruct* structure.
        """
        additional_func = ""
        attributes = ""

        # Model attributes
        for attr in pop.neuron_type.description['attributes']:
            # internal variables should not exposed to Python
            if attr.startswith("rand_"):
                continue

            attributes += """\t\t.def_rw("{name}", &PopStruct{id}::{name})\n""".format(id=pop.id, name=attr)

        # type-specific functions
        if pop.neuron_type.type == "spike":
            additional_func += """\t\t.def("compute_firing_rate", &PopStruct{id}::compute_firing_rate)\n""".format(id=pop.id)

        wrapper_code = pop_struct_wrapper % {
            'id': pop.id,
            'attributes': attributes,
            'additional': additional_func
        }
        wrapper_code += '\n'
        return wrapper_code

    def _generate_pop_mon_wrapper(self, pop: "Population") -> str:
        """
        Generate wrapper for the C++ *PopRecorder* structure.
        """
        record_flag = ""
        record_container = ""
        clear_container = ""
        for attr in pop.neuron_type.description['variables']:
            record_flag +="""\t\t.def_rw("record_{name}", &PopRecorder{id}::record_{name})\n""".format(id=pop.id, name=attr['name'])
            record_container += """\t\t.def_rw("{name}", &PopRecorder{id}::{name})\n""".format(id=pop.id, name=attr['name'])
            clear_container += """\t\t.def("clear_{name}", &PopRecorder{id}::clear_{name})\n""".format(id=pop.id, name=attr['name'])
        if pop.neuron_type.type == "spike":
            record_flag +="""\t\t.def_rw("record_{name}", &PopRecorder{id}::record_{name})\n""".format(id=pop.id, name='spike')
            record_container += """\t\t.def_rw("{name}", &PopRecorder{id}::{name})\n""".format(id=pop.id, name='spike')
            clear_container += """\t\t.def("clear_{name}", &PopRecorder{id}::clear_{name})\n""".format(id=pop.id, name='spike')

        wrapper_code = pop_mon_wrapper % {
            'id': pop.id,
            'record_flag': record_flag,
            'record_container': record_container,
            'clear_container': clear_container
        }
        wrapper_code += '\n'

        return wrapper_code
