"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.generator.NanoBind.BaseTemplate import basetemplate
from ANNarchy.generator.NanoBind.Population import *
from ANNarchy.generator.NanoBind.Projection import *
from ANNarchy.generator.NanoBind.Profiler import profiler_template

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.parser.Extraction import extract_functions
from ANNarchy.intern.Profiler import Profiler

class NanoBindGenerator:
    """
    Create a Python-wrapper for the ANNarchy simulation core using the nanobind package (https://nanobind.readthedocs.io/en/latest/index.html)
    """
    def __init__(self, annarchy_dir, net_id):
        """
        Store a list of population und projection objects for later processing.
        """
        self._annarchy_dir = annarchy_dir
        self.net_id = net_id
        self.network = NetworkManager().get_network(self.net_id)

    def generate(self):
        """
        Generate the code stored in `ANNarchyCore[net_id].cpp`.
        """
        pop_struct_code = ""
        proj_struct_code = ""
        pop_mon_code = ""
        proj_mon_code = ""

        # Add device specific functions
        if _check_paradigm("openmp", self.net_id):
            device_specific = "\tm.def(\"set_number_threads\", &setNumberThreads);"
        elif _check_paradigm("cuda", self.net_id):
            device_specific = "\tm.def(\"set_device\", &setDevice);"
        else:
            raise NotImplementedError

        # Functions
        functions_code = self._generate_functions()

        # Constants
        constant_code = self._generate_constant(self.network.get_constants())

        # Populations
        for pop in self.network.get_populations():
            if 'wrapper' in pop._specific_template.keys():
                pop_struct_code += pop._specific_template['wrapper']
            else:
                pop_struct_code += self._generate_pop_wrapper(pop)

            if 'monitor_wrapper' in pop._specific_template.keys():
                pop_mon_code += pop._specific_template['monitor_wrapper']
            else:
                pop_mon_code += self._generate_pop_mon_wrapper(pop)

        # Projections
        for proj in self.network.get_projections():
            if 'wrapper' in proj._specific_template.keys():
                proj_struct_code += proj._specific_template['wrapper']
            else:
                proj_struct_code += self._generate_proj_wrapper(proj)

            if 'monitor_wrapper' in proj._specific_template.keys():
                proj_mon_code += proj._specific_template['monitor_wrapper']
            else:
                proj_mon_code += self._generate_proj_mon_wrapper(proj)

        # Profiling of simulation kernel is optional
        profiling_code = profiler_template if Profiler().enabled else ""

        return basetemplate % {
            'net_id': self.net_id,
            'device_specific': device_specific,
            'functions_wrapper': functions_code,
            'constant_wrapper': constant_code,
            'pop_struct_wrapper': pop_struct_code,
            'proj_struct_wrapper': proj_struct_code,
            'pop_mon_wrapper': pop_mon_code,
            'proj_mon_wrapper': proj_mon_code,
            'profiling_wrapper': profiling_code
        }

    def _generate_pop_wrapper(self, pop: "Population") -> str:
        """
        Generate wrapper for the C++ *PopStruct* structure.
        """
        additional_func = ""
        attributes = ""

        # synaptic delay
        attributes += f"""\t\t.def("update_max_delay", &PopStruct{pop.id}::update_max_delay)\n"""

        # On GPUs, we need to trigger a device-to-host explicitly
        if _check_paradigm("cuda", self.net_id):
            attributes += f"""\t\t.def("device_to_host", &PopStruct{pop.id}::device_to_host)\n"""

        # Model attributes
        for attr in pop.neuron_type.description['attributes']:
            # internal variables should not exposed to Python
            if attr.startswith("rand_"):
                continue

            # Attributes are accessed directly from Python
            attributes += """\t\t.def_rw("{name}", &PopStruct{id}::{name})\n""".format(id=pop.id, name=attr)

            # On GPUs, we need to trigger a host-to-device transfer on next call of simulate()
            if _check_paradigm("cuda", self.net_id):
                # for local attributes
                if attr in pop.neuron_type.description['local']:
                    attributes += """\t\t.def_rw("{name}_host_to_device", &PopStruct{id}::{name}_host_to_device)\n""".format(id=pop.id, name=attr)
                # for global attributes which are updated over time
                if attr in pop.neuron_type.description['global'] and attr in [tmp['name'] for tmp in pop.neuron_type.description['variables']]:
                    attributes += """\t\t.def_rw("{name}_host_to_device", &PopStruct{id}::{name}_host_to_device)\n""".format(id=pop.id, name=attr)

        # Arrays for the post-synaptic potential
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                attributes += """\t\t.def_rw("{name}", &PopStruct{id}::{name})\n""".format(id=pop.id, name="_sum_"+target)

        # Functions
        for func in pop.neuron_type.description['functions']:
            # Wrapper
            additional_func += f"""
        .def("{func['name']}", &PopStruct{pop.id}::{func['name']})"""

        # Type-specific functions
        if pop.neuron_type.type == "spike":
            additional_func += f"""\t\t.def("compute_firing_rate", &PopStruct{pop.id}::compute_firing_rate)\n"""

            if (pop.neuron_type.refractory or pop.refractory):
                additional_func += f"""\t\t.def_rw("refractory", &PopStruct{pop.id}::refractory)\n"""
                if _check_paradigm("cuda", self.net_id):
                    additional_func += f"""\t\t.def_rw("refractory_dirty", &PopStruct{pop.id}::refractory_dirty)\n"""

        wrapper_code = pop_struct_wrapper % {
            'id': pop.id,
            'attributes': attributes,
            'additional': additional_func
        }
        wrapper_code += '\n'
        return wrapper_code

    def _generate_proj_wrapper(self, proj: "Projection") -> str:

        methods = ""
        attributes = ""
        additional_func = ""

        # Connectivity as LIL
        # HD (18th Aug. 2025):  The C++ template library offers in some cases a const- and non-const accessor.
        #                       To ensure that Python accesses only using the non-const accessor an additional
        #                       "nanobind::overload_cast<>" is needed. Otherwise, its compiler dependent which
        #                       version is bound consequently resulting in strange side-effects ...
        connectivity = f"""
        .def("init_from_lil", &ProjStruct{proj.id}::init_from_lil)
        .def("post_rank", nanobind::overload_cast<>(&ProjStruct{proj.id}::get_post_rank))
        .def("dendrite_size", &ProjStruct{proj.id}::dendrite_size)
        .def("nb_dendrites", &ProjStruct{proj.id}::nb_dendrites)
        .def("pre_ranks", &ProjStruct{proj.id}::get_pre_ranks)
        .def("pre_rank", &ProjStruct{proj.id}::get_dendrite_pre_rank)
        .def("nb_synapses", &ProjStruct{proj.id}::nb_synapses)"""

        # Special case for spiking projections
        if proj.synapse_type.type == "spike":
            connectivity += f"""
        .def("nb_efferent_synapses", &ProjStruct{proj.id}::nb_efferent_synapses)
"""

        # Contrary to *Population* attributes, we don't access the local attributes directly but use a common interface.
        datatypes = {
            'local': [],
            'semiglobal': [],
            'global': []
        }

        # Collect necessary data types
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            locality = var['locality']
            if var['name'] == "w" and proj._has_single_weight():
                locality = 'global'

            if var['ctype'] not in datatypes[locality]:
                datatypes[locality].append(var['ctype'])

        # Check if we need delay code
        has_delay = (proj.max_delay > 1)
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            if proj.synapse_type.type == "rate":
                key_delay = "nonuniform_rate_coded"
            else:
                key_delay = "nonuniform_spiking"
        if has_delay:
            attributes += proj_delays[key_delay] % {'id': proj.id}
        
        # Generate accessor for synapse attributes
        for ctype in datatypes["local"]:
            attributes += proj_local_attr % {'id': proj.id, 'ctype': ctype}

        for ctype in datatypes["semiglobal"]:
            attributes += proj_semiglobal_attr % {'id': proj.id, 'ctype': ctype}

        for ctype in datatypes["global"]:
            attributes += proj_global_attr % {'id': proj.id, 'ctype': ctype}

        # Functions
        for func in proj.synapse_type.description['functions']:
            # Wrapper
            additional_func += f"""
        .def("{func['name']}", &ProjStruct{proj.id}::{func['name']})"""  

        # Structural plasticity
        structural_plasticity = ConfigManager().get('structural_plasticity', self.net_id)
        if 'creating' in proj.synapse_type.description.keys():
            structural_plasticity = True
            attributes += f"""
        // Creating
        .def_rw("_creating", &ProjStruct{proj.id}::_creating)
        .def_rw("_creating_period", &ProjStruct{proj.id}::_creating_period)
        .def_rw("_creating_offset", &ProjStruct{proj.id}::_creating_offset)"""
            
        if 'pruning' in proj.synapse_type.description.keys():
            structural_plasticity = True
            attributes += f"""
        // Pruning
        .def_rw("_pruning", &ProjStruct{proj.id}::_pruning)
        .def_rw("_pruning_period", &ProjStruct{proj.id}::_pruning_period)
        .def_rw("_pruning_offset", &ProjStruct{proj.id}::_pruning_offset)"""

        if structural_plasticity:
            methods += f"""
        // Structural plasticity
        .def("synapse_exists", &ProjStruct{proj.id}::synapse_exists)
        .def("add_single_synapse", &ProjStruct{proj.id}::add_single_synapse)
        .def("add_multiple_synapses", &ProjStruct{proj.id}::add_multiple_synapses)
        .def("remove_single_synapse", &ProjStruct{proj.id}::remove_single_synapse)
        """
            
        # Generate the wrapper
        wrapper_code = proj_struct_wrapper % {
            'id': proj.id,
            'connectivity': connectivity,
            'methods': methods,
            'attributes': attributes,
            'additional': additional_func,
            'float_prec': ConfigManager().get("precision", self.net_id)
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

        # Arrays for the post-synaptic potential
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                record_flag += """\t\t.def_rw("record_{name}", &PopRecorder{id}::record_{name})\n""".format(id=pop.id, name="_sum_"+target)

                record_container += """\t\t.def_rw("{name}", &PopRecorder{id}::{name})\n""".format(id=pop.id, name="_sum_"+target)

                clear_container += """\t\t.def("clear_{name}", &PopRecorder{id}::clear_{name})\n""".format(id=pop.id, name="_sum_"+target)

        # Spike events        
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


    def _generate_proj_mon_wrapper(self, proj: "Projection") -> str:
        """
        Generate wrapper for the C++ *PopRecorder* structure.
        """
        record_flag = ""
        record_container = ""
        clear_container = ""

        for attr in proj.synapse_type.description['variables']:
            
            record_flag +="""\t\t.def_rw("record_{name}", &ProjRecorder{id}::record_{name})\n""".format(id=proj.id, name=attr['name'])
            
            record_container += """\t\t.def_rw("{name}", &ProjRecorder{id}::{name})\n""".format(id=proj.id, name=attr['name'])
            
            clear_container += """\t\t.def("clear_{name}", &ProjRecorder{id}::clear_{name})\n""".format(id=proj.id, name=attr['name'])

        wrapper_code = proj_mon_wrapper % {
            'id': proj.id,
            'record_flag': record_flag,
            'record_container': record_container,
            'clear_container': clear_container
        }
        wrapper_code += '\n'

        return wrapper_code


    def _generate_constant(self, constants) -> str:
        """
        Generate wrapper for the C++ constants.
        """

        if len(constants) == 0:
            return ""

        wrapper_code = """
    // Constants"""
        
        for constant in constants:
            wrapper_code += f"""
    m.def("set_{constant.name}", &set_{constant.name});"""

        return  wrapper_code

    def _generate_functions(self) -> str:
        """
        Generate wrapper for the global functions.
        """

        if GlobalObjectManager().number_functions() == 0:
            return ""

        wrapper_code = """
    // Custom global functions"""
        
        for  _, function in GlobalObjectManager().get_functions():
            # Extract the name
            desc = extract_functions(description=function, local_global=True, net_id=self.net_id)[0]
            # Wrapper
            wrapper_code += f"""
    m.def("{desc['name']}", &{desc['name']});"""

        return  wrapper_code
    