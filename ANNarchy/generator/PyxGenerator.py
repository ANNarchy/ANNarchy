#===============================================================================
#
#     PyxGenerator.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core import Global
from ANNarchy.extensions.bold import BoldMonitor

from ANNarchy.generator.Template import PyxTemplate

from ANNarchy.generator.Population import OpenMPTemplates as omp_templates
from ANNarchy.generator.Population import CUDATemplates as cuda_templates

from ANNarchy.generator.Projection import OpenMPTemplates as proj_omp_templates

from ANNarchy.generator.Projection.Connectivity import LIL_OpenMP, CSR_OpenMP
from ANNarchy.generator.Projection.Connectivity import LIL_CUDA, CSR_CUDA

class PyxGenerator(object):
    """
    Generate the python extension (*.pyx) file comprising of wrapper
    classes for the individual objects. Secondly the definition of accessible
    methods, e. g. simulate(int steps). Generally an extension consists of two
    parts: a struct definition (define accessible parts of the C++ object) and
    the wrapper object.

    In detail, there are extensions available for:

        * parsed populations
        * parsed projections
        * recorder objects

    Implementation Note (HD: 17.06.2015)

        This class could be implemented in general as a set of functions.
        Nevertheless we chose an object-oriented approach, as it is easier to
        use from Generator object. As the submethods has no access to data
        stored in PyxGenerator (there is no real data at all) all private
        functions are marked with @staticmethod to make this clear.

    TODO:

        * handling of specific populations, projections is currently done over
        generator struct in the population/projection object. This should be
        changed as in the C++ code generation.
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
        Perform code generation.
        """
        # Custom user-defined functions (add_function())
        custom_functions_export, functions_wrapper = self._custom_functions()

        # Custom user-defined constants (add_constant())
        custom_constants_export, constants_wrapper = self._custom_constants()

        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for pop in self._populations:
            # Header export
            pop_struct += self._pop_struct(pop)
            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}

        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self._projections:
            # Header export
            proj_struct += self._proj_struct(proj)

            # Projection instance
            proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {
    'id_proj': proj.id,
}

        # struct declaration for each monitor
        monitor_struct = "" #self._pyx_struct_monitor()
        for pop in self._populations:
            monitor_struct += self._pop_monitor_struct(pop)
        for proj in self._projections:
            monitor_struct += self._proj_monitor_struct(proj)
        for mon in Global._network[self._net_id]['monitors']:
            if isinstance(mon, BoldMonitor):
                mon_dict = {
                    'pop_id': mon.object.id,
                    'pop_name': mon.object.name,
                    'mon_id': mon.id,
                    'float_prec': Global.config['precision']
                }
                monitor_struct += mon._specific_template['pyx_struct'] % mon_dict

        # Cython wrappers for the populations
        pop_class = ""
        for pop in self._populations:
            pop_class += self._pop_wrapper(pop)

        # Cython wrappers for the projections
        proj_class = ""
        for proj in self._projections:
            proj_class += self._proj_wrapper(proj)

        # Cython wrappers for the monitors
        monitor_class = ""
        for pop in self._populations:
            monitor_class += self._pop_monitor_wrapper(pop)
        for proj in self._projections:
            monitor_class += self._proj_monitor_wrapper(proj)
        for mon in Global._network[self._net_id]['monitors']:
            if isinstance(mon, BoldMonitor):
                mon_dict = {
                    'pop_id': mon.object.id,
                    'pop_name': mon.object.name,
                    'mon_id': mon.id,
                    'float_prec': Global.config['precision']
                }
                monitor_class += mon._specific_template['pyx_wrapper'] % mon_dict

        from .Template.PyxTemplate import pyx_template
        return pyx_template % {
            'custom_functions_export': custom_functions_export,
            'custom_constants_export': custom_constants_export,
            'pop_struct': pop_struct,
            'pop_ptr': pop_ptr,
            'proj_struct': proj_struct,
            'proj_ptr': proj_ptr,
            'pop_class' : pop_class,
            'proj_class': proj_class,
            'monitor_struct': monitor_struct,
            'monitor_wrapper': monitor_class,
            'functions_wrapper': functions_wrapper,
            'constants_wrapper': constants_wrapper,
            'float_prec': Global.config['precision'],
            'device_specific_export': PyxTemplate.pyx_device_specific[Global.config['paradigm']]['export'],
            'device_specific_wrapper': PyxTemplate.pyx_device_specific[Global.config['paradigm']]['wrapper'],
        }

    @staticmethod
    def _get_proj_template(proj):
        # choose the correct template collection dependent on
        # target platform and storage formate
        if Global.config['paradigm'] == "openmp":
            if proj._storage_format == "lil":
                return LIL_OpenMP.conn_templates
            elif proj._storage_format == "csr":
                return CSR_OpenMP.conn_templates
            else:
                raise NotImplementedError

        elif Global.config['paradigm'] == "cuda":
            # LIL is internally transformed to CSR
            if proj._storage_format == "lil":
                return LIL_CUDA.conn_templates
            elif proj._storage_format == "csr":
                return CSR_CUDA.conn_templates
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

#######################################################################
############## Functions #############################################
#######################################################################
    def _custom_functions(self):
        if len(Global._objects['functions']) == 0:
            return "", ""
        from ANNarchy.parser.Extraction import extract_functions

        export = ""
        wrapper = ""
        for name, func in Global._objects['functions']:
            desc = extract_functions(func, local_global=True)[0]
            # Export
            export += ' '*4 + desc['return_type'] + " " + desc['name'] + '('
            for idx, arg in enumerate(desc['arg_types']):
                export += arg
                if idx < len(desc['arg_types']) - 1:
                    export += ', '
            export += ')' + '\n'

            # Wrapper
            arguments=""
            wrapper += "cpdef np.ndarray func_" + desc['name'] + '('
            for idx, arg in enumerate(desc['args']):
                # Function call
                wrapper += arg
                if idx < len(desc['args']) - 1:
                    wrapper += ', '
                # Element access
                arguments += arg + "[i]"
                if idx < len(desc['args']) - 1:
                    arguments += ', '
            wrapper += '):'
            wrapper += """
    return np.array([%(funcname)s(%(args)s) for i in range(len(%(first_arg)s))])
""" % {'funcname': desc['name'], 'first_arg' : desc['args'][0], 'args': arguments}

        return export, wrapper

#######################################################################
############## Constants  #############################################
#######################################################################
    def _custom_constants(self):
        if len(Global._objects['constants']) == 0:
            return "", ""

        export = ""
        wrapper = ""
        for obj in Global._objects['constants']:
            export += """
    void set_%(name)s(%(float_prec)s)""" % {'name': obj.name, 'float_prec': Global.config['precision']}
            wrapper += """
def _set_%(name)s(%(float_prec)s value):
    set_%(name)s(value)""" % {'name': obj.name, 'float_prec': Global.config['precision']}

        return export, wrapper


#######################################################################
############## Population #############################################
#######################################################################
    @staticmethod
    def _pop_struct(pop):
        """
        Generate population struct definition, mimics the c++ class.
        """

        # Spiking neurons have additional data
        export_refractory = ""
        if pop.neuron_type.type == 'spike':
            if pop.neuron_type.refractory or pop.refractory:
                if Global.config['paradigm'] == "openmp":
                    export_refractory = """
        vector[int] refractory
"""
                else:
                    export_refractory = """
        vector[int] refractory
        bool refractory_dirty
"""

        # Parameters and variables
        export_parameters_variables = ""
        for var in pop.neuron_type.description['parameters']:
            export_parameters_variables += PyxTemplate.pop_attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
        for var in pop.neuron_type.description['variables']:
            export_parameters_variables += PyxTemplate.pop_attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}
        if 'export_parameters_variables' in pop._specific_template.keys():
            export_parameters_variables = pop._specific_template['export_parameters_variables']

        # Arrays for the presynaptic sums of rate-coded neurons
        export_targets = ""
        if pop.neuron_type.type == 'rate':
            export_targets += """
        # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                export_targets += """
        vector[%(float_prec)s] _sum_%(target)s""" % {'target' : target, 'float_prec': Global.config['precision']}
        if 'export_targets' in pop._specific_template.keys():
            export_targets = pop._specific_template['export_targets']

        # Local functions
        export_functions = ""
        if len(pop.neuron_type.description['functions']) > 0:
            export_functions += """
        # Local functions
"""
            for func in pop.neuron_type.description['functions']:
                export_functions += ' '*8 + func['return_type'] + ' ' + func['name'] + '('
                for idx, arg in enumerate(func['arg_types']):
                    export_functions += arg
                    if idx < len(func['arg_types']) - 1:
                        export_functions += ', '
                export_functions += ')' + '\n'

        # Mean firing rate
        export_mean_fr = ""
        if pop.neuron_type.type == 'spike':
            export_mean_fr = """
        # Compute firing rate
        void compute_firing_rate(%(float_prec)s window)""" %{'float_prec': Global.config['precision']}

        # Additional exports
        export_additional = ""
        if 'export_additional' in pop._specific_template.keys():
            export_additional = pop._specific_template['export_additional']

        # Finalize the code
        return PyxTemplate.pop_pyx_struct % {
            'id': pop.id, 'name': pop.name,
            'export_refractory': export_refractory,
            'export_parameters_variables': export_parameters_variables,
            'export_functions': export_functions,
            'export_targets': export_targets,
            'export_mean_fr': export_mean_fr,
            'export_additional': export_additional,
        }


    @staticmethod
    def _pop_wrapper(pop):
        """
        Generate population wrapper definition.
        """
        wrapper_args = "size, max_delay"
        wrapper_init = """
        pop%(id)s.set_size(size)
        pop%(id)s.set_max_delay(max_delay)""" % {'id': pop.id}
        wrapper_access_parameters_variables = ""
        wrapper_access_targets = ""
        wrapper_access_refractory = ""
        wrapper_access_additional = ""


        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            if pop.neuron_type.refractory or pop.refractory:
                if Global.config['paradigm'] == "openmp":
                    wrapper_access_refractory += omp_templates.spike_specific['refractory']['pyx_wrapper'] % {'id': pop.id}
                elif Global.config['paradigm'] == "cuda":
                    wrapper_access_refractory += cuda_templates.spike_specific['refractory']['pyx_wrapper'] % {'id': pop.id}
                else:
                    raise NotImplementedError

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            wrapper_access_parameters_variables += PyxTemplate.pop_attribute_pyx_wrapper[var['locality']] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'parameter'}

        for var in pop.neuron_type.description['variables']:
            wrapper_access_parameters_variables += PyxTemplate.pop_attribute_pyx_wrapper[var['locality']] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'variable'}

        # Arrays for the presynaptic sums of rate-coded neurons
        if pop.neuron_type.type == 'rate':
            wrapper_access_targets += """
    # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                wrapper_access_targets += """
    cpdef np.ndarray get_sum_%(target)s(self):
        return np.array(pop%(id)s._sum_%(target)s)""" % {'id': pop.id, 'target' : target}

        # Local functions
        wrapper_access_functions = ""
        if len(pop.neuron_type.description['functions']) > 0:
            wrapper_access_functions += """
    # Local functions
"""
            for func in pop.neuron_type.description['functions']:
                wrapper_access_functions += ' '*4 + 'cpdef np.ndarray ' + func['name'] + '(self, '
                arguments = ""
                for idx, arg in enumerate(func['args']):
                    # Function call
                    wrapper_access_functions += arg
                    if idx < len(func['args']) - 1:
                        wrapper_access_functions += ', '
                    # Element access
                    arguments += arg + "[i]"
                    if idx < len(func['args']) - 1:
                        arguments += ', '
                wrapper_access_functions += '):'
                wrapper_access_functions += """
        return np.array([pop%(id)s.%(funcname)s(%(args)s) for i in range(len(%(first_arg)s))])
""" % {'id': pop.id, 'funcname': func['name'], 'first_arg' : func['args'][0], 'args': arguments}


        # Mean firing rate
        wrapper_access_mean_fr = ""
        if pop.neuron_type.type == 'spike':
            wrapper_access_mean_fr = """
    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop%(id)s.compute_firing_rate(window)"""% {'id': pop.id}

        # Specific populations can overwrite
        if 'wrapper_args' in pop._specific_template.keys():
            wrapper_args = pop._specific_template['wrapper_args']
        if 'wrapper_init' in pop._specific_template.keys():
            wrapper_init = pop._specific_template['wrapper_init']
        if 'wrapper_access_targets' in pop._specific_template.keys():
            wrapper_access_targets = pop._specific_template['wrapper_access_targets']
        if 'wrapper_access_refractory' in pop._specific_template.keys():
            wrapper_access_refractory = pop._specific_template['wrapper_access_refractory']
        if 'wrapper_access_parameters_variables' in pop._specific_template.keys():
            wrapper_access_parameters_variables = pop._specific_template['wrapper_access_parameters_variables']
        if 'wrapper_access_additional' in pop._specific_template.keys():
            wrapper_access_additional = pop._specific_template['wrapper_access_additional']

        # Finalize the code
        return PyxTemplate.pop_pyx_wrapper % {
            'id': pop.id, 'name': pop.name,
            'wrapper_args' : wrapper_args,
            'wrapper_init' : wrapper_init,
            'wrapper_access_parameters_variables' : wrapper_access_parameters_variables,
            'wrapper_access_targets' : wrapper_access_targets,
            'wrapper_access_functions' : wrapper_access_functions,
            'wrapper_access_refractory' : wrapper_access_refractory,
            'wrapper_access_mean_fr' : wrapper_access_mean_fr,
            'wrapper_access_additional' : wrapper_access_additional,
        }

#######################################################################
############## Projection #############################################
#######################################################################
    @staticmethod
    def _proj_struct(proj):
        """
        The python extension wrapper needs a definition of the corresponding
        C object. The pyx_struct contains all methods, which should be accessible
        by the python extension wrapper.

        Templates:

            attribute_cpp_export: normal accessors for variables/parameters
            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: variables accessed by the wrapper

        """
        # Check for exact intgeration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break

        # basic
        ids = {
            'id': proj.id,
            'float_prec': Global.config['precision']
        }

        # Check if we need delay code
        has_delay = (proj.max_delay > 1)
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            key_delay = "nonuniform"

        # get the base templates
        template_dict = PyxGenerator._get_proj_template(proj)

        # connectivity and weight definition
        connectivity_tpl = template_dict['connectivity_matrix']
        weight_tpl = template_dict['weight_matrix']

        if Global.config['paradigm'] == "openmp":
            sp_tpl = proj_omp_templates.structural_plasticity['pyx_struct']
        else:
            sp_tpl = {}

        # Special case for single weights
        if proj._has_single_weight():
            if Global.config['paradigm'] == "openmp":
                weight_tpl = template_dict['single_weight_matrix']
            else:
                raise NotImplementedError

        # Export connectivity matrix
        export_connectivity_matrix = connectivity_tpl['pyx_struct']
        export_connectivity_matrix += weight_tpl['pyx_struct'] % ids

        # Delay
        export_delay = ""
        if has_delay:
            if Global.config['paradigm'] == "openmp":
                export_delay = template_dict['delay'][key_delay]['pyx_struct'] % ids
            elif Global.config['paradigm'] == "cuda":
                export_delay = template_dict['delay'][key_delay]['pyx_struct'] % ids
            else:
                raise NotImplementedError

        # Event-driven
        export_event_driven = ""
        if has_event_driven:
            if Global.config['paradigm'] == "openmp":
                export_event_driven = template_dict['event_driven']['pyx_struct']
            elif Global.config['paradigm'] == "cuda":
                export_event_driven = template_dict['event_driven']['pyx_struct']
            else:
                raise NotImplementedError

        # Determine all export methods
        export_parameters_variables = ""
        # Parameters
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            export_parameters_variables += PyxTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
        # Variables
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            export_parameters_variables += PyxTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}

        # Local functions
        export_functions = ""
        if len(proj.synapse_type.description['functions']) > 0:
            export_functions += """
        # Local functions
"""
            for func in proj.synapse_type.description['functions']:
                export_functions += ' '*8 + func['return_type'] + ' ' + func['name'] + '('
                for idx, arg in enumerate(func['arg_types']):
                    export_functions += arg
                    if idx < len(func['arg_types']) - 1:
                        export_functions += ', '
                export_functions += ')' + '\n'

        # Structural plasticity
        structural_plasticity = ""
        if Global.config['structural_plasticity']:
            # Pruning in the synapse
            if 'pruning' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['pruning']
            if 'creating' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['creating']

            # Retrieve the names of extra attributes
            extra_args = ""
            for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name']
            # Generate the code
            structural_plasticity += sp_tpl['func'] % {'extra_args': extra_args}

        # Specific projections can overwrite
        if 'export_connectivity' in proj._specific_template.keys():
            export_connectivity_matrix = proj._specific_template['export_connectivity']
        if 'export_delay' in proj._specific_template.keys() and has_delay:
            export_delay = proj._specific_template['export_delay']
        if 'export_event_driven' in proj._specific_template.keys() and has_event_driven:
            export_event_driven = proj._specific_template['export_event_driven']
        if 'export_parameters_variables' in proj._specific_template.keys():
            export_parameters_variables = proj._specific_template['export_parameters_variables']


        return PyxTemplate.proj_pyx_struct % {
            'id_proj': proj.id,
            'export_connectivity': export_connectivity_matrix,
            'export_delay': export_delay,
            'export_event_driven': export_event_driven,
            'export_parameters_variables': export_parameters_variables,
            'export_functions': export_functions,
            'export_structural_plasticity': structural_plasticity,
            'export_additional': proj._specific_template['export_additional'] if 'export_additional' in proj._specific_template.keys() else ""
        }

    @staticmethod
    def _proj_wrapper(proj):
        """
        Generates the python extension wrapper, which allows access from Python
        to the C module. There are three optional parts (structural plasticity,
        non-uniform delays and exact integration of synaptic events) which we
        need to handle seperatly. The rest of the variables/parameters is handled
        by the standard accessors.

        Templates:

            attribute_pyx_wrapper: normal accessors for variables/parameters
            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: __cinit__ code

        """
        # Check for exact intgeration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break

        # basic
        ids = {
            'id_proj': proj.id,
            'float_prec': Global.config['precision']
        }

        # Check if we need delay code
        has_delay = (proj.max_delay > 1)
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            key_delay = "nonuniform"

        # Import attributes templates
        pyx_acc_tpl = PyxTemplate.attribute_pyx_wrapper

        # select the base template
        template_dict = PyxGenerator._get_proj_template(proj)

        connectivity_tpl = template_dict['connectivity_matrix']
        weight_tpl = template_dict['weight_matrix']

        # Special case for single weights
        if proj._has_single_weight():
            weight_tpl = template_dict['single_weight_matrix']

        if Global.config['paradigm'] == "openmp":
            sp_tpl = proj_omp_templates.structural_plasticity['pyx_wrapper']
        else:
            sp_tpl = {}

        # Arguments to the wrapper (default: synapses)
        wrapper_args = connectivity_tpl['pyx_wrapper_args']
        wrapper_args += weight_tpl['pyx_wrapper_args']

        # Wrapper constructor
        csr_type = 'CSRConnectivity' if proj._storage_order == 'post_to_pre' else 'CSRConnectivityPre1st'
        wrapper_init = connectivity_tpl['pyx_wrapper_init'] % {'id_proj': proj.id, 'csr_type':csr_type}
        wrapper_init += weight_tpl['pyx_wrapper_init'] % {'id_proj': proj.id}

        # Wrapper access to connectivity matrix
        wrapper_access_connectivity = connectivity_tpl['pyx_wrapper_accessor'] % ids
        wrapper_access_connectivity += weight_tpl['pyx_wrapper_accessor'] % ids

        # Delays
        if not has_delay:
            wrapper_init_delay = ""
            wrapper_access_delay = ""
        else:
            # Initialize the wrapper
            wrapper_init_delay = template_dict['delay'][key_delay]['pyx_wrapper_init'] % ids
            # Access in wrapper
            wrapper_access_delay = template_dict['delay'][key_delay]['pyx_wrapper_accessor'] % ids

        # Event-driven
        wrapper_init_event_driven = ""
        if has_event_driven:
            try:
                wrapper_init_event_driven = template_dict['event_driven']['pyx_wrapper_init'] % ids
            except KeyError:
                raise NotImplementedError

        # Determine all accessor methods
        wrapper_access_parameters_variables = ""
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            wrapper_access_parameters_variables += pyx_acc_tpl[var['locality']] % {'id' : proj.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'parameter'}
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            wrapper_access_parameters_variables += pyx_acc_tpl[var['locality']] % {'id' : proj.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'variable'}

        # Local functions
        wrapper_access_functions = ""
        if len(proj.synapse_type.description['functions']) > 0:
            wrapper_access_functions += """
    # Local functions
"""
            for func in proj.synapse_type.description['functions']:
                wrapper_access_functions += ' '*4 + 'cpdef np.ndarray ' + func['name'] + '(self, '
                arguments = ""
                for idx, arg in enumerate(func['args']):
                    # Function call
                    wrapper_access_functions += arg
                    if idx < len(func['args']) - 1:
                        wrapper_access_functions += ', '
                    # Element access
                    arguments += arg + "[i]"
                    if idx < len(func['args']) - 1:
                        arguments += ', '
                wrapper_access_functions += '):'
                wrapper_access_functions += """
        return np.array([proj%(id)s.%(funcname)s(%(args)s) for i in range(len(%(first_arg)s))])
""" % {'id': proj.id, 'funcname': func['name'], 'first_arg' : func['args'][0], 'args': arguments}


        # Additional declarations
        additional_declarations = ""

        # Structural plasticity (TODO: not templated yet)
        structural_plasticity = ""
        if Global.config['structural_plasticity']:
            # Pruning in the synapse
            if 'pruning' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['pruning'] % {'id' : proj.id}

            # Creating in the synapse
            if 'creating' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['creating'] % {'id' : proj.id}

            # Retrieve the names of extra attributes
            extra_args = ""
            extra_values = ""
            for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name']
                    extra_values += ', ' +  var['name']

            # Generate the code
            structural_plasticity += sp_tpl['func'] % {'id' : proj.id, 'extra_args': extra_args, 'extra_values': extra_values}

        # Specific projections can overwrite
        if 'wrapper_args' in proj._specific_template.keys():
            wrapper_args = proj._specific_template['wrapper_args']
        if 'wrapper_init_connectivity' in proj._specific_template.keys():
            wrapper_init = proj._specific_template['wrapper_init_connectivity']
        if 'wrapper_access_connectivity' in proj._specific_template.keys():
            wrapper_access_connectivity = proj._specific_template['wrapper_access_connectivity']
        if 'wrapper_init_delay' in proj._specific_template.keys() and has_delay:
            wrapper_init_delay = proj._specific_template['wrapper_init_delay']
        if 'wrapper_access_delay' in proj._specific_template.keys() and has_delay:
            wrapper_access_delay = proj._specific_template['wrapper_access_delay']
        if 'wrapper_init_event_driven' in proj._specific_template.keys() and has_event_driven:
            wrapper_init_event_driven = proj._specific_template['wrapper_init_event_driven']
        if 'wrapper_access_parameters_variables' in proj._specific_template.keys():
            wrapper_access_parameters_variables = proj._specific_template['wrapper_access_parameters_variables']
        if 'wrapper_access_additional' in proj._specific_template.keys():
            additional_declarations = proj._specific_template['wrapper_access_additional']

        return PyxTemplate.proj_pyx_wrapper % {
            'id_proj': proj.id,
            'wrapper_args': wrapper_args,
            'wrapper_init_connectivity': wrapper_init,
            'wrapper_init_delay': wrapper_init_delay,
            'wrapper_init_event_driven': wrapper_init_event_driven,
            'wrapper_access_connectivity': wrapper_access_connectivity,
            'wrapper_access_delay': wrapper_access_delay,
            'wrapper_access_parameters_variables': wrapper_access_parameters_variables,
            'wrapper_access_functions': wrapper_access_functions,
            'wrapper_access_structural_plasticity': structural_plasticity,
            'wrapper_access_additional': additional_declarations
        }

#######################################################################
############## Monitors  ##############################################
#######################################################################
    @staticmethod
    def _pop_monitor_struct(pop):
        """
        Generate recorder struct.
        """
        tpl_code = """
    # Population %(id)s (%(name)s) : Monitor
    cdef cppclass PopRecorder%(id)s (Monitor):
        PopRecorder%(id)s(vector[int], int, int, long) except +
        long int size_in_bytes()
        void clear()
"""
        attributes = []
        for var in pop.neuron_type.description['parameters'] + pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            if var['name'] in pop.neuron_type.description['local']:
                tpl_code += """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s
""" % {'name': var['name'], 'type': var['ctype']}
            elif var['name'] in pop.neuron_type.description['global']:
                tpl_code += """
        vector[%(type)s] %(name)s
        bool record_%(name)s
""" % {'name': var['name'], 'type': var['ctype']}

        if pop.neuron_type.type == 'spike':
            tpl_code += """
        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()
"""
            if pop.neuron_type.axon_spike:
                tpl_code += """
        map[int, vector[long]] axon_spike
        bool record_axon_spike
        void clear_axon_spike()
"""

        # Arrays for the presynaptic sums
        if pop.neuron_type.type == 'rate':
            tpl_code += """
        # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                tpl_code += """
        vector[vector[%(float_prec)s]] _sum_%(target)s
        bool record__sum_%(target)s
""" % {'target': target, 'float_prec': Global.config['precision']}

        return tpl_code % {'id' : pop.id, 'name': pop.name}

    @staticmethod
    def _pop_monitor_wrapper(pop):
        """
        Generate recorder wrapper.
        """
        tpl_code = """
# Population Monitor wrapper
cdef class PopRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new PopRecorder%(id)s(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<PopRecorder%(id)s *>self.thisptr).size_in_bytes()

    def clear(self):
        (<PopRecorder%(id)s *>self.thisptr).clear()

"""
        attributes = []
        for var in pop.neuron_type.description['parameters'] + pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])
            tpl_code += """
    property %(name)s:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).record_%(name)s = val
    def clear_%(name)s(self):
        (<PopRecorder%(id)s *>self.thisptr).%(name)s.clear()
""" % {'id' : pop.id, 'name': var['name']}

        if pop.neuron_type.type == 'spike':
            tpl_code += """
    property spike:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).spike
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).spike = val
    property record_spike:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).record_spike
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).record_spike = val
    def clear_spike(self):
        (<PopRecorder%(id)s *>self.thisptr).clear_spike()
""" % {'id' : pop.id}

            if pop.neuron_type.axon_spike:
                tpl_code += """
    property axon_spike:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).axon_spike
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).axon_spike = val
    property record_axon_spike:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).record_axon_spike
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).record_axon_spike = val
    def clear_axon_spike(self):
        (<PopRecorder%(id)s *>self.thisptr).clear_axon_spike()
""" % {'id' : pop.id}

        # Arrays for the presynaptic sums
        if pop.neuron_type.type == 'rate':
            tpl_code += """
    # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                tpl_code += """
    property %(name)s:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (<PopRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<PopRecorder%(id)s *>self.thisptr).record_%(name)s = val
    def clear_%(name)s(self):
        (<PopRecorder%(id)s *>self.thisptr).%(name)s.clear()
""" % {'id' : pop.id, 'name': '_sum_'+target}

        return tpl_code % {'id' : pop.id, 'name': pop.name}

    @staticmethod
    def _proj_monitor_struct(proj):
        """
        Generate projection recorder struct
        """

        # Specific template
        if 'monitor_export' in proj._specific_template.keys():
            return proj._specific_template['monitor_export']


        code = """
    # Projection %(id)s : Monitor
    cdef cppclass ProjRecorder%(id)s (Monitor):
        ProjRecorder%(id)s(vector[int], int, int, long) except +
"""

        templates = {
        'local': """
        vector[vector[vector[%(type)s]]] %(name)s
        bool record_%(name)s
""",
        'semiglobal': """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s
""",
        'global': """
        vector[%(type)s] %(name)s
        bool record_%(name)s
"""
        }

        attributes = []
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            # Get the locality
            locality = var['locality']
            # Special case for single weights
            if var['name'] == "w" and proj._has_single_weight():
                locality = 'global'

            # Use the correct template
            code +=  templates[locality] % {'name': var['name'], 'type': var['ctype']}

        return code % {'id' : proj.id}

    @staticmethod
    def _proj_monitor_wrapper(proj):
        """
        Generate projection recorder struct
        """

        # Specific template
        if 'monitor_wrapper' in proj._specific_template.keys():
            return proj._specific_template['monitor_wrapper']

        code = """
# Projection Monitor wrapper
cdef class ProjRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        self.thisptr = new ProjRecorder%(id)s(ranks, period, period_offset, offset)
"""

        attributes = []
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            code += """
    property %(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s = val
    def clear_%(name)s(self):
        (<ProjRecorder%(id)s *>self.thisptr).%(name)s.clear()
""" % {'id' : proj.id, 'name': var['name']}

        return code % {'id' : proj.id}
