"""

    PyxGenerator.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from ANNarchy.core import Global

import ANNarchy.generator.Template.PopulationTemplate as PopTemplate
import ANNarchy.generator.Template.ProjectionTemplate as ProjTemplate
import ANNarchy.generator.Template.PyxTemplate as PyxTemplate

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

        from .Template.PyxTemplate import pyx_template
        return pyx_template % {
            'pop_struct': pop_struct, 'pop_ptr': pop_ptr,
            'proj_struct': proj_struct, 'proj_ptr': proj_ptr,
            'pop_class' : pop_class, 'proj_class': proj_class,
            'monitor_struct': monitor_struct, 'monitor_wrapper': monitor_class
        }

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
                export_refractory = """
        vector[int] refractory
"""
        # Parameters and variables
        export_parameters_variables = ""
        for var in pop.neuron_type.description['parameters']:
            export_parameters_variables += PopTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
        for var in pop.neuron_type.description['variables']:
            export_parameters_variables += PopTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}
        if 'export_parameters_variables' in pop._specific_template.keys():
            export_parameters_variables = pop._specific_template['export_parameters_variables']

        # Arrays for the presynaptic sums of rate-coded neurons
        export_targets = ""
        if pop.neuron_type.type == 'rate':
            export_targets += """
        # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                export_targets += """
        vector[double] _sum_%(target)s""" % {'target' : target}
        if 'export_targets' in pop._specific_template.keys():
            export_targets = pop._specific_template['export_targets']

        # Additional exports
        export_additional = ""
        if 'export_additional' in pop._specific_template.keys():
            export_additional = pop._specific_template['export_additional']

        # Finalize the code
        return PyxTemplate.pop_pyx_struct % {
            'id': pop.id, 'name': pop.name,
            'export_refractory': export_refractory,
            'export_parameters_variables': export_parameters_variables,
            'export_targets': export_targets,
            'export_additional': export_additional,
        }


    @staticmethod
    def _pop_wrapper(pop):
        """
        Generate population wrapper definition.
        """
        wrapper_args = "size"
        wrapper_init = """
        pop%(id)s = size""" % {'id': pop.id}
        wrapper_access_parameters_variables = ""
        wrapper_access_targets = ""
        wrapper_access_refractory = ""
        wrapper_access_additional = ""


        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            if pop.neuron_type.refractory or pop.refractory:
                wrapper_access_refractory += """
    # Refractory period
    cpdef np.ndarray get_refractory(self):
        return pop%(id)s.refractory
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value
""" % {'id': pop.id}

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            wrapper_access_parameters_variables += PopTemplate.attribute_pyx_wrapper[var['locality']] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'parameter'}
        for var in pop.neuron_type.description['variables']:
            wrapper_access_parameters_variables += PopTemplate.attribute_pyx_wrapper[var['locality']] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'variable'}

        # Arrays for the presynaptic sums of rate-coded neurons
        if pop.neuron_type.type == 'rate':
            wrapper_access_targets += """
    # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                wrapper_access_targets += """
    cpdef np.ndarray get_sum_%(target)s(self):
        return np.array(pop%(id)s._sum_%(target)s)""" % {'id': pop.id, 'target' : target}


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
            'wrapper_access_refractory' : wrapper_access_refractory,
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

        # Check if we need delay code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Import templates
        connectivity_tpl = ProjTemplate.lil_connectivity_matrix_omp if Global.config['paradigm'] == "openmp" else ProjTemplate.csr_connectivity_matrix_cuda

        weight_tpl = ProjTemplate.lil_weight_matrix_omp if Global.config['paradigm'] == "openmp" else ProjTemplate.csr_weight_matrix_cuda

        sp_tpl = ProjTemplate.structural_plasticity['pyx_struct']

        # Special case for single weights
        if proj._has_single_weight():
            weight_tpl = ProjTemplate.single_weight_matrix_omp

        # Export connectivity matrix
        export_connectivity_matrix = connectivity_tpl['pyx_struct']
        export_connectivity_matrix += weight_tpl['pyx_struct']

        # Delay
        export_delay=""
        if has_delay:
            export_delay = ProjTemplate.delay['pyx_struct'] % {'id': proj.id}

        # Event-driven
        export_event_driven = ""
        if has_event_driven:
            export_event_driven = ProjTemplate.event_driven['pyx_struct']

        # Determine all export methods
        export_parameters_variables = ""
        # Parameters
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            export_parameters_variables += ProjTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
        # Variables
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            export_parameters_variables += ProjTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}

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

        # Check if we need delay code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Import attributes templates
        pyx_acc_tpl = ProjTemplate.attribute_pyx_wrapper

        # Import connectivity matrix template
        connectivity_tpl = ProjTemplate.lil_connectivity_matrix_omp if Global.config['paradigm'] == "openmp" else ProjTemplate.csr_connectivity_matrix_cuda

        # Import weight array template
        weight_tpl = ProjTemplate.lil_weight_matrix_omp if Global.config['paradigm'] == "openmp" else ProjTemplate.csr_weight_matrix_cuda

        # Special case for single weights
        if proj._has_single_weight():
            weight_tpl = ProjTemplate.single_weight_matrix_omp

        sp_tpl = ProjTemplate.structural_plasticity['pyx_wrapper']

        # Arguments to the wrapper (default: synapses)
        wrapper_args = connectivity_tpl['pyx_wrapper_args']
        wrapper_args += weight_tpl['pyx_wrapper_args']

        # Wrapper constructor
        wrapper_init = connectivity_tpl['pyx_wrapper_init'] % {'id_proj': proj.id}
        wrapper_init += weight_tpl['pyx_wrapper_init'] % {'id_proj': proj.id}

        # Wrapper sccess to connectivity matrix
        wrapper_access_connectivity = connectivity_tpl['pyx_wrapper_accessor'] % {'id_proj': proj.id}
        wrapper_access_connectivity += weight_tpl['pyx_wrapper_accessor'] % {'id_proj': proj.id}

        # Delays
        wrapper_init_delay = ""; wrapper_access_delay=""
        if has_delay:
            # Initialize the wrapper
            wrapper_init_delay = ProjTemplate.delay['pyx_wrapper_init'] % {'id_proj': proj.id}
            # Access in wrapper
            wrapper_access_delay = ProjTemplate.delay['pyx_wrapper_accessor'] % {'id_proj': proj.id}

        # Event-driven
        wrapper_init_event_driven = ""
        if has_event_driven:
            wrapper_init_event_driven = ProjTemplate.event_driven['pyx_wrapper_init'] % {'id_proj': proj.id}

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
            'wrapper_access_structural_plasticity': structural_plasticity,
            'wrapper_access_additional': additional_declarations
        }

#######################################################################
############## Recording ##############################################
#######################################################################
    @staticmethod
    def _pop_monitor_struct(pop):
        """
        Generate recorder struct.
        """
        tpl_code = """
    # Population %(id)s (%(name)s) : Monitor
    cdef cppclass PopRecorder%(id)s (Monitor):
        PopRecorder%(id)s(vector[int], int, long) except +
"""
        for var in pop.neuron_type.description['variables']:
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
"""

        # Arrays for the presynaptic sums
        if pop.neuron_type.type == 'rate':
            tpl_code += """
        # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                tpl_code += """
        vector[vector[double]] _sum_%(target)s
        bool record__sum_%(target)s
""" % {'target': target}

        return tpl_code % {'id' : pop.id, 'name': pop.name}

    @staticmethod
    def _pop_monitor_wrapper(pop):
        """
        Generate recorder wrapper.
        """
        tpl_code = """
# Population Monitor wrapper
cdef class PopRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, long offset):
        self.thisptr = new PopRecorder%(id)s(ranks, period, offset)
"""

        for var in pop.neuron_type.description['variables']:
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
        for idx in range((<PopRecorder%(id)s *>self.thisptr).spike.size()):
            (<PopRecorder%(id)s *>self.thisptr).spike[idx].clear()
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
        tpl_code = """
    # Projection %(id)s : Monitor
    cdef cppclass ProjRecorder%(id)s (Monitor):
        ProjRecorder%(id)s(vector[int], int, long) except +
"""
        for var in proj.synapse_type.description['variables']:
            if var['name'] in proj.synapse_type.description['local']:
                tpl_code += """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s""" % {'name': var['name'], 'type': var['ctype']}
            elif var['name'] in proj.synapse_type.description['global']:
                tpl_code += """
        vector[%(type)s] %(name)s
        bool record_%(name)s
""" % {'name': var['name'], 'type': var['ctype']}


        return tpl_code % {'id' : proj.id}

    @staticmethod
    def _proj_monitor_wrapper(proj):
        """
        Generate projection recorder struct
        """
        tpl_code = """
# Projection Monitor wrapper
cdef class ProjRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, long offset):
        self.thisptr = new ProjRecorder%(id)s(ranks, period, offset)
"""

        for var in proj.synapse_type.description['variables']:
            tpl_code += """
    property %(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s = val
    def clear_%(name)s(self):
        (<ProjRecorder%(id)s *>self.thisptr).%(name)s.clear()
""" % {'id' : proj.id, 'name': var['name']}

        return tpl_code % {'id' : proj.id}
