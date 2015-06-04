"""
    OMPGenerator.py
    
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
import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from .PopulationGenerator import PopulationGenerator
from .ProjectionGenerator import ProjectionGenerator

import numpy as np

class OMPGenerator(object):
    """
    Implements the code generator class for OpenMP (and sequential) code.
    OpenMP support is only enabled if the number of threads is higher then one.
    """
    def __init__(self, annarchy_dir, populations, projections, net_id):
        """
        Constructor initializes PopulationGenerator and ProjectionGenerator
        class and stores the provided information for later use.

        Parameters:

            * *net_id*: unique id for the current network
            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *populations*: list of populations
            * *populations*: list of projections
        """
        self._net_id = net_id
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections

        self._popgen = PopulationGenerator()
        self._projgen = ProjectionGenerator()

        self._pop_desc = []
        self._proj_desc = []

    def generate(self):
        """
        Generate code files and store them in target directory (self._annarchy_dir/generate).
        More detailed the following files are generated, by this class:

            * *ANNarchy.cpp*: main simulation loop, object instantiation
            * *ANNarchy.h*: collection of all objects, interface to Python
              extension
            * *ANNarchyCore.pyx*: Python extension file, gathering all functions /
              objects, which should be accessible from Python
            * for each population a seperate header file, contain semantic
              logic of a population respectively neuron object (filename:
              pop<id>).
            * for each projection a seperate header file, contain semantic
              logic of a projection respectively synapse object (filename:
              proj<id>)
        """
        if Global.config['verbose']:
            if Global.config['num_threads'] > 1:
                print('\nGenerate code for OpenMP ...')
            else:
                print('\nGenerate sequential code ...')

        # Propagte the global operations needed by the projections to the
        # corresponding populations.
        self._propagate_global_ops()

        # Create all populations
        for pop in self._populations:
            self._pop_desc.append(self._popgen.header_struct(pop, self._annarchy_dir))

        # Create all projections
        for proj in self._projections:
            self._proj_desc.append(self._projgen.header_struct(proj, self._annarchy_dir))

        # Generate header code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/ANNarchy.h', 'w') as ofile:
            ofile.write(self.generate_header())

        # Generate cpp code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/ANNarchy.cpp', 'w') as ofile:
            ofile.write(self.generate_body())

        # Generate cython code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/ANNarchyCore'+str(self._net_id)+'.pyx', 'w') as ofile:
            ofile.write(self._generate_pyx())

    def _propagate_global_ops(self):
        """
        
        """
        # Analyse the populations
        for pop in self._populations:
            pop.global_operations = pop.neuron_type.description['global_operations']
            pop.delayed_variables = []

        # Propagate the global operations from the projections to the populations
        for proj in self._projections:
            for op in proj.synapse.description['pre_global_operations']:
                if isinstance(proj.pre, PopulationView):
                    if not op in proj.pre.population.global_operations:
                        proj.pre.population.global_operations.append(op)
                else:
                    if not op in proj.pre.global_operations:
                        proj.pre.global_operations.append(op)

            for op in  proj.synapse.description['post_global_operations']:
                if isinstance(proj.post, PopulationView):
                    if not op in proj.post.population.global_operations:
                        proj.post.population.global_operations.append(op)
                else:
                    if not op in proj.post.global_operations:
                        proj.post.global_operations.append(op)
            if proj.max_delay > 1:
                for var in proj.synapse.description['dependencies']['pre']:
                    if isinstance(proj.pre, PopulationView):
                        proj.pre.population.delayed_variables.append(var)
                    else:
                        proj.pre.delayed_variables.append(var)

        # Make sure the operations are declared only once
        for pop in self._populations:
            pop.global_operations = list(np.unique(np.array(pop.global_operations)))
            pop.delayed_variables = list(set(pop.delayed_variables))



#######################################################################
############## HEADER #################################################
#######################################################################
    def generate_header(self):
        """
        Generate the ANNarchy.h code. This header represents the interface to
        the python extension and therefore includes all network objects.
        """
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""

        for pop in self._pop_desc:
            pop_struct += pop['include']
            pop_ptr += pop['extern']

        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self._proj_desc:
            proj_struct += proj['include']
            proj_ptr += proj['extern']

        # Custom functions
        custom_func = self.header_custom_functions()

        # Include OMP
        include_omp = "#include <omp.h>" if Global.config['num_threads'] > 1 else ""

        # Population recorders
        record_classes = self.header_recorder_classes()

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'custom_func': custom_func,
            'include_omp': include_omp,
            'record_classes': record_classes
        }

    def header_custom_functions(self):

        if len(Global._objects['functions']) == 0:
            return ""

        code = ""
        from ANNarchy.parser.Extraction import extract_functions
        for func in Global._objects['functions']:
            code += extract_functions(func, local_global=True)[0]['cpp'] + '\n'

        return code

    def header_recorder_classes(self):
        code = ""
        for pop in self._populations:
            code += self._popgen.recorder_class(pop)
        for proj in self._projections:
            code += self._projgen.recorder_class(proj)

        return code

#######################################################################
############## BODY ###################################################
#######################################################################
    def generate_body(self):
        # struct declaration for each population
        pop_ptr = ""
        for pop in self._pop_desc:
            pop_ptr += pop['instance']

        # struct declaration for each projection
        proj_ptr = ""
        for proj in self._proj_desc:
            proj_ptr += proj['instance']

        # Code for the global operations
        glop_definition = self.body_def_glops()

        # Reset presynaptic sums
        reset_sums = self.body_resetcomputesum_pop()

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()

        # Initialize populations
        pop_init = self.body_init_population()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()

        # Equations for the neural variables
        update_neuron = ""
        for pop in self._pop_desc:
            update_neuron += pop['update']

        # Enque delayed outputs
        delay_code = self.body_delay_neuron()

        # Global operations
        update_globalops = self.body_update_globalops()

        # Equations for the synaptic variables
        update_synapse = ""
        for proj in self._proj_desc:
            update_synapse += proj['update']

        # Equations for the post-events
        post_event = self.body_postevent_proj()

        # Structural plasticity
        structural_plasticity = self.body_structural_plasticity()

        # Early stopping
        run_until = self.body_run_until()

        # Number threads
        number_threads = "omp_set_num_threads(threads);" if Global.config['num_threads'] > 1 else ""

        #Profiling
        from ..Profile.Template import profile_generator_omp_template
        prof_include = "" if not Global.config["profiling"] else profile_generator_omp_template['include']
        prof_init = "" if not Global.config["profiling"] else profile_generator_omp_template['init']
        prof_step_pre = "" if not Global.config["profiling"] else profile_generator_omp_template['step_pre']
        prof_run_pre = "" if not Global.config["profiling"] else profile_generator_omp_template['run_pre']
        prof_run_post = "" if not Global.config["profiling"] else profile_generator_omp_template['run_post']

        # Generate cpp code for the analysed pops and projs
        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'glops_def': glop_definition,
            'run_until': run_until,
            'compute_sums' : compute_sums,
            'reset_sums' : reset_sums,
            'update_neuron' : update_neuron,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse,
            'random_dist_init' : rd_init_code,
            'random_dist_update' : rd_update_code,
            'delay_code' : delay_code,
            'pop_init' : pop_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'structural_plasticity': structural_plasticity,
            'set_number_threads' : number_threads,
            'prof_include': prof_include,
            'prof_init': prof_init,
            'prof_step_pre': prof_step_pre,
            'prof_run_pre': prof_run_pre,
            'prof_run_post': prof_run_post
        }

    def body_delay_neuron(self):
        code = ""
        for pop in self._populations:
            code += self._popgen.delay_code(pop)
        return code

    def body_computesum_proj(self):
        code = ""
        # Sum over all synapses
        for proj in self._projections:
            # Is it a specific projection?
            if proj.generator['omp']['body_compute_psp']:
                code += proj.generator['omp']['body_compute_psp']
                continue

            # Call the comput_psp method
            code += """    proj%(id)s.compute_psp();
""" % {'id' : proj.id}

        return code

    def body_resetcomputesum_pop(self):
        code = ""
        for pop in self._populations:
            if pop.neuron_type.type == 'rate':
                code += self._popgen.reset_computesum(pop)

        return code

    def body_postevent_proj(self):
        code = ""
        for proj in self._projections:
            if proj.synapse.type == 'spike':
                code += self._projgen.postevent(proj)

        return code

    def body_structural_plasticity(self):
        # Pruning if any
        pruning = ""
        creating = ""
        if Global.config['structural_plasticity']:
            for proj in self._projections:
                if 'pruning' in proj.synapse.description.keys():
                    pruning += self._projgen.pruning(proj)
                if 'creating' in proj.synapse.description.keys():
                    creating += self._projgen.creating(proj)

        return creating + pruning

    def body_init_randomdistributions(self):
        code = """
    // Initialize random distribution objects
"""
        for pop in self._populations:
            code += self._popgen.init_random_distributions(pop)

        for proj in self._projections:
            code += self._projgen.init_random_distributions(proj)

        return code

    def body_init_globalops(self):
        code = """
    // Initialize global operations
"""
        for pop in self._populations:
            code += self._popgen.init_globalops(pop)

        return code

    def body_def_glops(self):
        ops = []
        for pop in self._populations:
            for op in pop.global_operations:
                ops.append(op['function'])

        if ops == []:
            return ""

        from .GlobalOperationTemplate import global_operation_templates
        code = ""
        for op in list(set(ops)):
            code += global_operation_templates[op] % {'omp': '' if Global.config['num_threads'] > 1 else "//"}

        return code

    def body_init_delay(self):
        code = ""
        for pop in self._populations:
            if pop.max_delay > 1: # no need to generate the code otherwise
                code += self._popgen.init_delay(pop)

        return code

    def body_init_population(self):
        """
        Generate PopStruct::init_population() calls for the initialize()
        function.
        """
        code = """
    // Initialize populations
"""
        for pop in self._populations:
            code += """    pop%(id)s.init_population();
""" % {'id': pop.id}

        return code

    def body_init_projection(self):
        code = """
    // Initialize projections
"""
        for proj in self._projections:
            code += """    proj%(id)s.init_projection();
""" % {'id' : proj.id}
        return code

    def body_update_randomdistributions(self):
        code = ""
        for pop in self._populations:
            code += self._popgen.update_random_distributions(pop)

        for proj in self._projections:
            code += self._projgen.update_random_distributions(proj)

        return code

    def body_update_globalops(self):
        code = ""
        for pop in self._populations:
            code += self._popgen.update_globalops(pop)
        return code

    def body_run_until(self):
        """
        Generate the code for conditioned stop of simulation
        """
        from .BodyTemplate import run_until_template as tpl

        # Check if it is useful to generate anything at all
        for pop in self._populations:
            if pop.stop_condition:
                break
        else:
            # No stop conditions were detected
            return tpl['default']

        # a condition has been defined, so we generate corresponding code
        cond_code = ""
        for pop in self._populations:
            cond_code += self._popgen.stop_condition(pop)

        return tpl['body'] % {'run_until': cond_code}


#######################################################################
############## PYX ####################################################
#######################################################################
    def _generate_pyx(self):
        """
        Generate the python extension (*.pyx) file comprising of wrapper
        classes for the several objects. Secondly the definition of accessible
        methods, e. g. simulate(int steps).
        """
        # struct declaration for each population
        pop_struct, pop_ptr = self._pyx_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self._pyx_struct_proj()

        # struct declaration for each monitor
        monitor_struct = self._pyx_struct_monitor()

        # Cython wrappers for the populations
        pop_class = self._pyx_wrapper_pop()

        # Cython wrappers for the projections
        proj_class = self._pyx_wrapper_proj()

        # Cython wrappers for the monitors
        monitor_class = self._pyx_wrapper_monitor()

        from .PyxTemplate import pyx_template
        return pyx_template % {
            'pop_struct': pop_struct, 'pop_ptr': pop_ptr,
            'proj_struct': proj_struct, 'proj_ptr': proj_ptr,
            'pop_class' : pop_class, 'proj_class': proj_class,
            'monitor_struct': monitor_struct, 'monitor_wrapper': monitor_class
        }

    def _pyx_struct_pop(self):
        pop_struct = ""
        pop_ptr = ""
        for pop in self._populations:
            # Header export
            pop_struct += self._popgen.pyx_struct(pop)
            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}
        return pop_struct, pop_ptr

    def _pyx_struct_proj(self):
        proj_struct = ""
        proj_ptr = ""
        for proj in self._projections:
            # Header export
            proj_struct += self._projgen.pyx_struct(proj)

            # Projection instance
            proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {
    'id_proj': proj.id,
}
        return proj_struct, proj_ptr

    def _pyx_wrapper_pop(self):
        # Cython wrappers for the populations
        code = ""
        for pop in self._populations:
            code += self._popgen.pyx_wrapper(pop)
        return code

    def _pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        code = ""
        for proj in self._projections:
            code += self._projgen.pyx_wrapper(proj)
        return code

    # Monitors
    def _pyx_struct_monitor(self):
        code = ""
        for pop in self._populations:
            code += self._popgen.pyx_monitor_struct(pop)
        for proj in self._projections:
            code += self._projgen.pyx_monitor_struct(proj)
        return code

    def _pyx_wrapper_monitor(self):
        # Cython wrappers for the populations monitors
        code = ""
        for pop in self._populations:
            code += self._popgen.pyx_monitor_wrapper(pop)
        for proj in self._projections:
            code += self._projgen.pyx_monitor_wrapper(proj)
        return code
