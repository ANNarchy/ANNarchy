"""
    CodeGenerator.py

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
from .PyxGenerator import PyxGenerator
from .RecordGenerator import RecordGenerator

import numpy as np

class CodeGenerator(object):
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

        for pop in self._populations:
            pop._generate()
        for proj in self._projections:
            proj._generate()

        if Global.config['profiling']:
            from .ProfileGenerator import ProfileGenerator
            self._profgen = ProfileGenerator(self._annarchy_dir, net_id)
            self._profgen.generate()
        else:
            self._profgen = None

        self._popgen = PopulationGenerator(self._profgen, net_id)
        self._projgen = ProjectionGenerator(self._profgen, net_id)

        self._pyxgen = PyxGenerator(annarchy_dir, populations, projections, net_id)
        self._recordgen = RecordGenerator(annarchy_dir, populations, projections, net_id)

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
        with open(self._annarchy_dir+'/generate/net'+str(self._net_id)+'/ANNarchy.h', 'w') as ofile:
            ofile.write(self._generate_header())

        # Generate monitor code for the analysed pops and projs
        self._recordgen.generate()

        # Generate cpp code for the analysed pops and projs
        postfix = ".cpp" if Global.config['paradigm']=="openmp" else ".cu"
        with open(self._annarchy_dir+'/generate/net'+str(self._net_id)+'/ANNarchy'+postfix, 'w') as ofile:
            ofile.write(self.generate_body())

        # Generate cython code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/net'+str(self._net_id)+'/ANNarchyCore'+str(self._net_id)+'.pyx', 'w') as ofile:
            ofile.write(self._pyxgen.generate())

    def _propagate_global_ops(self):
        """

        """
        # Analyse the populations
        for pop in self._populations:
            pop.global_operations = pop.neuron_type.description['global_operations']
            pop.delayed_variables = []

        # Propagate the global operations from the projections to the populations
        for proj in self._projections:
            for op in proj.synapse_type.description['pre_global_operations']:
                if isinstance(proj.pre, PopulationView):
                    if not op in proj.pre.population.global_operations:
                        proj.pre.population.global_operations.append(op)
                else:
                    if not op in proj.pre.global_operations:
                        proj.pre.global_operations.append(op)

            for op in  proj.synapse_type.description['post_global_operations']:
                if isinstance(proj.post, PopulationView):
                    if not op in proj.post.population.global_operations:
                        proj.post.population.global_operations.append(op)
                else:
                    if not op in proj.post.global_operations:
                        proj.post.global_operations.append(op)
            if proj.max_delay > 1:
                for var in proj.synapse_type.description['dependencies']['pre']:
                    if isinstance(proj.pre, PopulationView):
                        proj.pre.population.delayed_variables.append(var)
                    else:
                        proj.pre.delayed_variables.append(var)

        # Make sure the operations are declared only once
        for pop in self._populations:
            pop.global_operations = [dict(y) for y in set(tuple(x.items()) for x in pop.global_operations)]
            pop.delayed_variables = list(set(pop.delayed_variables))


#######################################################################
############## HEADER #################################################
#######################################################################
    def _generate_header(self):
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

        if Global.config['paradigm']=="openmp":
            from .Template.BaseTemplate import omp_header_template
            return omp_header_template % {
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'custom_func': custom_func,
                'include_omp': include_omp
            }
        else:
            from Template.BaseTemplate import cuda_header_template
            return cuda_header_template % {
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr
            }

    def header_custom_functions(self):

        if len(Global._objects['functions']) == 0:
            return ""

        code = ""
        from ANNarchy.parser.Extraction import extract_functions
        for func in Global._objects['functions']:
            code += extract_functions(func, local_global=True)[0]['cpp'] + '\n'

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
        update_globalops = ""
        for pop in self._pop_desc:
            if 'gops_update' in pop.keys():
                update_globalops += pop['gops_update']

        # Reset presynaptic sums
        reset_sums = self.body_resetcomputesum_pop()

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Update random distributions
        rd_update_code = ""
        for desc in self._pop_desc + self._proj_desc:
            if 'rng_update' in desc.keys():
                rd_update_code += desc['rng_update']

        # Equations for the neural variables
        update_neuron = ""
        for pop in self._pop_desc:
            if 'update' in pop.keys():
                update_neuron += pop['update']

        # Enque delayed outputs
        delay_code = ""
        for pop in self._pop_desc:
            if 'delay_update' in pop.keys():
                delay_code += pop['delay_update']

        # Equations for the synaptic variables
        update_synapse = ""
        for proj in self._proj_desc:
            if 'update' in proj.keys():
                update_synapse += proj['update']

        # Equations for the post-events
        post_event = ""
        for proj in self._proj_desc:
            if 'post_event' in proj.keys():
                post_event += proj['post_event']

        # Structural plasticity
        structural_plasticity = self.body_structural_plasticity()

        # Early stopping
        run_until = self.body_run_until()

        # Number threads
        number_threads = "omp_set_num_threads(threads);" if Global.config['num_threads'] > 1 else ""

        #Profiling
        if self._profgen:
            prof_dict = self._profgen.generate_body_dict()
        else:
            from .ProfileGenerator import ProfileGenerator
            prof_dict = ProfileGenerator(self._annarchy_dir, self._net_id).generate_body_dict(True)

        #
        # Generate the ANNarchy.cpp code, the corrsponding template differs greatly
        # for further information take a look into the corresponding branch
        #

        # Generate cpp code for the analysed pops and projs
        if Global.config['paradigm']=="openmp":
            from .Template.BaseTemplate import omp_body_template
            base_dict = {
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'glops_def': glop_definition,
                'initialize': self._body_initialize(),
                'run_until': run_until,
                'compute_sums' : compute_sums,
                'reset_sums' : reset_sums,
                'update_neuron' : update_neuron,
                'update_globalops' : update_globalops,
                'update_synapse' : update_synapse,
                'random_dist_update' : rd_update_code,
                'delay_code' : delay_code,
                'post_event' : post_event,
                'structural_plasticity': structural_plasticity,
                'set_number_threads' : number_threads,
            }

            base_dict.update( prof_dict )
            return omp_body_template % base_dict
        else:
            # Implementation notice ( HD: 10. June, 2015 )
            #
            # The CUDA linking process is a big problem for object oriented approaches
            # and the seperation of implementation codes into several files. Even in the
            # current SDK 5.0 this problem is not fully solved. Linking is available, but
            # only for small, independent code pieces, by far not sufficient for full
            # object-oriented approaches ...
            #
            # For us, this currently have one consequence: we cannot completely seperate
            # the implementation of objects into several files. To hold a certain equality
            # between the structures of objects, I implemented the following workaround:
            #
            # We create the c-structs holding data fields and accessors as in OpenMP. We also
            # create the kernels, call entity in the corresponding generator objects, and
            # return the codes via the descriptor dictionary.
            #
            # This ensures a consistent interface in the generators and also in the generated
            # codes, but sometimes require additional overhead. Hopefully NVidia will improve
            # their linker in the next releases, so one could remove this overhead.
            psp_call = ""
            for proj in self._proj_desc:
                psp_call += proj['psp_call']

            pop_kernel = ""
            for pop in self._pop_desc:
                pop_kernel += pop['update_body']

            psp_kernel = ""
            for proj in self._proj_desc:
                psp_kernel += proj['psp_body']

            kernel_def = ""
            for pop in self._pop_desc:
                kernel_def += pop['update_header']
            for proj in self._proj_desc:
                kernel_def += proj['psp_header']
                kernel_def += proj['update_synapse_header']

            delay_code = ""
            for pop in self._pop_desc:
                if 'update_delay' in pop.keys():
                    delay_code += pop['update_delay']

            syn_kernel = ""
            for proj in self._proj_desc:
                syn_kernel += proj['update_synapse_body']

            syn_call = ""
            for proj in self._proj_desc:
                syn_call += proj['update_synapse_call']

            # global operations
            glob_ops_header, glob_ops_body = self.body_def_glops()
            kernel_def += glob_ops_header

            # determine number of threads per kernel and concurrent kernel execution
            threads_per_kernel, stream_setup = self._cuda_kernel_config()
            host_device_transfer, device_host_transfer = "", ""
            for pop in self._pop_desc + self._proj_desc:
                host_device_transfer += pop['host_to_device']
                device_host_transfer += pop['device_to_host']

            from Template.BaseTemplate import cuda_body_template
            return cuda_body_template % {
                # network definitions
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'run_until': run_until,
                'compute_sums' : psp_call,
                'update_neuron' : update_neuron,
                'update_globalops' : update_globalops,
                'update_synapse' : syn_call,
                'delay_code': delay_code,
                'initialize' : self._body_initialize(),
                'post_event' : post_event,
                'structural_plasticity': structural_plasticity,

                # cuda host specific
                'stream_setup': stream_setup,
                'host_device_transfer': host_device_transfer,
                'device_host_transfer': device_host_transfer,
                'kernel_def': kernel_def,

                #device stuff
                'kernel_config': threads_per_kernel,
                'pop_kernel': pop_kernel, #update_neuron_body,
                'psp_kernel': psp_kernel,
                'syn_kernel': syn_kernel, #update_synapse_body,
                'glob_ops_kernel': glob_ops_body,
                'custom_func': "", #custom_func
            }


    def _body_initialize(self):
        """
        Define codes for the method initialize(), comprising of population and projection
        initializations, optionally profiling class.
        """
        from .Template.ProfileTemplate import profile_template
        profiling_init = "" if not Global.config["profiling"] else profile_template['init']

        # Initialize populations
        population_init = "    // Initialize populations\n"
        for pop in self._pop_desc:
            population_init += pop['init']

        # Initialize projections
        projection_init = "    // Initialize projections\n"
        for proj in self._proj_desc:
            projection_init += proj['init']

        if Global.config['paradigm']=="openmp":
            from .Template.BaseTemplate import omp_initialize_template as init_tpl
        else:
            from .Template.BaseTemplate import cuda_initialize_template as init_tpl

        return init_tpl % { 'prof_init': profiling_init,
                            'pop_init': population_init,
                            'proj_init': projection_init
                          }

    def body_computesum_proj(self):
        code = ""
        # Sum over all synapses
        for proj in self._projections:
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

    def body_structural_plasticity(self):
        # Pruning if any
        pruning = ""
        creating = ""
        if Global.config['structural_plasticity']:
            for proj in self._projections:
                if 'pruning' in proj.synapse_type.description.keys():
                    pruning += self._projgen.pruning(proj)
                if 'creating' in proj.synapse_type.description.keys():
                    creating += self._projgen.creating(proj)

        return creating + pruning

    def body_def_glops(self):
        """
        Dependent on the used global operations we add pre-defined templates
        to the ANNarchy body file.

        Return:

            dependent on the used paradigm we return one string (OpenMP)
            or tuple(string, string) (CUDA).
        """
        ops = []
        for pop in self._populations:
            for op in pop.global_operations:
                ops.append(op['function'])

        if Global.config['paradigm'] == "openmp":
            if ops == []:
                return ""

            from .Template.GlobalOperationTemplate import global_operation_templates_openmp as omp_template
            code = ""
            for op in list(set(ops)):
                code += omp_template[op] % {'omp': '' if Global.config['num_threads'] > 1 else "//"}

            return code
        else:
            if ops == []:
                return "", ""

            header = ""
            body = ""

            from .Template.GlobalOperationTemplate import global_operation_templates_cuda as cuda_template
            for op in list(set(ops)):
                header += cuda_template[op]['header']
                body += cuda_template[op]['body']

            return header, body

    def body_run_until(self):
        """
        Generate the code for conditioned stop of simulation
        """
        from .Template.BaseTemplate import omp_run_until_template as tpl

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
            if pop.stop_condition:
                cond_code += tpl['single_pop'] % {'id': pop.id}

        return tpl['body'] % {'run_until': cond_code}

#######################################################################
############## CUDA specific ##########################################
#######################################################################
    def _cuda_kernel_config(self):
        """
        Generates a kernel config and stream setup (if a device with compute
        compability > 2.x available. The kernel configuration is needed for
        the device code, to have number of threads and blocks for calling
        device functions. Stream setup is for the concurrent kernel
        execution. Please note, that these parameter must be modified
        through Global.cuda_config dictionary, otherwise default values (no
        stream, 192 threads for psp, 32 threads for neurons) are used.

        Notice:

            Only related to the CUDA implementation
        """
        cu_config = Global.cuda_config

        code = "// Population config\n"
        for pop in self._populations:
            num_threads = 32
            if pop in cu_config.keys():
                num_threads = cu_config[pop]['num_threads']

            code+= """#define __pop%(id)s__ %(nr)s\n""" % { 'id': pop.id, 'nr': num_threads }

        code += "\n// Population config\n"
        for proj in self._projections:
            num_threads = 192
            if proj in cu_config.keys():
                num_threads = cu_config[proj]['num_threads']

            code+= """#define __pop%(pre)s_pop%(post)s_%(target)s__ %(nr)s\n""" % { 'pre': proj.pre.id, 'post': proj.post.id, 'target': proj.target, 'nr': num_threads }

        pop_assign = "    // populations\n"
        for pop in self._populations:
            if pop in Global.cuda_config.keys():
                pop_assign += """    pop%(pid)s.stream = streams[%(sid)s];
""" % {'pid': pop.id, 'sid': Global.cuda_config[pop]['stream'] }
            else:
                # default stream
                pop_assign += """    pop%(pid)s.stream = 0;
""" % {'pid': pop.id }

        proj_assign = "    // populations\n"
        for proj in self._projections:
            if proj in Global.cuda_config.keys():
                proj_assign += """    proj%(pid)s.stream = streams[%(sid)s];
""" % {'pid': proj.id, 'sid': Global.cuda_config[proj]['stream'] }
            else:
                # default stream
                proj_assign += """    proj%(pid)s.stream = 0;
""" % {'pid': proj.id }

        from Template.BaseTemplate import cuda_stream_setup
        stream_config = cuda_stream_setup % {
            'nbStreams': 2,
            'pop_assign': pop_assign,
            'proj_assign': proj_assign
        }

        return code, stream_config
