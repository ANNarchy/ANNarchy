"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import time

from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern import Messages
from ANNarchy.parser.Extraction import extract_functions

from ANNarchy.generator.PyxGenerator import PyxGenerator
from ANNarchy.generator.Monitor.MonitorGenerator import MonitorGenerator
from ANNarchy.generator.Population import SingleThreadGenerator, OpenMPGenerator, CUDAGenerator
from ANNarchy.generator.Projection import SingleThreadProjectionGenerator, OpenMPProjectionGenerator, CUDAProjectionGenerator
from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_st, global_operation_templates_openmp, global_operation_templates_cuda
from ANNarchy.generator.Utils import tabify
from ANNarchy.generator.Template import BaseTemplate
from ANNarchy.generator import Profile

class CodeGenerator(object):
    """
    The CodeGenerator class is responsible to control the code
    generation process.

    Unil now, it implements the code generation for OpenMP
    (including sequential) and CUDA.The decision whether as
    OpenMP or sequential code is dependent on the number of
    threads.
    """
    def __init__(self, annarchy_dir, populations, projections, net_id, cuda_config):
        """
        Constructor initializes the PopulationGenerator and ProjectionGenerator
        class and stores the provided information for later use.

        Parameters:

            * *net_id*: unique id for the current network
            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *populations*: list of populations
            * *projections*: list of projections
            * *cuda_config*: configuration dict for cuda. check the method
              _cuda_kernel_config for more details.
        """
        self._net_id = net_id
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._cuda_config = cuda_config

        # Profiling is optional, but if either Global.config["profiling"] set to True
        # or --profile was added on command line.
        if get_global_config('profiling'):
            if get_global_config('paradigm') == "openmp":
                self._profgen = Profile.CPP11Profile(self._annarchy_dir, net_id)
                self._profgen.generate()
            elif get_global_config('paradigm') == "cuda":
                self._profgen = Profile.CUDAProfile(self._annarchy_dir, net_id)
                self._profgen.generate()
            else:
                Messages._error('No ProfileGenerator available for '
                              + get_global_config('paradigm'))
        else:
            self._profgen = None

        # Instantiate code generator based on the target platform
        if get_global_config('paradigm') == "openmp":
            if get_global_config('num_threads') == 1:
                self._popgen = SingleThreadGenerator(self._profgen, net_id)
                self._projgen = SingleThreadProjectionGenerator(self._profgen, net_id)
            else:
                self._popgen = OpenMPGenerator(self._profgen, net_id)
                self._projgen = OpenMPProjectionGenerator(self._profgen, net_id)
        elif get_global_config('paradigm') == "cuda":
            self._popgen = CUDAGenerator(self._cuda_config['cuda_version'], self._profgen, net_id)
            self._projgen = CUDAProjectionGenerator(self._cuda_config['cuda_version'], self._profgen, net_id)
        else:
            Messages._error("No PopulationGenerator for " + get_global_config('paradigm'))

        # Py-extenstion and RecordGenerator are commonly defined
        self._pyxgen = PyxGenerator(annarchy_dir, populations, projections, net_id)
        self._recordgen = MonitorGenerator(annarchy_dir, populations, projections, net_id)

        # Target container for the generated code snippets
        self._pop_desc = []
        self._proj_desc = []

    def generate(self):
        """
        Generate code files and store them in target directory (located at
        self._annarchy_dir/generate). More detailed the following files are
        generated, by this class:

            * *ANNarchy.cpp*: main simulation loop, object instantiation
            * *ANNarchy.hpp*: collection of all objects, interface to Python
              extension
            * *ANNarchyCore.pyx*: Python extension file, gathering all
               functions/ objects, which should be accessible from Python
            * for each population a seperate header file, contain semantic
              logic of a population respectively neuron object (filename:
              pop<id>).
            * for each projection a seperate header file, contain semantic
              logic of a projection respectively synapse object (filename:
              proj<id>)
        """
        if Profiler().enabled:
            t0 = time.time()

        if get_global_config('verbose'):
            if get_global_config('paradigm') == "openmp":
                if get_global_config('num_threads') > 1:
                    Messages._print('\nGenerate code for OpenMP ...')
                else:
                    Messages._print('\nGenerate sequential code ...')
            elif get_global_config('paradigm') == "cuda":
                print('\nGenerate CUDA code ...')
            else:
                raise NotImplementedError

        # Specific populations/projections have an overwritten _generate()
        # method which will populate the self._specific_template dictionary
        for pop in self._populations:
            pop._generate()
        for proj in self._projections:
            proj._generate()

        # Propagate the global operations needed by the projections to the
        # corresponding populations.
        self._propagate_global_ops()

        # Create all populations
        for pop in self._populations:
            self._pop_desc.append(self._popgen.header_struct(pop, self._annarchy_dir))

        # Create all projections
        for proj in self._projections:
            self._proj_desc.append(self._projgen.header_struct(proj, self._annarchy_dir))

        # where all source files should take place
        source_dest = self._annarchy_dir+'/generate/net'+str(self._net_id)+'/'

        # Generate header code for the analysed pops and projs
        if get_global_config('paradigm') == "openmp":
            with open(source_dest+'ANNarchy.hpp', 'w') as ofile:
                ofile.write(self._generate_header())

        elif get_global_config('paradigm') == "cuda":
            invoke_header, host_header = self._generate_header()
            with open(source_dest+'ANNarchyKernel.cuh', 'w') as ofile:
                ofile.write(invoke_header)
            with open(source_dest+'ANNarchy.hpp', 'w') as ofile:
                ofile.write(host_header)

        else:
            raise NotImplementedError

        # Generate monitor code for the analysed pops and projs
        self._recordgen.generate()

        # Generate cpp code for the analysed pops and projs
        if get_global_config('paradigm') == "openmp":
            with open(source_dest+'ANNarchy.cpp', 'w') as ofile:
                ofile.write(self._generate_body())

        elif get_global_config('paradigm') == "cuda":
            device_code, host_code = self._generate_body()
            with open(source_dest+'ANNarchy.cpp', 'w') as ofile:
                ofile.write(host_code)
            with open(source_dest+'ANNarchyKernel.cu', 'w') as ofile:
                ofile.write(device_code)

        else:
            raise NotImplementedError

        # Generate cython code for the analysed pops and projs
        with open(source_dest+'ANNarchyCore'+str(self._net_id)+'.pyx', 'w') as ofile:
            ofile.write(self._pyxgen.generate())

        self._generate_file_overview(source_dest)

        if Profiler().enabled:
            t1 = time.time()
            Profiler().add_entry(t0, t1, "generate", "compile")

    def _generate_file_overview(self, source_dest):
        """
        Generate a logfile, where we log which Population/Projection object is stored in
        which file.

        Parameters:

        * source_dest: path to folder where generated files are stored.
        """
        # Equal to target path in CodeGenerator.generate()
        with open(source_dest+"codegen.log", 'w') as ofile:
            ofile.write("Filename, Object Description\n")
            for pop in self._populations:
                pop_type = type(pop).__name__
                desc = """pop%(id_pop)s, %(type_pop)s( name = %(name_pop)s )\n""" % {
                    'id_pop': pop.id, 'name_pop': pop.name, 'type_pop': pop_type
                }
                ofile.write(desc)

            for proj in self._projections:
                proj_type = type(proj).__name__
                desc_dict = {
                    'id_proj': proj.id,
                    'type_proj': proj_type,
                    'pre_name': proj.pre.name,
                    'post_name': proj.post.name,
                    'target': proj.target,
                    'name': proj.name
                }

                # In case of debug, we print the parameters otherwise not
                if get_global_config('debug'):
                    desc_dict.update({'pattern': proj.connector_description})
                else:
                    desc_dict.update({'pattern': proj.connector_description.split(',')[0]})

                desc = desc = """proj%(id_proj)s, %(type_proj)s( pre = %(pre_name)s, post = %(post_name)s, target = %(target)s, name = %(name)s ) using connector: %(pattern)s \n""" % desc_dict
                ofile.write(desc)

    def _propagate_global_ops(self):
        """
        The parser analyses the synapse and neuron definitions and
        store if global operations like min, max or mean are necessary.

        Furthermore for synapses accesses to population variales (e. g. pre.r)
        occure. In this case we need to generate special codes in the PopulationGenerator.
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
            pop.global_operations = [dict(y) for y in sorted(set(tuple(x.items())) for x in pop.global_operations)]
            pop.delayed_variables = sorted(list(set(pop.delayed_variables)))

    def _generate_header(self):
        """
        Generate the ANNarchy.hpp code. This header represents the interface to
        the Python extension and therefore includes all network objects.
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
        custom_func = self._header_custom_functions()

        # Custom constants
        custom_constant = self._header_custom_constants()

        # Final code
        header_code = ""
        if get_global_config('paradigm') == "openmp":
            header_code = BaseTemplate.omp_header_template % {
                'float_prec': get_global_config('precision'),
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'custom_func': custom_func,
                'custom_constant': custom_constant,
                'built_in': BaseTemplate.built_in_functions + BaseTemplate.integer_power_cpu % {'float_prec': get_global_config('precision')},
            }
            return header_code

        elif get_global_config('paradigm') == "cuda":
            # kernel declaration
            invoke_kernel_def = ""
            for pop in self._pop_desc:
                invoke_kernel_def += pop['update_header']

            for proj in self._proj_desc:
                invoke_kernel_def += proj['psp_kernel_decl']
                invoke_kernel_def += proj['update_synapse_header']
                invoke_kernel_def += proj['postevent_header']

            glob_ops_header, _, _ = self._body_def_glops()
            invoke_kernel_def += glob_ops_header

            device_invoke_header = BaseTemplate.cuda_device_invoke_header % {
                'float_prec': get_global_config('precision'),
                'invoke_kernel_def': invoke_kernel_def
            }

            host_header_code = BaseTemplate.cuda_header_template % {
                'float_prec': get_global_config('precision'),
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'custom_func': custom_func,
                'built_in': BaseTemplate.built_in_functions,
                'custom_constant': custom_constant
            }
            return device_invoke_header, host_header_code
        
        else:
            raise NotImplementedError

    def _header_custom_functions(self):
        """
        Generate code for custom functions defined globally and are usable
        witihn neuron or synapse descriptions. These functions can only rely on
        provided arguments.
        """
        if GlobalObjectManager().number_functions() == 0:
            return ""

        # Attention CUDA: this definition will work only on host side.
        code = ""
        for _, func in GlobalObjectManager().get_functions():
            code += extract_functions(func, local_global=True)[0]['cpp'] + '\n'

        return code

    def _header_custom_constants(self):
        """
        Generate code for custom constants
        """
        if GlobalObjectManager().number_constants() == 0:
            return ""

        code = ""
        for obj in GlobalObjectManager().get_constants():
            obj_str = {
                'name': obj.name,
                'float_prec': get_global_config('precision')
            }
            if _check_paradigm("openmp"):
                code += """
extern %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value);""" % obj_str
            elif _check_paradigm("cuda"):
                code += """
void set_%(name)s(%(float_prec)s value);""" % obj_str
            else:
                raise NotImplementedError

        return code

    def _body_custom_constants(self):
        """
        Generate code for custom constants dependent on the target paradigm
        set in global settings.

        Returns (openMP):

        * decl_code: declarations in header file
        * init_code: initialization code

        Returns (CUDA):

        * device_decl_code: declarations in header file (device side)
        * host_init_code: initialization code (host side)

        """
        if _check_paradigm("openmp"):
            if GlobalObjectManager().number_constants() == 0:
                return "", ""

            decl_code = ""
            init_code = ""
            for obj in GlobalObjectManager().get_constants():
                obj_str = {
                    'name': obj.name,
                    'value': obj.value,
                    'float_prec': get_global_config('precision')
                }
                decl_code += """
%(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value){%(name)s = value;};""" % obj_str
                init_code += """
        %(name)s = 0.0;""" % obj_str

            return decl_code, init_code

        elif _check_paradigm("cuda"):
            if GlobalObjectManager().number_constants() == 0:
                return "", ""

            host_init_code = ""
            device_decl_code = ""
            for obj in GlobalObjectManager().get_constants():
                obj_str = {
                    'name': obj.name,
                    'value': obj.value,
                    'float_prec': get_global_config('precision')
                }
                device_decl_code += """__device__ __constant__ %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value) {
    cudaError_t err = cudaMemcpyToSymbol(%(name)s, &value, sizeof(%(float_prec)s), 0, cudaMemcpyHostToDevice);
#ifdef _DEBUG
    std::cout << "set global constant %(name)s = " << value << std::endl;
    if ( err != cudaSuccess )
        std::cerr << cudaGetErrorString(err) << std::endl;
#endif
}""" % obj_str

                # TODO: is this really needed, it's overwritten anyways ?
                host_init_code += """
        set_%(name)s(0.0);""" % obj_str

            return device_decl_code, host_init_code

        else:
            raise NotImplementedError

    def _generate_body(self):
        """
        Generate the codes 'main' library file. The generated code
        will be used in different files, dependent on the chosen
        target platform:

        * openmp: ANNarchy.cpp
        * cuda: ANNarchyHost.cu and ANNarchyDevice.cu
        """
        # struct declaration for each population
        pop_ptr = ""
        for pop in self._pop_desc:
            pop_ptr += pop['instance']

        # struct declaration for each projection
        proj_ptr = ""
        for proj in self._proj_desc:
            proj_ptr += proj['instance']

        # Code for the global operations
        glop_definition = self._body_def_glops()
        update_globalops = ""
        for pop in self._pop_desc:
            if 'gops_update' in pop.keys():
                update_globalops += pop['gops_update']

        # Reset presynaptic sums
        reset_sums = self._body_resetcomputesum_pop()

        # Compute presynaptic sums
        compute_sums = ""
        # Sum over all synapses
        if _check_paradigm("openmp"):
            for proj in self._proj_desc:
                compute_sums += proj["compute_psp"]

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
        structural_plasticity, sp_spike_backward_view_update = self._body_structural_plasticity()

        # Early stopping
        run_until = self._body_run_until()

        #Profiling
        if self._profgen:
            prof_dict = self._profgen.generate_body_dict()
        else:
            prof_dict = Profile.ProfileGenerator(self._annarchy_dir, self._net_id).generate_body_dict()

        #
        # Generate the ANNarchy.cpp code, the corrsponding template differs
        # greatly. For further information take a look into the corresponding
        # branches.
        #
        if get_global_config('paradigm') == "openmp":
            # custom constants
            custom_constant, _ = self._body_custom_constants()

            # code fields for openMP/single thread template
            base_dict = {
                'float_prec': get_global_config('precision'),
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
                'custom_constant': custom_constant,
                'sp_spike_backward_view_update': sp_spike_backward_view_update
            }

            # profiling
            base_dict.update(prof_dict)

            # complete code template
            if get_global_config('num_threads') == 1:
                return BaseTemplate.st_body_template % base_dict
            else:
                return BaseTemplate.omp_body_template % base_dict

        elif get_global_config('paradigm') == "cuda":
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
                psp_call += proj['psp_host_call']

            # custom constants
            device_custom_constant, _ = self._body_custom_constants()

            # custom functions
            custom_func = ""
            for pop in self._pop_desc:
                custom_func += pop['custom_func']
            for proj in self._proj_desc:
                custom_func += proj['custom_func']
            for _, func in GlobalObjectManager().get_functions():
                custom_func += extract_functions(func, local_global=True)[0]['cpp'].replace("inline", "__device__") + '\n'

            # pre-defined/common available kernel
            common_kernel = self._cuda_common_kernel(self._projections)

            pop_kernel = ""
            pop_invoke_kernel = ""
            for pop in self._pop_desc:
                pop_kernel += pop['update_body']
                pop_invoke_kernel += pop['update_invoke']

            pop_update_fr = ""
            for pop in self._pop_desc:
                pop_update_fr += pop['update_FR']

            psp_device_kernel = ""
            psp_invoke_kernel = ""
            for proj in self._proj_desc:
                psp_device_kernel += proj['psp_device_kernel']
                psp_invoke_kernel += proj['psp_invoke_kernel']

            delay_code = ""
            for pop in self._pop_desc:
                if 'update_delay' in pop.keys():
                    delay_code += pop['update_delay']

            syn_kernel = ""
            syn_invoke_kernel = ""
            for proj in self._proj_desc:
                syn_kernel += proj['update_synapse_body']
                syn_invoke_kernel += proj['update_synapse_invoke']

            syn_call = ""
            for proj in self._proj_desc:
                syn_call += proj['update_synapse_call']

            postevent_device_kernel = ""
            postevent_invoke_kernel = ""
            for proj in self._proj_desc:
                postevent_device_kernel += proj['postevent_body']
                postevent_invoke_kernel += proj['postevent_invoke']

            postevent_call = ""
            for proj in self._proj_desc:
                postevent_call += proj['postevent_call']

            clear_sums = self._body_resetcomputesum_pop()

            # global operations
            _, glob_ops_invoke, glob_ops_body = self._body_def_glops()

            # determine number of threads per kernel
            threads_per_kernel = self._cuda_kernel_config()

            # concurrent kernel execution
            stream_setup = self._cuda_stream_config()

            # memory transfers
            host_device_transfer, device_host_transfer = "", ""
            for pop in self._pop_desc + self._proj_desc:
                host_device_transfer += pop['host_to_device']
                device_host_transfer += pop['device_to_host']

            # Profiling
            if self._profgen:
                prof_dict = self._profgen.generate_body_dict()
            else:
                prof_dict = Profile.ProfileGenerator(self._annarchy_dir, self._net_id).generate_body_dict()

            device_code = BaseTemplate.cuda_device_kernel % {      # Target: ANNarchyKernel.cu
                'common_kernel': common_kernel,
                'pop_kernel': pop_kernel,
                'pop_invoke_kernel': pop_invoke_kernel,
                'psp_kernel': psp_device_kernel,
                'psp_invoke_kernel': psp_invoke_kernel,
                'syn_kernel': syn_kernel,
                'syn_invoke_kernel': syn_invoke_kernel,
                'glob_ops_kernel': glob_ops_body,
                'glob_ops_invoke_kernel': glob_ops_invoke,
                'postevent_kernel': postevent_device_kernel,
                'postevent_invoke_kernel': postevent_invoke_kernel,
                'custom_func': custom_func,
                'custom_constant': device_custom_constant,
                'built_in': BaseTemplate.built_in_functions + BaseTemplate.integer_power_cuda % {'float_prec': get_global_config('precision')},
                'float_prec': get_global_config('precision')
            }

            base_dict = {
                # network definitions
                'float_prec': get_global_config('precision'),
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'run_until': run_until,
                'clear_sums': clear_sums,
                'compute_sums' : psp_call,
                'update_neuron' : update_neuron,
                'update_FR': pop_update_fr,
                'update_globalops' : update_globalops,
                'update_synapse' : syn_call,
                'post_event': postevent_call,
                'delay_code': delay_code,
                'initialize' : self._body_initialize(),
                'structural_plasticity': structural_plasticity,

                # cuda host specific
                'stream_setup': stream_setup,
                'host_device_transfer': host_device_transfer,
                'device_host_transfer': device_host_transfer,
                'kernel_config': threads_per_kernel,
                'sp_spike_backward_view_update': ""
            }
            base_dict.update(prof_dict)
            host_code = BaseTemplate.cuda_host_body_template % base_dict    # Target: ANNarchy.cpp

            return device_code, host_code
        else:
            raise NotImplementedError

    def _body_initialize(self):
        """
        Define codes for the method initialize(), comprising of population and projection
        initializations, optionally profiling class.
        """
        profiling_init = "" if not get_global_config('profiling') else self._profgen.generate_init_network()

        # Initialize populations
        population_init = "    // Initialize populations\n"
        for pop in self._pop_desc:
            population_init += pop['init']

        # Initialize projections
        projection_init = "    // Initialize projections\n"
        for proj in self._proj_desc:
            projection_init += proj['init']

        # Initialize custom constants
        if get_global_config('paradigm') == "openmp":
            # Custom  constants
            _, custom_constant = self._body_custom_constants()

            init_tpl = BaseTemplate.omp_initialize_template
        elif get_global_config('paradigm') == "cuda":
            # Custom  constants
            _, custom_constant = self._body_custom_constants()

            init_tpl = BaseTemplate.cuda_initialize_template
        else:
            raise NotImplementedError

        return init_tpl % {
            'prof_init': profiling_init,
            'pop_init': population_init,
            'proj_init': projection_init,
            'custom_constant': custom_constant
        }

    def _body_resetcomputesum_pop(self):
        """
        Rate-coded neurons sum up the received inputs in temporary variables.
        They need to be cleared in each time step.
        """
        code = ""
        for pop in self._populations:
            if pop.neuron_type.type == 'rate':
                code += self._popgen.reset_computesum(pop)

        return code

    def _body_structural_plasticity(self):
        """
        Call of pruning or creating methods if necessary.

        Returns two strings:
            * call statements called within singleStep()
            * call statements called at begin of simulation loop
        """
        # Pruning if any
        pruning = ""
        creating = ""
        rebuild_in_cpp = ""
        rebuild_out_cpp = ""

        if get_global_config('structural_plasticity'):
            for proj in self._projections:
                rebuild_needed = False
                if 'pruning' in proj.synapse_type.description.keys():
                    pruning += tabify("proj%(id)s.pruning();\n" % {'id': proj.id}, 1)
                    rebuild_needed = True
                if 'creating' in proj.synapse_type.description.keys():
                    creating += tabify("proj%(id)s.creating();\n" % {'id': proj.id}, 1)
                    rebuild_needed = True
                # we only check those projections which are possibly modified
                if rebuild_needed and proj.synapse_type.type == 'spike':
                    rebuild_in_cpp += tabify("proj%(id)s.check_and_rebuild_inverse_connectivity();\n" % {'id': proj.id}, 1)

                # we don't know which projection the user modifies, so we need to check all
                if proj.synapse_type.type == 'spike':
                    rebuild_out_cpp += tabify("proj%(id)s.check_and_rebuild_inverse_connectivity();\n" % {'id': proj.id}, 1)

        return creating + pruning + rebuild_in_cpp, rebuild_out_cpp

    def _body_def_glops(self):
        """
        Dependent on the used global operations we add pre-defined templates
        to the ANNarchy body file.

        Return:

            dependent on the used paradigm we return one string (single thread, OpenMP)
            or tuple(string, string) (CUDA).
        """
        ops = []
        for pop in self._populations:
            for op in pop.global_operations:
                ops.append(op['function'])

        # no global operations
        if ops == []:
            if _check_paradigm("openmp"):
                return ""
            elif _check_paradigm("cuda"):
                return "", "", ""
            else:
                raise NotImplementedError("CodeGenerator._body_def_glops(): no implementation for "+get_global_config('paradigm'))

        type_def = {
            'type': get_global_config('precision')
        }

        # the computation kernel depends on the paradigm
        if _check_paradigm("openmp"):
            if get_global_config('num_threads') == 1:
                global_op_template = global_operation_templates_st
            else:
                global_op_template = global_operation_templates_openmp

            code = ""
            for op in sorted(list(set(ops))):
                code += global_op_template[op] % type_def

            return code

        elif _check_paradigm("cuda"):
            header = ""
            invoke = ""
            body = ""

            for op in sorted(list(set(ops))):
                header += global_operation_templates_cuda[op]['header'] % type_def
                invoke += global_operation_templates_cuda[op]['invoke'] % type_def
                body += global_operation_templates_cuda[op]['body'] % type_def

            return header, invoke, body
        else:
            raise NotImplementedError("CodeGenerator._body_def_glops(): no implementation for "+get_global_config('paradigm'))

    def _body_run_until(self):
        """
        Generate the code for conditioned stop of simulation
        """
        tpl = BaseTemplate.omp_run_until_template

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
        Each GPU kernel requires a launch configuration, established in
        the ANNarchyHost.cu code. Until ANNarchy 4.7 we always defined
        the configuration as pre-processor symbol. In context of variations
        required by different formats, we changed the strategy and this
        symbols are only used IF the user wants to overwrite something.

        * 192 threads for psp and synapse update
        * guessed amount of threads for neurons, based on population size
          (see _guess_pop_kernel_config)

        Notice:

            Only related to the CUDA implementation
        """
        if self._cuda_config is None:
            return ""

        from math import ceil

        # Population config adjust neuron_update
        configuration = "// Populations\n"
        for pop in self._populations:
            if pop in self._cuda_config.keys():
                if 'num_threads' in self._cuda_config[pop].keys():
                    num_threads = self._cuda_config[pop]['num_threads']
                    num_blocks = int(ceil(float(pop.size)/float(num_threads)))

                if 'num_blocks' in self._cuda_config[pop].keys():
                    num_blocks = self._cuda_config[pop]['num_blocks']

                cfg = """#define __pop%(id)s_tpb__ %(nr)s
#define __pop%(id)s_nb__ %(nb)s
"""
                configuration += cfg % {
                    'id': pop.id,
                    'nr': num_threads,
                    'nb': num_blocks
                }

                if get_global_config('verbose'):
                    Messages._print('population', pop.id, ' - kernel config: (', num_blocks, ',', num_threads, ')')

        # Projection config - adjust psp, synapse_local_update, synapse_global_update
        configuration += "\n// Projections\n"
        for proj in self._projections:
            if proj in self._cuda_config.keys():
                if 'num_threads' in self._cuda_config[proj].keys():
                    num_threads = self._cuda_config[proj]['num_threads']
                if 'num_blocks' in self._cuda_config[proj].keys():
                    num_blocks = self._cuda_config[proj]['num_blocks']

                cfg = """#define __proj%(id_proj)s_%(target)s_tpb__ %(nr)s
#define __proj%(id_proj)s_%(target)s_nb__ %(nb)s
"""

                # proj.target can hold a single or multiple targets. We use
                # one configuration for all but need to define single names anyways
                target_list = proj.target if isinstance(proj.target, list) else [proj.target]
                for target in target_list:
                    configuration += cfg % {
                        'id_proj': proj.id,
                        'target': target,
                        'nr': num_threads,
                        'nb': num_blocks
                    }

                    if get_global_config('verbose'):
                        Messages._print('projection', proj.id, 'with target', target, ' - kernel config: (', num_blocks, ',', num_threads, ')')

        return configuration

    def _cuda_stream_config(self):
        """
        With Fermi Nvidia introduced multiple streams respectively concurrent
        kernel execution (requires device with compute compability > 2.x).

        Notice:

            Only related to the CUDA implementation
        """
        if self._cuda_config is None:
            pop_assign = "    // populations\n"
            proj_assign = "    // projections\n"
            max_number_streams = 0
        else:
            # TODO: maybe this should be a parameter too? As one could schedule multiple objects
            #       in one stream, the maximum number is not exploited
            max_number_streams = max(len(self._populations), len(self._projections))

            # HD:
            # the try-except blocks here are a REALLY lazy method.
            # TODO: it should be implemented more carefully in future
            pop_assign = "    // populations\n"
            for pop in self._populations:
                try:
                    sid = self._cuda_config[pop]['stream']
                    pop_assign += """    pop%(pid)s.stream = streams[%(sid)s];
""" % {'pid': pop.id, 'sid': sid}
                except KeyError:
                    # default stream, if either no cuda_config at all or
                    # the population is not configured by user
                    pop_assign += """    pop%(pid)s.stream = 0;
""" % {'pid': pop.id}

            proj_assign = "    // projections\n"
            for proj in self._projections:
                try:
                    sid = self._cuda_config[proj]['stream']
                    proj_assign += """    proj%(pid)s.stream = streams[%(sid)s];
""" % {'pid': proj.id, 'sid': sid}
                except KeyError:
                    # default stream, if either no cuda_config at all or
                    # the projection is not configured by user
                    proj_assign += """    proj%(pid)s.stream = 0;
""" % {'pid': proj.id}

        # Write config
        stream_config = BaseTemplate.cuda_stream_setup % {
            'nbStreams': max_number_streams,
            'pop_assign': pop_assign,
            'proj_assign': proj_assign
        }

        return stream_config

    def _guess_pop_kernel_config(self, pop):
        """
        Instead of a fixed amount of threads for each kernel, we try
        to guess a good configuration based on the population size.
        """
        from math import log

        max_tpb = 512
        warp_size = 32

        num_neur = pop.size / 2 # at least 2 iterations per thread
        guess = warp_size       # smallest block is 1 warp

        # Simplest case: we have more neurons than
        # available threads per block
        if num_neur > max_tpb:
            guess = max_tpb

        # check which is the closest possible thread amount
        pow_of_2 = [2**x for x in range(int(log(warp_size, 2)), int(log(max_tpb, 2))+1)]
        for i in range(len(pow_of_2)):
            if pow_of_2[i] < num_neur:
                continue
            else:
                guess = pow_of_2[i]
                break

        return guess

    def _guess_proj_kernel_config(self, proj):
        """
        Instead of a fixed amount of threads for each kernel, we try
        to guess a good configuration based on the pre-synaptic population size.
        """
        from math import log

        max_tpb = 512
        warp_size = 32

        num_neur = proj.pre.size / 4 # at least 1/4 of the neurons are connected
        guess = warp_size       # smallest block is 1 warp

        # Simplest case: we have more neurons than
        # available threads per block
        if num_neur > max_tpb:
            guess = max_tpb

        # check which is the closest possible thread amount
        pow_of_2 = [2**x for x in range(int(log(warp_size, 2)), int(log(max_tpb, 2))+1)]
        for i in range(len(pow_of_2)):
            if pow_of_2[i] < num_neur:
                continue
            else:
                guess = pow_of_2[i]
                break

        if get_global_config('verbose'):
            Messages._print('projection', proj.id, ' - kernel size:', guess)

        return guess

    def _cuda_common_kernel(self, projections):
        """
        Some sparse matrix formats require additional functions. Which we need to
        define only once.
        """
        fmts = []

        for proj in projections:
            fmts.append(proj._storage_format)

        fmts = list(set(fmts))

        code = ""
        # TODO: generalize!
        if "csr" in fmts:
            from ANNarchy.generator.Projection.CUDA.CSR import additional_global_functions
            code += additional_global_functions
        elif "csr_vector" in fmts:
            from ANNarchy.generator.Projection.CUDA.CSR_Vector import additional_global_functions
            code += additional_global_functions

        return code
