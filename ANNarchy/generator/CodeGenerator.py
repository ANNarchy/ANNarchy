#==============================================================================
#
#     CodeGenerator.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2019  Julien Vitay <julien.vitay@gmail.com>,
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
#==============================================================================
import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.parser.Extraction import extract_functions

from .PyxGenerator import PyxGenerator
from .MonitorGenerator import MonitorGenerator

from .Population import OpenMPGenerator, CUDAGenerator
from .Projection import OpenMPProjectionGenerator, CUDAProjectionGenerator

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
        Constructor initializes PopulationGenerator and ProjectionGenerator
        class and stores the provided information for later use.

        Parameters:

            * *net_id*: unique id for the current network
            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *populations*: list of populations
            * *populations*: list of projections
            * *cuda_config*: configuration dict for cuda. check the method
              _cuda_kernel_config for more details.
        """
        self._net_id = net_id
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._cuda_config = cuda_config

        for pop in self._populations:
            pop._generate()
        for proj in self._projections:
            proj._generate()

        if Global.config['profiling']:
            if Global.config['paradigm'] == "openmp":
                #from .Profile import PAPIProfile
                #self._profgen = PAPIProfile(self._annarchy_dir, net_id)
                #self._profgen.generate()
                from .Profile import CPP11Profile
                self._profgen = CPP11Profile(self._annarchy_dir, net_id)
                self._profgen.generate()
            elif Global.config['paradigm'] == "cuda":
                from .Profile import CUDAProfile
                self._profgen = CUDAProfile(self._annarchy_dir, net_id)
                self._profgen.generate()
            else:
                Global._error('No ProfileGenerator available for '
                              + Global.config['paradigm'])
        else:
            self._profgen = None

        if Global.config['paradigm'] == "openmp":
            self._popgen = OpenMPGenerator(self._profgen, net_id)
            self._projgen = OpenMPProjectionGenerator(self._profgen, net_id)
        elif Global.config['paradigm'] == "cuda":
            self._popgen = CUDAGenerator(self._profgen, net_id)
            self._projgen = CUDAProjectionGenerator(self._profgen, net_id)
        else:
            Global._error("No PopulationGenerator for " + Global.config['paradigm'])

        self._pyxgen = PyxGenerator(annarchy_dir, populations, projections, net_id)
        self._recordgen = MonitorGenerator(annarchy_dir, populations, projections, net_id)

        self._pop_desc = []
        self._proj_desc = []

    def generate(self):
        """
        Generate code files and store them in target directory (located at
        self._annarchy_dir/generate). More detailed the following files are
        generated, by this class:

            * *ANNarchy.cpp*: main simulation loop, object instantiation
            * *ANNarchy.h*: collection of all objects, interface to Python
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
        if Global.config['verbose']:
            if Global.config['paradigm'] == "openmp":
                if Global.config['num_threads'] > 1:
                    Global._print('\nGenerate code for OpenMP ...')
                else:
                    Global._print('\nGenerate sequential code ...')
            elif Global.config['paradigm'] == "cuda":
                print('\nGenerate CUDA code ...')
            else:
                raise NotImplementedError

        # check if the user access some new features, or old ones
        # which changed.
        self._check_experimental_features(self._populations, self._projections)

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
        with open(source_dest+'ANNarchy.h', 'w') as ofile:
            ofile.write(self._generate_header())

        # Generate monitor code for the analysed pops and projs
        self._recordgen.generate()

        # Generate cpp code for the analysed pops and projs
        if Global.config['paradigm'] == "openmp":
            with open(source_dest+'ANNarchy.cpp', 'w') as ofile:
                ofile.write(self._generate_body())

        elif Global.config['paradigm'] == "cuda":
            device_code, host_code = self._generate_body()
            with open(source_dest+'ANNarchyHost.cu', 'w') as ofile:
                ofile.write(host_code)
            with open(source_dest+'ANNarchyDevice.cu', 'w') as ofile:
                ofile.write(device_code)

        else:
            raise NotImplementedError

        # Generate cython code for the analysed pops and projs
        with open(source_dest+'ANNarchyCore'+str(self._net_id)+'.pyx', 'w') as ofile:
            ofile.write(self._pyxgen.generate())

        self._generate_file_overview(source_dest)

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
                desc = """proj%(id_proj)s, %(type_proj)s( pre = %(pre_name)s, post = %(post_name)s, target = %(target)s ) using connector: %(pattern)s \n""" % {
                    'id_proj': proj.id,
                    'type_proj': proj_type,
                    'pre_name': proj.pre.name,
                    'post_name': proj.post.name,
                    'target': proj.target,
                    'pattern': proj.connector_description
                }
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
            pop.global_operations = [dict(y) for y in set(tuple(x.items()) for x in pop.global_operations)]
            pop.delayed_variables = sorted(list(set(pop.delayed_variables)))

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
        custom_func = self._header_custom_functions()

        # Custom constants
        custom_constant = self._header_custom_constants()

        # Include OMP
        include_omp = "#include <omp.h>" if Global.config['num_threads'] > 1 else ""

        if Global.config['paradigm'] == "openmp":
            from .Template.BaseTemplate import omp_header_template, built_in_functions, integer_power_cpu
            return omp_header_template % {
                'float_prec': Global.config['precision'],
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'custom_func': custom_func,
                'custom_constant': custom_constant,
                'built_in': built_in_functions + integer_power_cpu % {'float_prec': Global.config['precision']},
                'include_omp': include_omp
            }
        elif Global.config['paradigm'] == "cuda":
            from .Template.BaseTemplate import cuda_header_template, built_in_functions
            return cuda_header_template % {
                'float_prec': Global.config['precision'],
                'pop_struct': pop_struct,
                'proj_struct': proj_struct,
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'custom_func': custom_func,
		'built_in': built_in_functions,
                'custom_constant': custom_constant
            }
        else:
            raise NotImplementedError

    def _header_custom_functions(self):
        """
        Generate code for custom functions defined globally and are usable
        witihn neuron or synapse descriptions. These functions can only rely on
        provided arguments.
        """
        if len(Global._objects['functions']) == 0:
            return ""

        # Attention CUDA: this definition will work only on host side.
        code = ""
        for _, func in Global._objects['functions']:
            code += extract_functions(func, local_global=True)[0]['cpp'] + '\n'

        return code

    def _header_custom_constants(self):
        """
        Generate code for custom constants
        """
        if len(Global._objects['constants']) == 0:
            return ""

        code = ""
        for obj in Global._objects['constants']:
            obj_str = {
                'name': obj.name,
                'float_prec': Global.config['precision']
            }
            if Global._check_paradigm("openmp"):
                code += """
extern %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value);""" % obj_str
            elif Global._check_paradigm("cuda"):
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

        * host_decl_code: declarations in header file (host side)
        * host_init_code: initialization code (host side)
        * device_decl_code: declarations in header file (device side)

        """
        if Global._check_paradigm("openmp"):
            if len(Global._objects['constants']) == 0:
                return "", ""

            decl_code = ""
            init_code = ""
            for obj in Global._objects['constants']:
                obj_str = {
                    'name': obj.name,
                    'value': obj.value,
                    'float_prec': Global.config['precision']
                }
                decl_code += """
%(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value){%(name)s = value;};""" % obj_str
                init_code += """
        %(name)s = 0.0;""" % obj_str

            return decl_code, init_code
        elif Global._check_paradigm("cuda"):
            if len(Global._objects['constants']) == 0:
                return "", "", ""

            host_decl_code = ""
            host_init_code = ""
            device_decl_code = ""
            for obj in Global._objects['constants']:
                obj_str = {
                    'name': obj.name,
                    'value': obj.value,
                    'float_prec': Global.config['precision']
                }
                host_decl_code += """
__device__ __constant__ %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value){
    cudaError_t err = cudaMemcpyToSymbol(%(name)s, &value, sizeof(%(float_prec)s), 0, cudaMemcpyHostToDevice);
#ifdef _DEBUG
    std::cout << "set %(name)s " << value << std::endl;
    if ( err != cudaSuccess )
        std::cerr << cudaGetErrorString(err) << std::endl;
#endif
}""" % obj_str
                device_decl_code += "__device__ __constant__ %(float_prec)s %(name)s;\n" % obj_str
                host_init_code += """
        %(name)s = 0.0;""" % obj_str

            return host_decl_code,  host_init_code, device_decl_code
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
        compute_sums = self._body_computesum_proj()

        # Init rng dist
        init_rng_dist = ""
        for pop in self._populations:
            init_rng_dist += """pop%(id)s.init_rng_dist();\n""" % {'id': pop.id}

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
        structural_plasticity = self._body_structural_plasticity()

        # Early stopping
        run_until = self._body_run_until()

        # Number threads
        number_threads = "omp_set_num_threads(threads);" if Global.config['num_threads'] > 1 else ""

        #Profiling
        if self._profgen:
            prof_dict = self._profgen.generate_body_dict()
        else:
            from .Profile import ProfileGenerator
            prof_dict = ProfileGenerator(self._annarchy_dir, self._net_id).generate_body_dict()

        #
        # Generate the ANNarchy.cpp code, the corrsponding template differs
        # greatly. For further information take a look into the corresponding
        # branches.
        #
        if Global.config['paradigm'] == "openmp":
            # custom constants
            custom_constant, _ = self._body_custom_constants()

            from .Template.BaseTemplate import omp_body_template
            base_dict = {
                'float_prec': Global.config['precision'],
                'pop_ptr': pop_ptr,
                'proj_ptr': proj_ptr,
                'glops_def': glop_definition,
                'initialize': self._body_initialize(),
                'init_rng_dist': init_rng_dist,
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
                'custom_constant': custom_constant,
            }

            base_dict.update(prof_dict)
            return omp_body_template % base_dict

        elif Global.config['paradigm'] == "cuda":
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

            # custom constants
            host_custom_constant, _, device_custom_constant = self._body_custom_constants()

            # custom functions
            custom_func = ""
            for pop in self._pop_desc:
                custom_func += pop['custom_func']
            for proj in self._proj_desc:
                custom_func += proj['custom_func']
            for _, func in Global._objects['functions']:
                custom_func += extract_functions(func, local_global=True)[0]['cpp'].replace("inline", "__device__") + '\n'

            pop_kernel = ""
            for pop in self._pop_desc:
                pop_kernel += pop['update_body']

            pop_update_fr = ""
            for pop in self._pop_desc:
                pop_update_fr += pop['update_FR']

            psp_kernel = ""
            for proj in self._proj_desc:
                psp_kernel += proj['psp_body']

            kernel_def = ""
            for pop in self._pop_desc:
                kernel_def += pop['update_header']

            for proj in self._proj_desc:
                kernel_def += proj['psp_header']
                kernel_def += proj['update_synapse_header']
                kernel_def += proj['postevent_header']

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

            postevent_kernel = ""
            for proj in self._proj_desc:
                postevent_kernel += proj['postevent_body']

            postevent_call = ""
            for proj in self._proj_desc:
                postevent_call += proj['postevent_call']

            clear_sums = self._body_resetcomputesum_pop()

            # global operations
            glob_ops_header, glob_ops_body = self._body_def_glops()
            kernel_def += glob_ops_header

            # determine number of threads per kernel
            threads_per_kernel = self._cuda_kernel_config()

            #  concurrent kernel execution
            stream_setup = self._cuda_stream_config()

            # memory transfers
            host_device_transfer, device_host_transfer = "", ""
            for pop in self._pop_desc + self._proj_desc:
                host_device_transfer += pop['host_to_device']
                device_host_transfer += pop['device_to_host']

            #Profiling
            if self._profgen:
                prof_dict = self._profgen.generate_body_dict()
            else:
                from .Profile import ProfileGenerator
                prof_dict = ProfileGenerator(self._annarchy_dir, self._net_id).generate_body_dict()

            #
            # HD ( 31.07.2016 ):
            #
            # I'm not really sure, what exactly causes the problem with this
            # atomicAdd function. If we move it into ANNarchyDevice.cu, the
            # macro seems to be evaluated wrongly and the atomicAdd() function
            # appears doubled or appears not.
            #
            # So as "solution", the atomicAdd definition block resides in
            # ANNarchyHost and only the computation kernels are placed in
            # ANNarchyDevice. If we decide to use SDK8 as lowest requirement,
            # one can move this kernel too.
            from .Template.BaseTemplate import cuda_device_kernel_template, cuda_host_body_template, built_in_functions, integer_power_cuda
            device_code = cuda_device_kernel_template % {
                #device stuff
                'pop_kernel': pop_kernel,
                'psp_kernel': psp_kernel,
                'syn_kernel': syn_kernel,
                'glob_ops_kernel': glob_ops_body,
                'postevent_kernel': postevent_kernel,
                'custom_func': custom_func,
                'custom_constant': device_custom_constant,
                'built_in': built_in_functions + integer_power_cuda % {'float_prec': Global.config['precision']},
                'float_prec': Global.config['precision']
            }

            base_dict = {
                # network definitions
                'float_prec': Global.config['precision'],
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
                'kernel_def': kernel_def,
                'kernel_config': threads_per_kernel,
                'custom_constant': host_custom_constant
            }
            base_dict.update(prof_dict)
            host_code = cuda_host_body_template % base_dict
            return device_code, host_code
        else:
            raise NotImplementedError

    def _body_initialize(self):
        """
        Define codes for the method initialize(), comprising of population and projection
        initializations, optionally profiling class.
        """
        profiling_init = "" if not Global.config['profiling'] else self._profgen.generate_init_network()

        # Initialize populations
        population_init = "    // Initialize populations\n"
        for pop in self._pop_desc:
            population_init += pop['init']

        # Initialize projections
        projection_init = "    // Initialize projections\n"
        for proj in self._proj_desc:
            projection_init += proj['init']

        # Initialize custom constants
        if Global.config['paradigm'] == "openmp":
            # Custom  constants
            _, custom_constant = self._body_custom_constants()

            from .Template.BaseTemplate import omp_initialize_template as init_tpl
        elif Global.config['paradigm'] == "cuda":
            # Custom  constants
            _, custom_constant, _ = self._body_custom_constants()

            from .Template.BaseTemplate import cuda_initialize_template as init_tpl
        else:
            raise NotImplementedError

        return init_tpl % {
            'prof_init': profiling_init,
            'pop_init': population_init,
            'proj_init': projection_init,
            'custom_constant': custom_constant
        }

    def _body_computesum_proj(self):
        """
        Call the copmpute_psp() method of Projection structs, only in case of
        openMP.
        """
        code = ""
        # Sum over all synapses
        for proj in self._projections:
            # Call the comput_psp method
            code += """    proj%(id)s.compute_psp();
""" % {'id' : proj.id}

        return code

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
        """
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

    def _body_def_glops(self):
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
                code += omp_template[op] % {
                    'type': Global.config['precision'],
                    'omp': '' if Global.config['num_threads'] > 1 else "//"
                }

            return code
        elif Global.config['paradigm'] == "cuda":
            if ops == []:
                return "", ""

            header = ""
            body = ""

            from .Template.GlobalOperationTemplate import global_operation_templates_cuda as cuda_template
            for op in list(set(ops)):
                header += cuda_template[op]['header'] % {'type': Global.config['precision']}
                body += cuda_template[op]['body'] % {'type': Global.config['precision']}

            return header, body
        else:
            raise NotImplementedError

    def _body_run_until(self):
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
        Each GPU kernel requires a launch configuration, established in
        the ANNarchyHost.cu code, minimum number of threads and blocks
        for calling the device functions.

        The default configuration is:

        * 192 threads for psp and synapse update
        * guessed amount of threads for neurons, based on population size
          (see _guess_pop_kernel_config)

        Notice:

            Only related to the CUDA implementation
        """
        from math import ceil

        # Population config adjust neuron_update
        configuration = "// Population config\n"
        for pop in self._populations:
            num_threads = self._guess_pop_kernel_config(pop)
            num_blocks = int(ceil(float(pop.size)/float(num_threads)))

            if self._cuda_config:
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

            if Global.config['verbose']:
                Global._print('population', pop.id, ' - kernel config: (', num_blocks, ',', num_threads, ')')

        # Projection config - adjust psp, synapse_local_update, synapse_global_update
        configuration += "\n// Projection config\n"
        for proj in self._projections:
            num_threads = 64 # self._guess_proj_kernel_config(proj)
            num_blocks = proj.post.size
            if self._cuda_config:
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

                if Global.config['verbose']:
                    Global._print('projection', proj.id, 'with target', target, ' - kernel config: (', num_blocks, ',', num_threads, ')')

        return configuration

    def _cuda_stream_config(self):
        """
        With Fermi Nvidia introduced multiple streams respectively concurrent
        kernel execution (requires device with compute compability > 2.x).

        Notice:

            Only related to the CUDA implementation
        """
        if self._cuda_config == None:
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
        from .Template.BaseTemplate import cuda_stream_setup
        stream_config = cuda_stream_setup % {
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
        from .CudaCheck import CudaCheck
        from math import log

        # HD (30. Nov. 2016):
        # neuons are typically computational heavy, thatswhy the number of
        # registers available is easily exceeded, so I use the next smaller
        # size as upper limit.
        max_tpb = CudaCheck().max_threads_per_block() / 2
        warp_size = CudaCheck().warp_size()
        if max_tpb==(-1/2) or warp_size==-1:
            # CudaCheck wasn't working correctly ...
            return 32

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
        from .CudaCheck import CudaCheck
        from math import log

        # HD (30. Nov. 2016):
        # neuons are typically computational heavy, thatswhy the number of
        # registers available is easily exceeded, so I use the next smaller
        # size as upper limit.
        max_tpb = CudaCheck().max_threads_per_block() / 2
        warp_size = CudaCheck().warp_size()
        if max_tpb==(-1/2) or warp_size==-1:
            # CudaCheck wasn't working correctly ...
            return 192

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

        if Global.config['verbose']:
            Global._print('projection', proj.id, ' - kernel size:', guess)

        return guess

    def _check_experimental_features(self, populations, projections):
        """
        The idea behind this method, is to check if new experimental features are used. This
        should help also the user to be aware of changes.
        """
        if Global.config['paradigm'] == "openmp":
            for proj in projections:
                if proj._storage_format == "csr":
                    Global._warning("CSR representation is an experimental feature, we greatly appreciate bug reports.")
                    break

        elif Global.config['paradigm'] == "cuda":
            for pop in populations:
                if pop.neuron_type.description['type'] == "spike":
                    Global._warning('Spiking neurons on GPUs is an experimental feature. We greatly appreciate bug reports.')
                    break

            for proj in projections:
                if proj._storage_format == "csr":
                    Global._warning("CSR representation is an experimental feature, we greatly appreciate bug reports.")
                    break
        else:
            pass
