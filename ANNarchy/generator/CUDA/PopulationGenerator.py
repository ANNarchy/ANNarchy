import ANNarchy.core.Global as Global

class PopulationGenerator(object):

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop):
        # Is it a specific population?
        if pop.generator['cuda']['header_pop_struct']:
            return pop.generator['cuda']['header_pop_struct'] % {'id': pop.id}

        # Generate the structure for the population.
        code = """
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    // Active
    bool _active;

    // assigned stream for concurrent kernel execution ( CC > 2.x )
    cudaStream_t stream;
"""
        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            Global._error("Spike coded neurons are not supported on GPUs yet ... ")
            exit(0)

        # Record
        code+="""
    // Record parameter
    int record_period;
    long int record_offset;
"""
        #
        # create for each variable a host, device array and add dirty flag
        # if the flag is set to true, the next possible time, the data will
        # exchanged between host and GPU

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Arrays for the presynaptic sums
        code += """
    // Targets
"""
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets']+pop.targets)):
                code += """    std::vector<double> _sum_%(target)s;
    double *gpu_sum_%(target)s;
""" % {'target' : target}

        # Global operations
        code += """
    // Global operations
"""
        for op in pop.global_operations:
            code += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}

        # Arrays for the random numbers
        code += """
    // cuda RNG states
"""
        for rd in pop.neuron_type.description['random_distributions']:
            code += """    curandState* gpu_%(rd_name)s;
""" % { 'rd_name' : rd['name'] }

        # Delays
        if pop.max_delay > 1:
            if pop.neuron_type.type == "rate":
                code += """
    // Delays for rate-coded population
    std::deque< double* > gpu_delayed_r;
"""
            else:
                Global._error("synaptic delays for spiking neurons are not implemented yet ...")
                exit(0)

        # Local functions
        if len(pop.neuron_type.description['functions'])>0:
            code += """
    // Local functions
"""
            for func in pop.neuron_type.description['functions']:
                code += ' '*4 + func['cpp'] + '\n'

        # Finish the structure
        code += """
};
"""
        return code % {'id': pop.id}

#######################################################################
############## BODY ###################################################
#######################################################################

    def update_neuron(self, pop):
        """
        returns all data needed for population step kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        # Is it a specific population?
        if pop.generator['cuda']['body_update_neuron']:
            Global._error("Customized populations are not usable on CUDA yet.")
            return "", "", ""

        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", ""

        # Neural update
        from ..Utils import generate_equation_code

        header = ""
        body = ""
        call = ""

        # determine variables and attributes
        var = ""
        par = ""
        tar = ""
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] in pop.neuron_type.description['local']:
                var += """, %(type)s* %(name)s""" % { 'type': attr['ctype'], 'name': attr['name'] }
            else:
                par += """, %(type)s %(name)s""" % { 'type': attr['ctype'], 'name': attr['name'] }

        # random variables
        for rd in pop.neuron_type.description['random_distributions']:
            var += """, curandState* %(rd_name)s""" % { 'rd_name' : rd['name'] }

        # global operations
        for op in pop.global_operations:
            par += """, double _%(op)s_%(var)s """ % {'op': op['function'], 'var': op['variable']}

        # targets
        for target in pop.neuron_type.description['targets']:
            tar += """, double* _sum_%(target)s""" % {'target' : target}

        #Global variables
        glob_eqs = ""
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global') % {'id': pop.id}
        if eqs.strip() != "":
            glob_eqs = """
    if ( threadIdx.x == 0)
    {
%(eqs)s
    }
""" % {'id': pop.id, 'eqs': eqs }
            glob_eqs = glob_eqs.replace("pop"+str(pop.id)+".", "")

        # Local variables
        loc_eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local') % {'id': pop.id}
        loc_eqs = loc_eqs.replace("pop"+str(pop.id)+".", "")

        # we replace the rand_%(id)s by the corresponding curand... term
        for rd in pop.neuron_type.description['random_distributions']:
            if rd['dist'] == "Uniform":
                term = """curand_uniform_double( &%(rd)s[i]) * (%(max)s - %(min)s) + %(min)s""" % { 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
            elif rd['dist'] == "Normal":
                term = """curand_normal_double( &%(rd)s[i])""" % { 'rd': rd['name'] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
            elif rd['dist'] == "LogNormal":
                term = """curand_log_normal_double( &%(rd)s[i], %(mean)s, %(std_dev)s)""" % { 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
            else:
                Global._error("Unsupported random distribution on GPUs: " + rd['dist'])

        # remove all types
        repl_types = ["double*", "float*", "int*", "curandState*", "double", "float", "int"]
        tar_wo_types = tar
        var_wo_types = var
        par_wo_types = par
        for type in repl_types:
            tar_wo_types = tar_wo_types.replace(type, "")
            var_wo_types = var_wo_types.replace(type, "")
            par_wo_types = par_wo_types.replace(type, "")

        #
        # create kernel prototypes
        from .cuBodyTemplate import pop_kernel
        body += pop_kernel % {  'id': pop.id,
                                'local_eqs': loc_eqs,
                                'global_eqs': glob_eqs,
                                'pop_size': str(pop.size),
                                'tar': tar,
                                'tar2': tar_wo_types,
                                'var': var,
                                'var2': var_wo_types,
                                'par': par,
                                'par2': par_wo_types
                             }

        #
        # create kernel prototypes
        header += """
void Pop%(id)s_step( cudaStream_t stream, double dt%(tar)s%(var)s%(par)s );
""" % { 'id': pop.id, 'tar': tar, 'var': var, 'par': par }

        #
        #    for calling entites we need to determine again all members
        var = ""
        par = ""
        tar = ""
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] in pop.neuron_type.description['local']:
                var += """, pop%(id)s.gpu_%(name)s""" % { 'id': pop.id, 'name': attr['name'] } 
            else:
                par += """, pop%(id)s.%(name)s""" % { 'id': pop.id, 'name': attr['name'] }

        # random variables
        for rd in pop.neuron_type.description['random_distributions']:
            var += """, pop%(id)s.gpu_%(rd_name)s""" % { 'id': pop.id, 'rd_name' : rd['name'] }

        # targets
        for target in pop.neuron_type.description['targets']:
            tar += """, pop%(id)s.gpu_sum_%(target)s""" % { 'id': pop.id, 'target' : target}

        # global operations
        for op in pop.global_operations:
            par += """, pop%(id)s._%(op)s_%(var)s""" % { 'id': pop.id, 'op': op['function'], 'var': op['variable'] }

        from .cuBodyTemplate import pop_kernel_call
        call += pop_kernel_call % { 'id': pop.id,
                                    'tar': tar.replace("double*","").replace("int*",""),
                                    'var': var.replace("double*","").replace("int*",""),
                                    'par': par.replace("double","").replace("int","")
                                  }

        return body, header, call

    def delay_code(self, pop):
        """
        Implementation Note:

            Currently I see no better way to implement delays, as consequence of missing device-device memory transfers ...

            This implementation is from a performance point of view problematic, cause of low host-device memory bandwith,
            maybe enhancable through pinned memory (CC 2.x), or asynchronous device transfers (CC 3.x)
        """
        # No delay
        if pop.max_delay <= 1:
            return ""

        # Is it a specific population?
        if pop.generator['cuda']['body_delay_code']:
            return pop.generator['cuda']['body_delay_code'] % {'id': pop.id}

        code = ""
        if pop.neuron_type.type == 'rate':
            code += """
    // Enqueuing outputs of pop%(id)s (%(name)s)
    if ( pop%(id)s._active ) {
        double* endPtr_pop%(id)s = pop%(id)s.gpu_delayed_r.back();
        pop%(id)s.gpu_delayed_r.pop_back();
        pop%(id)s.gpu_delayed_r.push_front(endPtr_pop%(id)s);
        std::vector<double> tmp_r_pop%(id)s = std::vector<double>( pop%(id)s.size, 0.0);
        cudaMemcpy( tmp_r_pop%(id)s.data(), pop%(id)s.gpu_r, sizeof(double) * pop%(id)s.size, cudaMemcpyDeviceToHost);
        cudaMemcpy( endPtr_pop%(id)s, tmp_r_pop%(id)s.data(), sizeof(double) * pop%(id)s.size, cudaMemcpyHostToDevice);
    }
""" % {'id': pop.id, 'name' : pop.name }
        else:
            Global._error("Customized delay code is not usable on CUDA yet.")
            return ""

        return code

    def init_globalops(self, pop):
        # Is it a specific population?
        if pop.generator['cuda']['body_globalops_init']:
            return pop.generator['cuda']['body_globalops_init'] %{'id': pop.id}

        code = ""
        for op in pop.global_operations:
            code += """    pop%(id)s._%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
        return code

    def init_random_distributions(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_random_dist_init']:
            return pop.generator['omp']['body_random_dist_init'] %{'id': pop.id}

        code = ""
        for rd in pop.neuron_type.description['random_distributions']:
            code += """    cudaMalloc((void**)&pop%(id)s.gpu_%(rd_name)s, pop%(id)s.size * sizeof(curandState));
    init_curand_states( pop%(id)s.size, pop%(id)s.gpu_%(rd_name)s, seed );
""" % {'id': pop.id, 'rd_name': rd['name'] }
        return code

    def init_delay(self, pop):
        code = "    // Delays from pop%(id)s (%(name)s)\n" % {'id': pop.id, 'name': pop.name}

        # Is it a specific population?
        if pop.generator['omp']['body_delay_init']:
            code += pop.generator['omp']['body_delay_init'] %{'id': pop.id, 'delay': pop.max_delay}
            return code

        if pop.neuron_type.type == 'rate':
            code += """    pop%(id)s.gpu_delayed_r = std::deque< double* >(%(delay)s, NULL);
    for ( int i = 0; i < %(delay)s; i++ )
        cudaMalloc( (void**)& pop%(id)s.gpu_delayed_r[i], sizeof(double)*pop%(id)s.size);
""" % {'id': pop.id, 'delay': pop.max_delay}
        else: # SPIKE
            Global._error("no synaptic delays for spiking synapses on cuda implemented ...")
            pass

        return code

    def init_population(self, pop):
        # active is true by default
        code = """
    /////////////////////////////
    // Population %(id)s
    /////////////////////////////
    pop%(id)s._active = true;
    pop%(id)s.record_period = 1;
    pop%(id)s.record_offset = 0;
""" % {'id': pop.id}

        # Is it a specific population?
        if pop.generator['cuda']['body_spike_init']:
            code += pop.generator['cuda']['body_spike_init'] %{'id': pop.id}
            return code

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:     
                code += """
    // Local parameter %(name)s
    pop%(id)s.%(name)s = std::vector<%(type)s>(pop%(id)s.size, %(init)s);
    pop%(id)s.%(name)s_dirty = true;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
    // Global parameter %(name)s
    pop%(id)s.%(name)s = %(init)s;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Variables
        for var in pop.neuron_type.description['variables']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local variable %(name)s
    pop%(id)s.%(name)s = std::vector<%(type)s>(pop%(id)s.size, %(init)s);
    pop%(id)s.recorded_%(name)s = std::vector<std::vector<%(type)s> >(0, std::vector<%(type)s>(0,%(init)s));
    pop%(id)s.record_%(name)s = false;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
    // Global variable %(name)s
    pop%(id)s.%(name)s = %(init)s;
    pop%(id)s.recorded_%(name)s = std::vector<%(type)s>(0, %(init)s);
    pop%(id)s.record_%(name)s = false;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += """
    pop%(id)s._sum_%(target)s = std::vector<double>(pop%(id)s.size, 0.0);""" %{'id': pop.id, 'target': target}

        if pop.neuron_type.type == 'spike':
            code += """
    // Spiking neuron
    pop%(id)s.refractory = std::vector<int>(pop%(id)s.size, 0);
    pop%(id)s.record_spike = false;
    pop%(id)s.recorded_spike = std::vector<std::vector<long int> >();
    for(int i = 0; i < pop%(id)s.size; i++)
        pop%(id)s.recorded_spike.push_back(std::vector<long int>());
    pop%(id)s.spiked = std::vector<int>(0, 0);
    pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
    pop%(id)s.refractory_remaining = std::vector<int>(pop%(id)s.size, 0);
""" % {'id': pop.id}

        return code

    def reset_computesum(self, pop):
        code = ""
        for target in pop.targets:
            code += """
    if (pop%(id)s._active)
        memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(double));
""" % {'id': pop.id, 'target': target}
        return code

    def update_globalops(self, pop):
        from .GlobalOperationTemplate import global_operation_templates

        # Is it a specific population?
        if pop.generator['cuda']['body_update_globalops']:
            return pop.generator['cuda']['body_update_globalops'] %{ 'id': pop.id}

        code = ""
        for op in pop.global_operations:
            call = global_operation_templates[op['function']]['call'] % { 'id': pop.id, 'var': op['variable']  }
            code += """
    double *tmp;
    cudaMalloc((void**)&tmp, sizeof(double));
%(call)s
    cudaFree(tmp);
""" % { 'call': call}

        return code

    def record(self, pop):
        # Is it a specific population?
        if pop.generator['cuda']['body_record']:
            return pop.generator['cuda']['body_record'] %{'id': pop.id}

        # TODO:
        return ""

#######################################################################
############## PYX ####################################################
#######################################################################

    def pyx_struct(self, pop):
        # Is it a specific population?
        if pop.generator['cuda']['pyx_pop_struct']:
            return pop.generator['cuda']['pyx_pop_struct'] %{'id': pop.id}

        code = """
    # Population %(id)s (%(name)s)
    cdef struct PopStruct%(id)s :
        int size
        bool _active
"""

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        # Local parameter %(name)s
        vector[%(type)s] %(name)s
        bool %(name)s_dirty
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
        # Global parameter %(name)s
        %(type)s  %(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        # Local variable %(name)s
        vector[%(type)s] %(name)s
        bool %(name)s_dirty
        #vector[vector[%(type)s]] recorded_%(name)s 
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
        # Global variable %(name)s
        %(type)s  %(name)s
        bool %(name)s_dirty
        #vector[%(type)s] recorded_%(name)s
        #bool record_%(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

        # Arrays for the presynaptic sums of rate-coded neurons
        if pop.neuron_type.type == 'rate':
            code += """
        # Targets
"""
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += """        vector[double] _sum_%(target)s
""" % {'target' : target}

        # Finalize the code
        return code % {'id': pop.id, 'name': pop.name}

    def pyx_wrapper(self, pop):
        # Is it a specific population?
        if pop.generator['cuda']['pyx_pop_class']:
            return pop.generator['cuda']['pyx_pop_class'] %{'id': pop.id}

        # Init
        code = """
# Population %(id)s (%(name)s)
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size"""% {'id': pop.id, 'name': pop.name}

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)
        pop%(id)s.%(name)s_dirty = True""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
        pop%(id)s.%(name)s = %(init)s""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Variables
        for var in pop.neuron_type.description['variables']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)
        #pop%(id)s.recorded_%(name)s = vector[vector[%(type)s]](0, vector[%(type)s](0,%(init)s))
        #pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
        pop%(id)s.%(name)s = %(init)s
        pop%(id)s.%(name)s_dirty = True
        #pop%(id)s.recorded_%(name)s = vector[%(type)s](0, %(init)s)
        #pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += """
        pop%(id)s._sum_%(target)s = vector[double](size, 0.0)""" %{'id': pop.id, 'target': target}

        # Size property
        code += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

        # Activate population
        code += """
    def activate(self, bool val):
        pop%(id)s._active = val
""" % {'id': pop.id}

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    # Local parameter %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    # Global parameter %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        return code