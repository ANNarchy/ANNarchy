import ANNarchy.core.Global as Global

class PopulationGenerator(object):

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop):
        """
        Create the c-type definition of population *pop*
        """
        # Generate the structure for the population.
        code = """
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // assigned stream for concurrent kernel execution ( CC > 2.x )
    cudaStream_t stream;

    // Active
    bool _active;
"""

        # Record
        code+="""
    // Record parameter
    int record_period;
    long int record_offset;
"""

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;    // host
    %(type)s *gpu_%(name)s;    // device
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
    std::vector< %(type)s > %(name)s ;    // host
    %(type)s *gpu_%(name)s;    // device
    bool %(name)s_dirty;
    // std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    // bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    // std::vector< %(type)s > recorded_%(name)s ;
    // bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Arrays for the presynaptic sums
        code += """
    // Targets
"""
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets']+pop.targets)):
                code += """    std::vector<double> _sum_%(target)s;    // host
    double *gpu_sum_%(target)s;    // device
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
    // RNG states
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

    def update_neuron(self, pop):#
        """
        returns all data needed for population step kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
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
        #
        #    Currently I see no better way to implement delays, as consequence of missing device-device memory transfers ...
        #
        #    This implementation is from a performance point of view problematic, cause of low host-device memory bandwith,
        #    maybe enhancable through pinned memory (CC 2.x), or asynchronous device transfers (CC 3.x)
        code = ""

        if pop.max_delay <= 1:
            return ""

        code += """
    // Enqueuing outputs of pop%(id)s
    if ( pop%(id)s._active ) {
        double* endPtr_pop%(id)s = pop%(id)s.gpu_delayed_r.back();
        pop%(id)s.gpu_delayed_r.pop_back();
        pop%(id)s.gpu_delayed_r.push_front(endPtr_pop%(id)s);
        std::vector<double> tmp_r_pop%(id)s = std::vector<double>( pop%(id)s.size, 0.0);
        cudaMemcpy( tmp_r_pop%(id)s.data(), pop%(id)s.gpu_r, sizeof(double) * pop%(id)s.size, cudaMemcpyDeviceToHost);
        cudaMemcpy( endPtr_pop%(id)s, tmp_r_pop%(id)s.data(), sizeof(double) * pop%(id)s.size, cudaMemcpyHostToDevice);
    }
"""
        return code % {'id': pop.id }
