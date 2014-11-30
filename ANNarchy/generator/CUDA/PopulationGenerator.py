import ANNarchy.core.Global as Global

class PopulationGenerator(object):

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop):
        """
        Create the c-type definition of population *pop*
        """
        # Is it a specific population?
        if pop.generator['cuda']['header_pop_struct']:
            Global._error("Customized populations are not usable on CUDA yet.")
            return ""

        if pop.neuron_type.type == 'spike':
            Global._error("Spiking populations are not usable on CUDA yet.")
            return ""

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
