# Template for specific population bypassing code generation.
# The id of the population should be let free with %(id)s
pop_generator_template = {
    'omp': {
        # C++ struct to encapsulate all data
        # Example:
        # struct PopStruct%(id)s{
        #     // Number of neurons
        #     int size;
        #     // Local parameter baseline
        #     std::vector< double > baseline ;
        #     // Local variable r
        #     std::vector< double > r ;
        #     // Targets
        #     std::vector<double> sum_exc;
        #     // Random numbers
        #     std::vector<double> rand_0;
        #     std::uniform_real_distribution<double> dist_rand_0;
        # }; 
        'header_pop_struct' : None,

        # Initialize the random distribution objects
        # Example:
        # pop%(id)s.rand_0 = std::vector<double>(pop%(id)s.size, 0.0);
        # pop%(id)s.dist_rand_0 = std::uniform_real_distribution<double>(-0.5, 0.5);
        'body_random_dist_init': None,

        # Initialize the delayed arrays. Needs the extra %(delay)s argument, which represents the maximum delay in the population.
        # Example:
        # pop%(id)s._delayed_r = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(pop%(id)s.size, 0.0));
        'body_delay_init': None,

        # Initialize spike-specific arrays. Does not need to overriden in most cases.
        # Example:
        # pop%(id)s.spiked = std::vector<int>(0, 0);
        # pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
        # pop%(id)s.refractory_remaining = std::vector<int>(pop%(id)s.size, 0);
        'body_spike_init': None,

        # Initialize the global operations
        # Example:
        # pop%(id)s._mean_r = 0.0;
        'body_globalops_init': None,

        # Updates the random numbers
        # Example:
        # // RD of pop%(id)s
        # #pragma omp parallel for
        # for(int i = 0; i < pop%(id)s.size; i++)
        # {
        #     pop%(id)s.rand_0[i] = pop%(id)s.dist_rand_0(rng[omp_get_thread_num()]);
        # }
        'body_random_dist_update': None,

        # Updates the neural variables
        # Example:
        # // Updating the local variables of population %(id)s
        # #pragma omp parallel for
        # for(int i = 0; i < pop%(id)s.size; i++){
        #     pop1.mp[i] += dt*(-pop%(id)s.mp[i] + pop%(id)s.rand_0[i] + pop%(id)s.sum_exc[i] + pop%(id)s.sum_inh[i])/pop%(id)s.tau;
        #     pop1.r[i] = clip(pop1.mp[i], 0.0, 1.0);
        # }
        'body_update_neuron': None,

        # Delays variables
        # Example:
        # // Enqueuing outputs of pop%(id)s
        # pop%(id)s._delayed_r.push_front(pop%(id)s.r);
        # pop%(id)s._delayed_r.pop_back();
        'body_delay_code': None,

        # Computes global operations
        # Example:
        # pop%(id)s._mean_r = mean_value(pop%(id)s.r;
        'body_update_globalops': None,

        # Export of the C++ struct to Cython (must have an indent of 4)
        # Example:
        # cdef struct PopStruct%(id)s :
        #     int size
        #     # Global parameter tau
        #     double  tau 
        #     # Local variable r
        #     vector[double] r 
        #     # Targets
        #     vector[double] sum_exc
        'pyx_pop_struct': None,

        # Wrapper class in Cython (no indentation)
        # Example:
        # cdef class pop%(id)s_wrapper :
        #     def __cinit__(self, size):
        #         pop%(id)s.size = size
        #         pop%(id)s.r = vector[double](size, 0.0)
        #         pop%(id)s.sum_exc = vector[double](size, 0.0)
        #     property size:
        #         def __get__(self):
        #             return pop%(id)s.size
        #     # Global parameter tau
        #     cpdef double get_tau(self):
        #         return pop%(id)s.tau
        #     cpdef set_tau(self, double value):
        #         pop%(id)s.tau = value
        #     # Local variable r
        #     cpdef np.ndarray get_r(self):
        #         return np.array(pop%(id)s.r)
        #     cpdef set_r(self, np.ndarray value):
        #         pop%(id)s.r = value
        #     cpdef double get_single_r(self, int rank):
        #         return pop%(id)s.r[rank]
        #     cpdef set_single_r(self, int rank, double value):
        #         pop%(id)s.r[rank] = value
        'pyx_pop_class': None,
    },
    'cuda': {   # TODO
        'header_pop_struct' : None,
        'body_random_dist_init': None,
        'body_delay_init': None,
        'body_spike_init': None,
        'body_globalops_init': None,
        'body_random_dist_update': None,
        'body_update_neuron': None,
        'body_delay_code': None,
        'body_update_globalops': None,
        'pyx_pop_struct': None,
        'pyx_pop_class': None,             
    } 
}

# Template for specific population bypassing code generation.
# The id of the population should be let free with %(id)s
proj_generator_template = {
    'omp': {
        # C++ struct to encapsulate all data
        # Example:
        # struct ProjStruct%(id_proj)s{
        #     // Number of dendrites
        #     int size;
        #     // Connectivity
        #     std::vector<int> post_rank ;
        #     std::vector< std::vector< int > > pre_rank ;
        #     std::vector< std::vector< int > > delay ;
        #    
        #     // Local parameter w
        #     std::vector< std::vector< double > > w ;
        # }; 
        'header_proj_struct' : None,

        # Initilaize the projection
        # Example:
        # 
        #    TODO:
        'body_proj_init': None,

        # Updates the random numbers
        # Example:
        #   TODO
        'body_random_dist_update': None,

        # Initializes the random numbers
        # Example:
        #   TODO
        'body_random_dist_init': None,

        # Updates the synapse variables
        # Example:
        # 
        #    TODO:
        'body_update_synapse': None,

        # compute the postsynaptic potential
        # Example:
        # 
        #    TODO:
        'body_compute_psp': None,
        
        # Export of the C++ struct to Cython (must have an indent of 4)
        # Example:
        # 
        #    TODO:
        'pyx_proj_struct': None,
        
        # Wrapper class in Cython (no indentation)
        # Example:
        # 
        #    TODO:
        'pyx_proj_class': None,
    },
    'cuda': {
        'header_proj_struct' : None,
        'body_proj_init': None,
        'body_random_dist_update': None,
        'body_random_dist_init': None,
        'body_update_synapse': None,
        'body_compute_psp': None,
        'pyx_proj_struct': None,
        'pyx_proj_class': None,
    }
}