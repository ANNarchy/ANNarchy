"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

population_header = """/*
 *  ANNarchy-version: %(annarchy_version)s
 */
 #pragma once
#include "ANNarchy.hpp"

// host defines
extern %(float_prec)s dt;
extern long int t;

// RNG - defined in ANNarchy.cu
extern unsigned long long global_seed;
extern void init_curand_states( int numBlocks, int numThreads, curandState* states, unsigned long long seed );

%(include_additional)s
%(include_profile)s

%(extern_global_operations)s
%(struct_additional)s

///////////////////////////////////////////////////////////////
// Main Structure for the population of id %(id)s (%(name)s)
///////////////////////////////////////////////////////////////
extern struct PopStruct%(id)s *pop%(id)s;
struct PopStruct%(id)s{

    PopStruct%(id)s(int size, int max_delay) {
        this->size = size;
        this->max_delay = max_delay;

        // HACK: the object constructor is now called by nanobind, need to update reference in C++ library
        pop%(id)s = this;

    #ifdef _TRACE_INIT
        std::cout << "  PopStruct%(id)s - this = " << this << " has been allocated." << std::endl;
    #endif
    }

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission
    
    // CUDA launch configuration
    cudaStream_t stream;
    unsigned int _nb_blocks;
    unsigned int _threads_per_block;

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }

%(declare_spike_arrays)s
    // Neuron specific parameters and variables
%(declare_parameters_variables)s
%(declare_delay)s
%(declare_FR)s
%(declare_additional)s

    // Profiling
%(declare_profile)s

    // Access methods to the parameters and variables
%(access_parameters_variables)s
%(access_additional)s

    // Method called to initialize the data structures
    void init_population() {
    #ifdef _TRACE_INIT
        std::cout << "  PopStruct%(id)s::init_population()" << std::endl;
    #endif
        _active = true;

        //
        // Launch configuration
        _threads_per_block = 128;
        _nb_blocks = static_cast<unsigned int>(ceil( static_cast<double>(size) / static_cast<double>(_threads_per_block) ) );
        _nb_blocks = std::min<unsigned int>(_nb_blocks, 65535);

        //
        // Model equations/parameters
%(init_parameters_variables)s
%(init_spike)s
%(init_delay)s
%(init_FR)s
%(init_additional)s
%(init_profile)s
    }

    // Method called to reset the population
    void reset() {
%(reset_spike)s
%(reset_delay)s
%(reset_additional)s
%(reset_read_flags)s
    }

    // Method to draw new random numbers
    void update_rng() {
%(update_rng)s
    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {
%(update_delay)s
    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {
%(update_max_delay)s
    }

    // Main method to update neural variables
    void update() {
%(update_variables)s
    }

    // Mean-firing rate computed on host
    void update_FR() {
%(update_FR)s
    }

    // Stop condition
    %(stop_condition)s

    // Memory transfers
    void host_to_device() {
%(host_to_device)s
    }

    void device_to_host(std::string attr_name) {
%(device_to_host)s
    #ifdef _DEBUG
        std::cout << "No memory transfer performed for attribute '" << attr_name << "'" << std::endl;
    #endif
    }

    // Memory Management: track memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(size_in_bytes)s
        return size_in_bytes;
    }

    // Memory Management: clear container
    void clear() {
%(clear_container)s
    }
};
"""

# c like definition of neuron attributes, where 'local' is used if values can vary across
# neurons, consequently 'global' is used if values are common to all neurons.Currently two
# types of sets are defined: openmp and cuda. In cuda case additional 'dirty' flags are
# created.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
#
attribute_decl = {
    'local': """
    // Local attribute %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s *gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
""",
    'global': {
        'parameter': """
    // Global parameter %(name)s
    %(type)s %(name)s;
""",
        'variable': """
    // Global variable %(name)s
    %(type)s %(name)s;
    %(type)s *gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
"""
    }
}

# Initialization of parameters due to the init_population method.
#
# Parameters:
#
#    name: name of the variable
#    init: initial value
attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
        cudaMalloc(&gpu_%(name)s, size * sizeof(%(type)s));
        cudaMemcpy(gpu_%(name)s, %(name)s.data(), size * sizeof(%(type)s), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_%(name)s = cudaGetLastError();
        if ( err_%(name)s != cudaSuccess )
            std::cout << "    allocation of %(name)s failed: " << cudaGetErrorString(err_%(name)s) << std::endl;
    #endif
        // memory transfer flags
        %(name)s_host_to_device = false;
        %(name)s_device_to_host = t;
""",
    'global': {
        'parameter': """
        // Global parameter %(name)s
        %(name)s = 0.0;
""",
        'variable': """
        // Global variable %(name)s
        %(name)s = %(init)s;
        cudaMalloc(&gpu_%(name)s, sizeof(%(type)s));
        cudaMemcpy(gpu_%(name)s, &%(name)s, sizeof(%(type)s), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_%(name)s = cudaGetLastError();
        if ( err_%(name)s != cudaSuccess )
            std::cout << "    allocation of %(name)s failed: " << cudaGetErrorString(err_%(name)s) << std::endl;
    #endif
"""
    }
}

# We need to initialize the queue directly with the
# values (init) as the data arrays for the variables are
# only updated in front of a simulate call.
attribute_delayed = {
    'local': {
        'declare': """
    std::deque< %(type)s* > gpu_delayed_%(var)s; // list of gpu arrays""",
        'init': """
        gpu_delayed_%(name)s = std::deque< %(type)s* >(max_delay, NULL);
        std::vector<%(type)s> tmp_%(name)s = std::vector<%(type)s>( size, 0.0);
        for ( int i = 0; i < max_delay; i++ ) {
            cudaMalloc( (void**)& gpu_delayed_%(name)s[i], sizeof(%(type)s) * size);
            cudaMemcpy(gpu_delayed_%(name)s[i], tmp_%(name)s.data(), size * sizeof(%(type)s), cudaMemcpyHostToDevice);
        }
    #ifdef _DEBUG
        cudaError_t err_delay_%(name)s = cudaGetLastError();
        if (err_delay_%(name)s != cudaSuccess)
            std::cout << "pop%(id)s - init container for delayed %(name)s (max_delay = " << max_delay << " steps): " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
    #endif
""",
        'clear': """
for ( int i = 0; i < max_delay; i++ )
    cudaFree( gpu_delayed_%(name)s[i] );
gpu_delayed_%(name)s.clear();
gpu_delayed_%(name)s.shrink_to_fit();
""",
        'update': """
            %(type)s* last_%(name)s = gpu_delayed_%(name)s.back();
            gpu_delayed_%(name)s.pop_back();
            gpu_delayed_%(name)s.push_front(last_%(name)s);
            cudaMemcpy( last_%(name)s, gpu_%(name)s, sizeof(%(type)s) * size, cudaMemcpyDeviceToDevice );
        #ifdef _DEBUG
            cudaError_t err_delay_%(name)s = cudaGetLastError();
            if (err_delay_%(name)s != cudaSuccess)
                std::cout << "pop%(id)s - delay %(name)s: " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
        #endif
""",
        # Implementation notice:
        #    to ensure correctness of results, we need transfer from host here. The corresponding
        #    gpu arrays gpu_%(name)s are not resetted at this point of time (they will be resetted
        #    if simulate() invoked.
        'reset' : """
        // reset %(name)s
        for ( int i = 0; i < gpu_delayed_%(name)s.size(); i++ ) {
            cudaMemcpy( gpu_delayed_%(name)s[i], %(name)s.data(), sizeof(%(type)s) * size, cudaMemcpyHostToDevice );
        }
    #ifdef _DEBUG
        cudaError_t err_delay_%(name)s = cudaGetLastError();
        if ( err_delay_%(name)s != cudaSuccess )
            std::cout << "pop%(id)s - reset delayed %(name)s failed: " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
    #endif
""",
        'resize': """
        std::cerr << "ProjStruct::update_max_delay() is not implemented for local variables." << std::endl;
"""
    },
    'global': {
        'declare': """
    std::deque< %(type)s* > gpu_delayed_%(var)s; // list of gpu arrays""",
        'init': """
        gpu_delayed_%(name)s = std::deque< %(type)s* >(max_delay, NULL);
        %(type)s tmp_%(name)s = static_cast<%(type)s>(0.0);
        for ( int i = 0; i < max_delay; i++ ) {
            cudaMalloc( (void**)& gpu_delayed_%(name)s[i], sizeof(%(type)s));
            cudaMemcpy( gpu_delayed_%(name)s[i], &tmp_%(name)s, sizeof(%(type)s), cudaMemcpyHostToDevice );
        }
    #ifdef _DEBUG
        cudaError_t err_delay_%(name)s = cudaGetLastError();
        if (err_delay_%(name)s != cudaSuccess)
            std::cout << "pop%(id)s - init container for delayed %(name)s (max_delay = " << max_delay << " steps): " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
    #endif
""",
        'clear': """
for ( int i = 0; i < max_delay; i++ )
    cudaFree( gpu_delayed_%(name)s[i] );
gpu_delayed_%(name)s.clear();
gpu_delayed_%(name)s.shrink_to_fit();
""",
        'update': """
            %(type)s* last_%(name)s = gpu_delayed_%(name)s.back();
            gpu_delayed_%(name)s.pop_back();
            gpu_delayed_%(name)s.push_front(last_%(name)s);
            cudaMemcpy( last_%(name)s, gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToDevice );
        #ifdef _DEBUG
            cudaError_t err_delay_%(name)s = cudaGetLastError();
            if (err_delay_%(name)s != cudaSuccess)
                std::cout << "pop%(id)s - delay %(name)s: " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
        #endif
""",
        # Implementation notice:
        #    to ensure correctness of results, we need transfer from host here. The corresponding
        #    gpu arrays gpu_%(name)s are not resetted at this point of time (they will be resetted
        #    if simulate() invoked.
        'reset' : """
        // reset %(name)s
        for ( int i = 0; i < gpu_delayed_%(name)s.size(); i++ ) {
            cudaMemcpy( gpu_delayed_%(name)s[i], &%(name)s, sizeof(%(type)s), cudaMemcpyHostToDevice );
        }
    #ifdef _DEBUG
        cudaError_t err_delay_%(name)s = cudaGetLastError();
        if ( err_delay_%(name)s != cudaSuccess )
            std::cout << "pop%(id)s - reset delayed %(name)s failed: " << cudaGetErrorString(err_delay_%(name)s) << std::endl;
    #endif
""",
        'resize': """
        std::cerr << "ProjStruct::update_max_delay() is not implemented for global variables. " << std::endl;
"""
    }
}

# Transfer of variables before and after a simulation
#
# Parameters:
attribute_transfer = {
    'HtoD_local': """
        // %(attr_name)s: local
        if ( %(attr_name)s_host_to_device )
        {
        #if defined(_TRACE_INIT) || defined(_DEBUG)
            std::cout << "HtoD %(attr_name)s ( PopStruct%(id)s, t = " << t << " )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(attr_name)s, %(attr_name)s.data(), size * sizeof(%(type)s), cudaMemcpyHostToDevice);
            %(attr_name)s_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_%(attr_name)s = cudaGetLastError();
            if ( err_%(attr_name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(attr_name)s) << std::endl;
        #endif
        }
""",
    'HtoD_global': """
        // %(attr_name)s: global
        if ( %(attr_name)s_host_to_device )
        {
        #if defined(_TRACE_INIT) || defined(_DEBUG)
            std::cout << "HtoD: %(attr_name)s ( PopStruct%(id)s, t = " << t << " )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(attr_name)s, &%(attr_name)s, sizeof(%(type)s), cudaMemcpyHostToDevice);
            %(attr_name)s_host_to_device = false;

        #ifdef _DEBUG
            cudaError_t err_%(attr_name)s = cudaGetLastError();
            if ( err_%(attr_name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(attr_name)s) << std::endl;
        #endif
        }
""",
    'DtoH_local':"""
        // %(attr_name)s: local
        if (attr_name.compare("%(attr_name)s") == 0)
        {
            if (%(attr_name)s_device_to_host >= t)
                return;     // already up to date

        #if defined(_TRACE_INIT) || defined(_DEBUG)
            std::cout << "DtoH: %(attr_name)s ( PopStruct%(id)s, t = " << t << " )" << std::endl;
        #endif
            cudaMemcpy( %(attr_name)s.data(),  gpu_%(attr_name)s, size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
            %(attr_name)s_device_to_host = t;

        #ifdef _DEBUG
            cudaError_t err_%(attr_name)s = cudaGetLastError();
            if ( err_%(attr_name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(attr_name)s) << std::endl;
        #endif
            return;
        }
""",
    'DtoH_global':"""
    // %(attr_name)s: global
        if (attr_name.compare("%(attr_name)s") == 0)
        {
            if (%(attr_name)s_device_to_host >= t)
                return;     // already up to date

        #if defined(_TRACE_INIT) || defined(_DEBUG)
            std::cout << "DtoH: %(attr_name)s ( PopStruct%(id)s, t = " << t << " )" << std::endl;
        #endif
            cudaMemcpy( &%(attr_name)s,  gpu_%(attr_name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
            %(attr_name)s_device_to_host = t;

        #ifdef _DEBUG
            cudaError_t err_%(attr_name)s = cudaGetLastError();
            if ( err_%(attr_name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(attr_name)s) << std::endl;
        #endif
            return;
        }
"""
}

# Definition for the usage of CUDA device random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
curand = {
    'local': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, _nb_blocks * _threads_per_block * sizeof(curandState));
        init_curand_states( _nb_blocks, _threads_per_block, gpu_%(rd_name)s, global_seed );
""",
	'clear': """
cudaFree(gpu_%(rd_name)s);
""",
    },
    'global': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, sizeof(curandState));
        init_curand_states(1, 1, gpu_%(rd_name)s, global_seed );
#ifdef _DEBUG
        cudaError_t err_%(rd_name)s = cudaGetLastError();
        if ( err_%(rd_name)s != cudaSuccess )
            std::cout << "pop%(id)s - init_population: " << cudaGetErrorString(err_%(rd_name)s) << std::endl;
#endif
""",
	'clear': ""
    }
}

spike_specific = {
    'spike': {
        'declare':"""
    // Structures for managing spikes
    std::vector<long int> last_spike;
    long int* gpu_last_spike;
    std::vector<int> spiked;
    int* gpu_spiked;
    unsigned int spike_count;
    unsigned int* gpu_spike_count;
""",
        'init': """
        // Spiking variables
        spiked = std::vector<int>(size, 0);
        cudaMalloc((void**)&gpu_spiked, size * sizeof(int));
        cudaMemcpy(gpu_spiked, spiked.data(), size * sizeof(int), cudaMemcpyHostToDevice);

        last_spike = std::vector<long int>(size, -10000L);
        cudaMalloc((void**)&gpu_last_spike, size * sizeof(long int));
        cudaMemcpy(gpu_last_spike, last_spike.data(), size * sizeof(long int), cudaMemcpyHostToDevice);

        spike_count = 0;
        cudaMalloc((void**)&gpu_spike_count, sizeof(unsigned int));
        cudaMemcpy(gpu_spike_count, &spike_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
""",
        'reset': """
        spiked = std::vector<int>(size, 0);
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);
        spike_count = 0;
""",
        'clear': """
// Spike events (host-side)
spiked.clear();
spiked.shrink_to_fit();

last_spike.clear();
last_spike.shrink_to_fit();

// Spike events (device-side)
cudaFree(gpu_spiked);
cudaFree(gpu_last_spike);
"""
    },
    'refractory': {
        'declare': """
    // Refractory period
    std::vector<int> refractory;
    int *gpu_refractory;
    bool refractory_dirty;
    std::vector<int> refractory_remaining;
    int *gpu_refractory_remaining;
""",
        'init': """
        // Refractory period
        refractory = std::vector<int>(size, 0);
        cudaMalloc((void**)&gpu_refractory, size * sizeof(int));
        refractory_remaining = std::vector<int>(size, 0);
        cudaMemcpy(gpu_refractory, refractory.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        refractory_dirty = false;
        cudaMalloc((void**)&gpu_refractory_remaining, size * sizeof(int));
        cudaMemcpy(gpu_refractory_remaining, refractory_remaining.data(), size * sizeof(int), cudaMemcpyHostToDevice);
""",
        'init_extern': """
        // Refractory period
        refractory_remaining = std::vector<int>(size, 0);
        cudaMalloc((void**)&gpu_refractory_remaining, size * sizeof(int));
        cudaMemcpy(gpu_refractory_remaining, refractory_remaining.data(), size * sizeof(int), cudaMemcpyHostToDevice);
""",
        'reset': """
        refractory_remaining.clear();
        refractory_remaining = std::vector<int>(size, 0);
        cudaMemcpy(gpu_refractory_remaining, refractory_remaining.data(), size * sizeof(int), cudaMemcpyHostToDevice);
"""
    }
}

# Contains all codes related to the population update
#
# 1st level distinguish 'local' and 'global' update
# 2nd level distinguish 'device_kernel', 'invoke_kernel', 'header' and 'host_call' template
population_update_kernel = {
    'global': {
        'device_kernel': """// Updating global variables of population %(id)s
__global__ void cuPop%(id)s_global_step( %(add_args)s )
{
%(pre_loop)s

%(global_eqs)s
}
""",
        'invoke_kernel': """void pop%(id)s_global_step(RunConfig cfg, %(add_args)s ) {
    cuPop%(id)s_global_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        // arguments
        %(add_args_call)s
    );
}""",
        'kernel_decl': """void pop%(id)s_global_step(RunConfig cfg, %(add_args)s );
""",
        'host_call': """
        pop%(id)s_global_step(RunConfig(1, 1, 0, pop%(id)s->stream), %(add_args)s );
    #ifdef _DEBUG
        cudaError_t err_pop%(id)s_global_step = cudaGetLastError();
        if( err_pop%(id)s_global_step != cudaSuccess) {
            std::cout << "pop%(id)s_step: " << cudaGetErrorString(err_pop%(id)s_global_step) << std::endl;
            exit(0);
        }
    #endif
"""
    },
    'local': {
        'device_kernel': """// Updating local variables of population %(id)s
__global__ void cuPop%(id)s_local_step( %(add_args)s )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i = tid;
%(pre_loop)s

    while ( i < %(pop_size)s )
    {
%(local_eqs)s

        i += blockDim.x * gridDim.x;
    }

%(post_loop)s
}
""",
        'invoke_kernel': """
void pop%(id)s_local_step(RunConfig cfg, %(add_args)s ) {
    cuPop%(id)s_local_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        // arguments
        %(add_args_call)s
    );
}
""",
        'kernel_decl': "void pop%(id)s_local_step(RunConfig cfg, %(add_args)s );\n",
        'host_call': """
    #if defined (__pop%(id)s_nb__)
        pop%(id)s_local_step(RunConfig(__pop%(id)s_nb__, __pop%(id)s_tpb__, 0, pop%(id)s->stream), %(add_args)s );
    #else
        pop%(id)s_local_step(RunConfig(pop%(id)s->_nb_blocks, pop%(id)s->_threads_per_block, 0, pop%(id)s->stream), %(add_args)s );
    #endif

    #ifdef _DEBUG
        cudaError_t err_pop%(id)s_local_step = cudaGetLastError();
        if( err_pop%(id)s_local_step != cudaSuccess) {
            std::cout << "pop%(id)s_step: " << cudaGetErrorString(err_pop%(id)s_local_step) << std::endl;
            exit(0);
        }
    #endif
"""
    }
}

spike_gather_kernel = {
    'device_kernel': """
// gpu device kernel for population %(id)s
__global__ void cu_pop%(id)s_spike_gather( unsigned int* num_events, const long int t, const %(float_prec)s dt, int* spiked, long int* last_spike%(args)s )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_spiked[32];
    __shared__ int num_local_events[1];
    num_local_events[0] = 0;
    __syncthreads();

    // Determine if neuron i emited a spike
    while ( i < %(pop_size)s )
    {
        if ( %(cond)s ) {
            %(reset)s

            // store spike event
            int pos = atomicAdd ( &num_local_events[0], 1);
            local_spiked[pos] = i;
            last_spike[i] = t;

            // refractory
            %(refrac_inc)s
        }

        __syncthreads();
        if ((threadIdx.x == 0) & (num_local_events[0] > 0)) {
            int pos = atomicAdd ( num_events, num_local_events[0]);
            for (int j = 0; j < num_local_events[0]; j++)
                spiked[pos+j] = local_spiked[j];
            num_local_events[0] = 0;
        }
        __syncthreads();
        i += gridDim.x * blockDim.x;
    }
}
""",
    'invoke_kernel': """
void pop%(id)s_spike_gather(RunConfig cfg, unsigned int* num_events, const long int t, const %(float_prec)s dt, int* spiked, long int* last_spike%(args)s ) {
    // Compute current events
    cu_pop%(id)s_spike_gather<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(
        /* default arguments */
        num_events, t, dt, spiked, last_spike
        /* other variables */
        %(args_call)s
    );
}
""",
    'kernel_decl': """
void pop%(id)s_spike_gather(RunConfig cfg, unsigned int* num_events, %(default)s%(args)s );
""",
    # As we use atomicAdd operations, multiple blocks are not
    # working correctly, consequently spawn only one block.
    'host_call': """
    // Check if neurons emit a spike in population %(id)s
    if ( pop%(id)s->_active ) {
        // Reset old events
        call_clear_num_events(RunConfig(1, 1, 0, pop%(id)s->stream), pop%(id)s->gpu_spike_count);

        // launch configuration
%(launch_config)s

        // Compute current events
        pop%(id)s_spike_gather(
            /* kernel config */
            RunConfig(nb_blocks, tpb, 0, pop%(id)s->stream),
            /* number of emeitted events (return value) */
            pop%(id)s->gpu_spike_count,
            /* default arguments */
            %(default)s
            /* other variables */
            %(args)s
        );

    #ifdef _DEBUG
        cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
        if(err_pop_spike_gather_%(id)s != cudaSuccess)
            std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
    #endif

        // transfer back the spike counter (needed by record as well as launch psp - kernel)
        cudaMemcpy( &pop%(id)s->spike_count, pop%(id)s->gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost );
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "record_spike_count: " << cudaGetErrorString(err) << std::endl;
    #endif

        // transfer back the spiked indices if there were any events (needed by record)
        if (pop%(id)s->spike_count > 0)
        {
            cudaMemcpy( pop%(id)s->spiked.data(), pop%(id)s->gpu_spiked, pop%(id)s->spike_count*sizeof(int), cudaMemcpyDeviceToHost );
        #ifdef _DEBUG
            err = cudaGetLastError();
            if ( err != cudaSuccess )
                std::cout << "record_spike: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
    }
"""
}

#
# Final dictionary
cuda_templates = {
    'population_header': population_header,
    'attr_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_delayed': attribute_delayed,
    'attribute_transfer': attribute_transfer,
    'rng': curand,

    'spike_specific': spike_specific
}
