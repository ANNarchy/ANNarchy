#===============================================================================
#
#     CUDATemplates.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
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
#===============================================================================
population_header = """#pragma once
extern double dt;
extern long int t;

// RNG - defined in ANNarchy.cu
extern long seed;
extern void init_curand_states( int N, curandState* states, unsigned long seed );

%(include_additional)s
%(include_profile)s

%(extern_global_operations)s
%(struct_additional)s

///////////////////////////////////////////////////////////////
// Main Structure for the population of id %(id)s (%(name)s)
///////////////////////////////////////////////////////////////
struct PopStruct%(id)s{
    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    cudaStream_t stream; // assigned stream for concurrent kernel execution ( CC > 2.x )

    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }
%(declare_spike_arrays)s
    // Neuron specific parameters and variables
%(declare_parameters_variables)s
%(declare_delay)s
%(declare_additional)s

    // Profiling
%(declare_profile)s

    // Access methods to the parameters and variables
%(access_parameters_variables)s
%(access_additional)s

    // Method called to initialize the data structures
    void init_population() {
        size = %(size)s;
        _active = true;
%(init_parameters_variables)s
%(init_spike)s
%(init_delay)s
%(init_additional)s
%(init_profile)s
    }

    // Method called to reset the population
    void reset() {
%(reset_spike)s
%(reset_delay)s
%(reset_additional)s
    }

    // Method to draw new random numbers
    void update_rng() {
%(update_rng)s
    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {
%(update_delay)s
    }

    // Main method to update neural variables
    void update() {
%(update_variables)s
    }

    // Stop condition
    %(stop_condition)s

    // Memory transfers
    void host_to_device() {
%(host_to_device)s
    }

    void device_to_host() {
%(device_to_host)s
    }
};
"""

# c like definition of neuron attributes, whereas 'local' is used if values can vary across
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
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global': """
    // Global parameter %(name)s
    %(type)s %(name)s;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;    
"""
}

# c like definition of accessors for neuron attributes, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons. Currently two
# types of sets are defined: openmp and cuda. In cuda case additional 'dirty' flags are created for
# each variable (set to true, in case of setters).
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
#
attribute_acc = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; %(name)s_dirty = true; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; %(name)s_dirty = true; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s val) { %(name)s = val; %(name)s_dirty = true; }
"""
}


# Initialization of parameters due to the init_population method.
#
# Parameters:
#
#    name: name of the variable
#    init: initial value
attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
        cudaMalloc(&gpu_%(name)s, size * sizeof(%(type)s));
        cudaMemcpy(gpu_%(name)s, %(name)s.data(), size * sizeof(%(type)s), cudaMemcpyHostToDevice);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
        cudaMalloc(&gpu_%(name)s, sizeof(%(type)s));
        cudaMemcpy(gpu_%(name)s, &%(name)s, sizeof(%(type)s), cudaMemcpyHostToDevice);
"""
}

# We need to initialize the queue directly with the
# values (init) as the data arrays for the variables are
# only updated in front of a simulate call.
attribute_delayed = {
   'local': """
        gpu_delayed_%(var)s = std::deque< %(type)s* >(%(delay)s, NULL);
        std::vector< %(type)s > tmp = std::vector< %(type)s >( size, %(init)s );
        for ( int i = 0; i < %(delay)s; i++ ) {
            cudaMalloc( (void**)& gpu_delayed_%(var)s[i], sizeof(%(type)s) * size);
            cudaMemcpy( gpu_delayed_%(var)s[i], tmp.data(), sizeof(%(type)s) * size, cudaMemcpyHostToDevice );
        }
        tmp.clear();
""",
    'global': "//TODO: implement code template",
    'update': """
            %(type)s* last_%(var)s = gpu_delayed_%(var)s.back();
            gpu_delayed_%(var)s.pop_back();
            gpu_delayed_%(var)s.push_front(last_%(var)s);
            cudaMemcpy( last_%(var)s, gpu_%(var)s, sizeof(%(type)s) * size, cudaMemcpyDeviceToDevice );
        #ifdef _DEBUG
            cudaError_t err_%(var)s = cudaGetLastError();
            if (err_%(var)s != cudaSuccess)
                std::cout << "pop%(id)s - delay %(var)s: " << cudaGetErrorString(err_%(var)s) << std::endl;
        #endif
""",
    'reset' : """
        std::vector< %(type)s > tmp = std::vector< %(type)s >( size, %(init)s );
        for ( int i = 0; i < gpu_delayed_%(var)s.size(); i++ ) {
            cudaMemcpy( gpu_delayed_%(var)s[i], tmp.data(), sizeof(%(type)s) * size, cudaMemcpyHostToDevice );
        }
        tmp.clear();
"""
}

# Transfer of variables before and after a simulation
#
# Parameters:
attribute_transfer = {
    'HtoD_local': """
        // %(attr_name)s: local
        if( %(attr_name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD %(attr_name)s ( pop%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(attr_name)s, %(attr_name)s.data(), size * sizeof(%(type)s), cudaMemcpyHostToDevice);
            %(attr_name)s_dirty = false;

        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
    """,
    'HtoD_global': """
        // %(attr_name)s: global
        if( %(attr_name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD %(attr_name)s ( pop%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(attr_name)s, &%(attr_name)s, sizeof(%(type)s), cudaMemcpyHostToDevice);
            %(attr_name)s_dirty = false;

        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
    """,
    'DtoH_local':"""
    // %(attr_name)s: local
    cudaMemcpy( %(attr_name)s.data(),  gpu_%(attr_name)s, size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
    """,
    'DtoH_global':"""
    // %(attr_name)s: global
    cudaMemcpy( &%(attr_name)s,  gpu_%(attr_name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
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
        cudaMalloc((void**)&gpu_%(rd_name)s, size * sizeof(curandState));
        init_curand_states( size, gpu_%(rd_name)s, seed );
"""
    },
    'global': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, sizeof(curandState));
        init_curand_states( 1, gpu_%(rd_name)s, seed );
#ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "pop%(id)s - init_population: " << cudaGetErrorString(err) << std::endl;
#endif
"""
    }
}

rate_psp = {
   'decl': """
    std::vector<double> _sum_%(target)s;
    double* gpu__sum_%(target)s;
""",
   'init': """
        // Post-synaptic potential
        _sum_%(target)s = std::vector<double>(size, 0.0);
        cudaMalloc((void**)&gpu__sum_%(target)s, size * sizeof(double));
        cudaMemcpy(gpu__sum_%(target)s, _sum_%(target)s.data(), size * sizeof(double), cudaMemcpyHostToDevice);
"""
}

spike_specific = {
    'declare_spike': """
    // Structures for managing spikes
    std::vector<long int> last_spike;
    long int* gpu_last_spike;
    std::vector<int> spiked;
    int* gpu_spiked;
""",
    'init_spike': """
        // Spiking variables
        spiked = std::vector<int>(size, 0);
        cudaMallocHost((void**)&gpu_spiked, size * sizeof(int));
        cudaMemcpyAsync(gpu_spiked, spiked.data(), size * sizeof(int), cudaMemcpyHostToDevice, 0);

        last_spike = std::vector<long int>(size, -10000L);
        cudaMalloc((void**)&gpu_last_spike, size * sizeof(long int));
        cudaMemcpy(gpu_last_spike, last_spike.data(), size * sizeof(long int), cudaMemcpyHostToDevice);
""",
    'declare_refractory': """
    // Refractory period
    std::vector<int> refractory;
    int *gpu_refractory;
    bool refractory_dirty;
    std::vector<int> refractory_remaining;
    int *gpu_refractory_remaining;""",
    'init_refractory': """
        // Refractory period
        refractory = std::vector<int>(size, 0);
        cudaMalloc((void**)&gpu_refractory, size * sizeof(int));
        refractory_remaining = std::vector<int>(size, 0);
        cudaMemcpy(gpu_refractory, refractory.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        refractory_dirty = false;
        cudaMalloc((void**)&gpu_refractory_remaining, size * sizeof(int));
        cudaMemcpy(gpu_refractory_remaining, refractory_remaining.data(), size * sizeof(int), cudaMemcpyHostToDevice); 
""",
    'init_event-driven': """
        last_spike = std::vector<long int>(size, -10000L);
""",
    'reset_spike': """
        spiked = std::vector<int>(size, 0);
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);
""",
    'reset_refractory': """
        refractory_remaining.clear();
        refractory_remaining = std::vector<int>(size, 0);
        cudaMemcpy(gpu_refractory_remaining, refractory_remaining.data(), size * sizeof(int), cudaMemcpyHostToDevice);
""",
    'pyx_wrapper': """
    # Refractory period
    cpdef np.ndarray get_refractory(self):
        return pop%(id)s.refractory
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value
        pop%(id)s.refractory_dirty = True
"""
}

population_update_kernel = \
"""
// gpu device kernel for population %(id)s
__global__ void cuPop%(id)s_step(%(default)s%(refrac)s%(tar)s%(var)s%(par)s)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Updating global variables of population %(id)s
%(global_eqs)s
    __syncthreads();

    // Updating local variables of population %(id)s
    while ( i < %(pop_size)s )
    {
%(local_eqs)s

        i += blockDim.x * gridDim.x;
    }
}
"""

population_update_header = """
__global__ void cuPop%(id)s_step( %(default)s%(refrac)s%(var)s%(par)s );
"""

population_update_call = \
"""
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s._active ) {
        cuPop%(id)s_step<<< __pop%(id)s_nb__, __pop%(id)s_tpb__, 0, %(stream_id)s >>>(
              /* default arguments */
              %(default)s
              /* refractoriness (only spike) */
              %(refrac)s
              /* targets (only rate-code) */
              %(tar)s
              /* kernel variables */
              %(var)s
              /* kernel constants */
              %(par)s );

        #ifdef _DEBUG
            cudaError_t err_pop_step_%(id)s = cudaGetLastError();
            if(err_pop_step_%(id)s != cudaSuccess) {
                std::cout << "pop%(id)s_step: " << cudaGetErrorString(err_pop_step_%(id)s) << std::endl;
                exit(0);
            }
        #endif
    }
"""

spike_gather_kernel = \
"""
// gpu device kernel for population %(id)s
__global__ void cuPop%(id)s_spike_gather( %(default)s%(refrac)s%(args)s )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Determine if neuron i emited a spike 
    while ( i < %(pop_size)s )
    {
%(spike_gather)s

        i += gridDim.x * blockDim.x;
    }
}
"""

spike_gather_header = """
__global__ void cuPop%(id)s_spike_gather( %(default)s%(refrac)s%(args)s );
"""

spike_gather_call = \
"""
    // Check if neurons emit a spike in population %(id)s
    if ( pop%(id)s._active ) {
        cuPop%(id)s_spike_gather<<< 1, __pop%(id)s_tpb__, 0, streams[%(stream_id)s] >>>(
              /* default arguments */
              %(default)s
              /* refractoriness */
              %(refrac)s
              /* other variables */
              %(args)s );
    }
#ifdef _DEBUG
    cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
    if(err_pop_spike_gather_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
#endif

    // transfer back the spiked array (needed by record)
    cudaMemcpyAsync( pop%(id)s.spiked.data(), pop%(id)s.gpu_spiked, pop%(id)s.size*sizeof(int), cudaMemcpyDeviceToHost, streams[%(stream_id)s]);
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike: " << cudaGetErrorString(err) << std::endl;
#endif
"""

#
# Final dictionary
cuda_templates = {
    'population_header': population_header,
    'attr_decl': attribute_decl,
    'attr_acc': attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_delayed': attribute_delayed,
    'attribute_transfer': attribute_transfer,
    'rng': curand,

    'rate_psp': rate_psp,
    'spike_specific': spike_specific
}
