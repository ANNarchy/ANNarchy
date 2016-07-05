"""

    PopulationTemplate.py

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
# Definition of a population as a c-like struct, divided
# into two groups: rate or spike
#
# Parameters:
#
#    id: id of the population
#    additional: neuron specific definitions
#    accessors: set of functions to export population data to python
header_struct_omp = """#pragma once
#include <random>
%(include_additional)s
%(include_profile)s
extern double dt;
extern long int t;
extern std::mt19937 rng;
%(extern_global_operations)s
%(struct_additional)s
///////////////////////////////////////////////////////////////
// Main Structure for the population of id %(id)s (%(name)s)
///////////////////////////////////////////////////////////////
struct PopStruct%(id)s{
    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }
%(declare_spike_arrays)s
    // Neuron specific parameters and variables
%(declare_parameters_variables)s
%(declare_delay)s
%(declare_FR)s
%(declare_additional)s
%(declare_profile)s
    // Access methods to the parameters and variables
%(access_parameters_variables)s

    // Method called to initialize the data structures
    void init_population() {
        size = %(size)s;
        _active = true;
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
    }

    // Method to draw new random numbers
    void update_rng() {
%(update_rng)s
    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {
%(update_global_ops)s
    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {
%(update_delay)s
    }

    // Main method to update neural variables
    void update() {
%(update_variables)s
    }

    %(stop_condition)s
};
"""

header_struct_cuda = """#pragma once
extern double dt;
extern long int t;

// RNG - defined in ANNarchy.cu
extern long seed;
extern void init_curand_states( int N, curandState* states, unsigned long seed );

%(include_additional)s
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
    // Access methods to the parameters and variables
%(access_parameters_variables)s

    // Method called to initialize the data structures
    void init_population() {
        size = %(size)s;
        _active = true;
%(init_parameters_variables)s
%(init_spike)s
%(init_delay)s
%(init_additional)s
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

    %(stop_condition)s
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
attribute_decl = {
    'openmp': {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
    },
    'cuda': {
    'local': """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s *gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global': """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
"""
    }
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
# TODO:
#
#    For CUDA we currently double the memory: python accesses modify the host
#    data. Only before and after simulate() the GPU memory is involved. To speedup
#    host-to-device transfers, we use a 'dirty' flag to mark changed variables.
attribute_acc = {
    'openmp':{
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s val) { %(name)s = val; }
"""
    },
    'cuda':{
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
    void set_%(name)s(%(type)s val) { %(name)s = val; }
"""
    }
}

# export of accessors for parameter members towards python, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
attribute_cpp_export = {
    'local':
"""
        # Local %(attr_type)s %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_single_%(name)s(int rk)
        void set_%(name)s(vector[%(type)s])
        void set_single_%(name)s(int, %(type)s)
""",
    'global':
"""
        # Global %(attr_type)s %(name)s
        %(type)s  get_%(name)s()
        void set_%(name)s(%(type)s)
"""
}

# export of accessors for parameter members towards python, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons. Functions marked as cpdef
# can be accessed from python as well as cython. Local parameters allows access to single as well as all values.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
attribute_pyx_wrapper = {
    'local':
"""
    # Local %(attr_type)s %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.get_%(name)s())
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.set_%(name)s( value )
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.get_single_%(name)s(rank)
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.set_single_%(name)s(rank, value)
""",
    'global':
"""
    # Global %(attr_type)s %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.get_%(name)s()
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.set_%(name)s(value)
"""
}

# Initialization of parameters due to the init_population method.
#
# Parameters:
#
#    name: name of the variable
#    init: initial value
attribute_cpp_init = {
    'openmp':
    {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
    },
    'cuda':
    {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
        cudaMalloc(&gpu_%(name)s, size * sizeof(double));
        cudaMemcpy(gpu_%(name)s, %(name)s.data(), size * sizeof(double), cudaMemcpyHostToDevice);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
    }
}

attribute_delayed = {
    'openmp':{
        'local': """
        _delayed_%(var)s = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(size, 0.0));""",
        'global': """
        _delayed_%(var)s = std::deque< double >(%(delay)s, 0.0);""",
        'reset' : """
        for ( int i = 0; i < _delayed_%(var)s.size(); i++ ) {
            _delayed_%(var)s[i] = %(var)s;
        }
        """
    },
    'cuda':{
        'local': """
    gpu_delayed_%(var)s = std::deque< double* >(%(delay)s, NULL);
    for ( int i = 0; i < %(delay)s; i++ )
        cudaMalloc( (void**)& gpu_delayed_%(var)s[i], sizeof(double) * size);
""",
        'global': "//TODO: implement code template",
        'reset' : """
    for ( int i = 0; i < gpu_delayed_%(var)s.size(); i++ ) {
        cudaMemcpy( gpu_delayed_%(var)s[i], gpu_%(var)s, sizeof(double) * size, cudaMemcpyDeviceToDevice );
    }
    """
    }
}

# Definition for the usage of C++11 STL template random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
cpp_11_rng = {
    'local': {
        'decl': """    std::vector<double> %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = std::vector<double>(size, 0.0);
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
                %(rd_name)s[i] = dist_%(rd_name)s(rng);
    """
    },
    'global': {
        'decl': """    double %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = 0.0;
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
            %(rd_name)s = dist_%(rd_name)s(rng);
    """
    }
}

# Definition for the usage of CUDA device random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
cuda_rng = {
    'local': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, size * sizeof(curandState));
        init_curand_states( size, gpu_%(rd_name)s, seed );
#ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "pop%(id)s - init_population: " << cudaGetErrorString(err) << std::endl;
#endif
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

cuda_pop_kernel=\
"""
// gpu device kernel for population %(id)s
__global__ void cuPop%(id)s_step(double dt%(tar)s%(var)s%(par)s)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Updating global variables of population %(id)s
%(global_eqs)s

    // Updating local variables of population %(id)s
    if ( i < %(pop_size)s )
    {
%(local_eqs)s
    }
}
"""

cuda_pop_kernel_call =\
"""
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s._active ) {
        int nb = ceil ( double( pop%(id)s.size ) / (double)__pop%(id)s__ );

        cuPop%(id)s_step<<<nb, __pop%(id)s__>>>(/* default arguments */
              dt
              /* population targets */
              %(tar)s
              /* kernel gpu arrays */
              %(var)s
              /* kernel constants */
              %(par)s );
    }

#ifdef _DEBUG
    cudaError_t err_pop_step_%(id)s = cudaGetLastError();
    if(err_pop_step_%(id)s != cudaSuccess)
        std::cout << cudaGetErrorString(err_pop_step_%(id)s) << std::endl;
#endif
"""

# Rate respectively spiking models require partly special variables or have different operations.
# This dictionary contain these unique initializations.
#
# Parameters (may differ):
#
#     id: id of the population
#     target: target name (e. g. FF, LAT ...)
spike_specific = {
    'declare_spike': """
    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;
""",
    'init_spike': """
        // Spiking variables
        spiked = std::vector<int>(0, 0);
        last_spike = std::vector<long int>(size, -10000L);
""",
    'declare_refractory': """
    // Refractory period
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;""",
    'init_refractory': """
        // Refractory period
        refractory = std::vector<int>(size, 0);
        refractory_remaining = std::vector<int>(size, 0);
""",
    'init_event-driven': """
        last_spike = std::vector<long int>(size, -10000L);
""",
    'reset_spike': """
        spiked.clear();
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);
""",
    'reset_refractory': """
        refractory_remaining.clear();
        refractory_remaining = std::vector<int>(size, 0);
"""
}

rate_psp = {
    'openmp':
    {
        'decl': """
    std::vector<double> _sum_%(target)s;""",
        'init': """
        // Post-synaptic potential
        _sum_%(target)s = std::vector<double>(size, 0.0);""",
    },
    'cuda':
    {
        'decl': """
    std::vector<double> _sum_%(target)s;
    double* gpu_sum_%(target)s;
        """,
        'init': """
        // Post-synaptic potential
        _sum_%(target)s = std::vector<double>(size, 0.0);
        cudaMalloc((void**)&gpu_sum_%(target)s, size * sizeof(double));
        cudaMemcpy(gpu_sum_%(target)s, _sum_%(target)s.data(), size * sizeof(double), cudaMemcpyHostToDevice);""",
    }
}
