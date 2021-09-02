#===============================================================================
#
#     OpenMPTemplates.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2021  Julien Vitay <julien.vitay@gmail.com>,
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


# Definition of a population as a c-like struct, divided
# into two groups: rate or spike
#
# Parameters:
#
#    id: id of the population
#    additional: neuron specific definitions
#    accessors: set of functions to export population data to python
population_header = """/*
 *  ANNarchy-version: %(annarchy_version)s
 */
#pragma once
#include "ANNarchy.h"
#include <random>
%(include_additional)s
%(include_profile)s
extern %(float_prec)s dt;
extern long int t;
extern int global_num_threads;
extern std::vector<std::mt19937> rng;
%(extern_global_operations)s
%(struct_additional)s
///////////////////////////////////////////////////////////////
// Main Structure for the population of id %(id)s (%(name)s)
///////////////////////////////////////////////////////////////
struct PopStruct%(id)s{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

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
%(declare_profile)s
    // Access methods to the parameters and variables
%(access_parameters_variables)s
%(access_additional)s

    // Method called to initialize the data structures
    void init_population() {
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

    // Init rng dist
    void init_rng_dist() {
%(init_rng_dist)s
    }

    // Method to draw new random numbers
    void update_rng(int tid) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct%(id)s::update_rng()" << std::endl;
#endif
%(update_rng)s
    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops(int tid, int nt) {
%(update_global_ops)s
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
    void update(int tid) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct%(id)s::update()" << std::endl;
#endif
%(update_variables)s
    }

    void spike_gather(int tid, int num_threads) {
%(test_spike_cond)s
    }

    %(stop_condition)s

    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(determine_size)s
        return size_in_bytes;
    }

    // Memory management: track the memory consumption
    void clear() {
%(clear_container)s
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
    'local_get_all': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            return %(name)s;
        }
""",
    'local_get_single': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            return %(name)s[rk];
        }
""",
    'local_set_all': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(name)s = value;
            return;
        }
""",
    'local_set_single': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(name)s[rk] = value;
            return;
        }
""",
    'global_get': """
        // Global %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            return %(name)s;
        }
""",
    'global_set': """
        // Global %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(name)s = value;
            return;
        }
"""
}

# This function offers a generic function per data-type to the Python frontend which
# should return the data based on the variable name.
#
# Parameters:
#
#   ctype:      data type of the variable (double, float, int ...)
#   ctype_name: function names should not contain spaces like in unsigned int is therefore transformed to unsigned_int
#   id:         object ID
attribute_template = {
    'local': """
    std::vector<%(ctype)s> get_local_attribute_all_%(ctype_name)s(std::string name) {
%(local_get1)s

        // should not happen
        std::cerr << "PopStruct%(id)s::get_local_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<%(ctype)s>();
    }

    %(ctype)s get_local_attribute_%(ctype_name)s(std::string name, int rk) {
        assert( (rk < size) );
%(local_get2)s

        // should not happen
        std::cerr << "PopStruct%(id)s::get_local_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return static_cast<%(ctype)s>(0.0);
    }

    void set_local_attribute_all_%(ctype_name)s(std::string name, std::vector<%(ctype)s> value) {
        assert( (value.size() == size) );
%(local_set1)s

        // should not happen
        std::cerr << "PopStruct%(id)s::set_local_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
    }

    void set_local_attribute_%(ctype_name)s(std::string name, int rk, %(ctype)s value) {
        assert( (rk < size) );
%(local_set2)s

        // should not happen
        std::cerr << "PopStruct%(id)s::set_local_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
    }
""",
    'global': """
    %(ctype)s get_global_attribute_%(ctype_name)s(std::string name) {
%(global_get)s

        // should not happen
        std::cerr << "PopStruct%(id)s::get_global_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return static_cast<%(ctype)s>(0.0);
    }

    void set_global_attribute_%(ctype_name)s(std::string name, %(ctype)s value)  {
%(global_set)s

        std::cerr << "PopStruct%(id)s::set_global_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
    }
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
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
}

attribute_delayed = {
    'local': {
        'init': """
        _delayed_%(name)s = std::deque< std::vector< %(type)s > >(max_delay, std::vector< %(type)s >(size, 0.0));""",

        'update': """
        #pragma omp single
        {
            _delayed_%(name)s.push_front(%(name)s);
            _delayed_%(name)s.pop_back();
        }
""",
        'reset' : """
        for ( int i = 0; i < _delayed_%(name)s.size(); i++ ) {
            _delayed_%(name)s[i] = %(name)s;
        }
""",
        'resize' : """
    _delayed_%(name)s.resize(max_delay, std::vector< %(type)s >(size, 0.0));
"""
    },
    'global':{
        'init': """
        _delayed_%(name)s = std::deque< %(type)s >(max_delay, 0.0);""",
        'update': """
        #pragma omp single
        {
            _delayed_%(name)s.push_front(%(name)s);
            _delayed_%(name)s.pop_back();
        }
""",
        'reset' : """
        for ( int i = 0; i < _delayed_%(name)s.size(); i++ ) {
            _delayed_%(name)s[i] = %(name)s;
        }
""",
        'resize' : """
    _delayed_%(name)s.resize(max_delay, 0.0);
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
        'decl': """    std::vector<%(type)s> %(rd_name)s;
    std::vector<%(template)s> dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = std::vector<%(type)s>(size, 0.0);
    """,
        'init_dist': """
        dist_%(rd_name)s = std::vector< %(template)s >(global_num_threads);
        #pragma omp parallel num_threads(global_num_threads)
        {
            dist_%(rd_name)s[omp_get_thread_num()] = %(rd_init)s;
        }
    """,
        'update': """
                %(rd_name)s[i] = dist_%(rd_name)s[%(index)s](rng[%(index)s]);
    """
    },
    'global': {
        'decl': """    %(type)s %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = 0.0;
    """,
        'init_dist': """
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
            %(rd_name)s = dist_%(rd_name)s(rng[0]);
    """
    },
    'omp_code_seq': """
        if (_active){

            #pragma omp single
            {
%(update_rng_global)s
                for(int i = 0; i < size; i++) {
%(update_rng_local)s
                }
            }
        }
    """,
    'omp_code_par': """
        if (_active){
            #pragma omp single nowait
            {
%(update_rng_global)s
            }

            #pragma omp for
            for (int i = 0; i < size; i++) {
%(update_rng_local)s
            }
        }
    """     
}

rate_psp = {
    'decl': """
    std::vector<%(float_prec)s> _sum_%(target)s;""",
    'init': """
        // Post-synaptic potential
        _sum_%(target)s = std::vector<%(float_prec)s>(size, 0.0);""",
    'reset': """
    // pop%(id)s: %(name)s
    #pragma omp single nowait
    {
        if (pop%(id)s._active)
            memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(%(float_prec)s));
    }
"""
}

spike_specific = {
    'spike': {
        'declare': """
    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> local_spiked_sizes;
""",
        'init': """
        // Spiking variables
        spiked = std::vector<int>();
        local_spiked_sizes = std::vector<int>(global_num_threads+1, 0);
        last_spike = std::vector<long int>(size, -10000L);
""",
        'reset': """
        spiked.clear();
        spiked.shrink_to_fit();
        local_spiked_sizes = std::vector<int>(global_num_threads+1, 0);
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);
"""
    },
    'axon_spike': {
        'declare': """
    // Structures for managing axonal spikes
    std::vector<int> axonal;
""",
        'init': """
        // Axonal spike containter
        axonal = std::vector<int>();
""",
        'reset': """
        axonal.clear();
        axonal.shrink_to_fit();
""",
        'pyx_wrapper': """
    # Axonal spike events
"""
    },    
    'refractory': {

        'declare': """
    // Refractory period
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    std::vector<short int> in_ref;
""",

        'init': """
        // Refractory period
        refractory = std::vector<int>(size, 0);
        refractory_remaining = std::vector<int>(size, 0);
        in_ref = std::vector<short int>(size, 0);
""",

        # If the refractory variable is defined by the user
        'init_extern': """
        // Refractory period
        refractory_remaining = std::vector<int>(size, 0);
        in_ref = std::vector<short int>(size, 0);
""",

        'reset': """
        // Refractory period
        refractory_remaining.clear();
        refractory_remaining = std::vector<int>(size, 0);
""",

        'pyx_wrapper': """
    # Refractory period
    cpdef np.ndarray get_refractory(self):
        return np.array(pop%(id)s.refractory)
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value
"""
    },
    'init_event-driven': """
        last_spike = std::vector<long int>(size, -10000L);
""",
}

#
# Final dictionary
openmp_templates = {
    'population_header': population_header,
    'attr_decl': attribute_decl,
    'attr_acc': attribute_acc,
    'accessor_template': attribute_template,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_delayed': attribute_delayed,
    'rng': cpp_11_rng,

    'rate_psp': rate_psp,
    'spike_specific': spike_specific
}
