"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

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

#include "ANNarchy.hpp"
#include <random>

%(include_additional)s
%(include_profile)s
extern %(float_prec)s dt;
extern long int t;
extern std::vector<std::mt19937> rng;
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
%(access_additional)s

    // Method called to initialize the data structures
    void init_population() {
    #ifdef _TRACE_INIT
        std::cout << "  PopStruct%(id)s::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
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
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct%(id)s::update_rng()" << std::endl;
#endif
%(update_rng)s
    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct%(id)s::update_global_ops()" << std::endl;
#endif
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
    void update() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct%(id)s::update()" << std::endl;
#endif    
%(update_variables)s
    }

    void spike_gather() {
%(test_spike_cond)s
    }

    %(stop_condition)s

    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(size_in_bytes)s
        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct%(id)s::clear() - this = " << this << std::endl;
#endif
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
        _delayed_%(name)s.push_front(%(name)s);
        _delayed_%(name)s.pop_back();
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
        _delayed_%(name)s.push_front(%(name)s);
        _delayed_%(name)s.pop_back();
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
    'dist_decl': "auto dist_%(rd_name)s = %(rd_init)s;",
    'local': {
        'decl': "std::vector<%(type)s> %(rd_name)s ;",
        'init': "%(rd_name)s = std::vector<%(type)s>(size, 0.0);",
        'update': "%(rd_name)s[i] = dist_%(rd_name)s(rng[%(index)s]);",
        'clear': "%(rd_name)s.clear();\n%(rd_name)s.shrink_to_fit();"
    },
    'global': {
        'decl': "%(type)s %(rd_name)s;",
        'init': "%(rd_name)s = 0.0;",
        'update': "%(rd_name)s = dist_%(rd_name)s(rng[0]);",
        'clear': ""
    },
    'update': """
        if (_active) {
%(rng_dist)s
%(update_rng_global)s
            for(int i = 0; i < size; i++) {
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
    if (pop%(id)s->_active)
        std::fill(pop%(id)s->_sum_%(target)s.begin(), pop%(id)s->_sum_%(target)s.end(), static_cast<%(float_prec)s>(0.0) );
"""
}

spike_specific = {
    'spike': {
        'declare': """
    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;
""",
        'init': """
        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);
""",
        'reset': """
        // Spiking variables
        spiked.clear();
        spiked.shrink_to_fit();
        std::fill(last_spike.begin(), last_spike.end(), -10000L);
""",
        'clear': """
// Spike events
spiked.clear();
spiked.shrink_to_fit();

last_spike.clear();
last_spike.shrink_to_fit();
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
        std::fill(refractory_remaining.begin(), refractory_remaining.end(), 0);
"""
    }
}

#
# Final dictionary
single_thread_templates = {
    'population_header': population_header,
    'attr_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_delayed': attribute_delayed,
    'rng': cpp_11_rng,

    'rate_psp': rate_psp,
    'spike_specific': spike_specific
}
