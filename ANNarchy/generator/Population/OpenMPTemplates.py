#===============================================================================
#
#     OpenMPTemplates.py
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
    void set_size(int s) { size  = s; }
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
        _delayed_%(name)s = std::deque< std::vector< %(type)s > >(%(delay)s, std::vector< %(type)s >(size, 0.0));""",
        'update': """
        _delayed_%(name)s.push_front(%(name)s);
        _delayed_%(name)s.pop_back();
""",
        'reset' : """
        for ( int i = 0; i < _delayed_%(name)s.size(); i++ ) {
            _delayed_%(name)s[i] = %(name)s;
        }
"""
    },
    'global':{
        'init': """
        _delayed_%(name)s = std::deque< %(type)s >(%(delay)s, 0.0);""",
        'update': """
        _delayed_%(name)s.push_front(%(name)s);
        _delayed_%(name)s.pop_back();
""",
        'reset' : """
        for ( int i = 0; i < _delayed_%(name)s.size(); i++ ) {
            _delayed_%(name)s[i] = %(name)s;
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
        'decl': """    std::vector<%(type)s> %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = std::vector<%(type)s>(size, 0.0);
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
                %(rd_name)s[i] = dist_%(rd_name)s(rng);
    """
    },
    'global': {
        'decl': """    %(type)s %(rd_name)s;
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

rate_psp = {
    'decl': """
    std::vector<%(float_prec)s> _sum_%(target)s;""",
    'init': """
        // Post-synaptic potential
        _sum_%(target)s = std::vector<%(float_prec)s>(size, 0.0);""",
    'reset': """
    if (pop%(id)s._active)
        memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(%(float_prec)s));
"""
}

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
""",
    'pyx_wrapper': """
    # Refractory period
    cpdef np.ndarray get_refractory(self):
        return pop%(id)s.refractory
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value
"""
}

#
# Final dictionary
openmp_templates = {
    'population_header': population_header,
    'attr_decl': attribute_decl,
    'attr_acc': attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_delayed': attribute_delayed,
    'rng': cpp_11_rng,

    'rate_psp': rate_psp,
    'spike_specific': spike_specific
}
