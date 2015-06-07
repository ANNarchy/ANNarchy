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
header_struct = {
    'rate' :
"""#pragma once
#include<random>

extern double dt;
extern long int t;
extern std::mt19937 rng;

struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

%(additional)s

    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    bool set_active(bool val) { _active = val; }

    // Neuron specific
%(accessor)s

    void init_population() {
%(init)s
    }

    void update_rng() {
        if (_active){
            for(int i = 0; i < size; i++) {
%(update_rng)s
            }
        }
    }

    void update() {
%(update)s
    }
};
""",
    'spike':
"""#pragma once
#include<random>

extern double dt;
extern long int t;
extern std::mt19937 rng;

struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

%(additional)s

    // Spiking events
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;

    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    bool set_active(bool val) { _active = val; }

    // Neuron specific
%(accessor)s

    void init_population() {
%(init)s
    }

    void update() {
%(update)s
    }
};
"""
}

# c like definition of neuron attributes, whereas 'local' is used if values can vary across
# neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
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
# across neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
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

# Definition for the usage of C++11 STL template random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
cpp_11_rng = {
    'init': """
        %(rd_name)s = std::vector<double>(size, 0.0);
        dist_%(rd_name)s = %(rd_init)s;
""",
    'update': """
            pop%(id)s.%(rd_name)s[i] = pop%(id)s.dist_%(rd_name)s(rng);
"""
}
# Rate respectively spiking models require partly special variables or have different operations.
# This dictionary contain these unique initializations.
#
# Parameters (may differ):
#
#     id: id of the population
#     target: target name (e. g. FF, LAT ...)
model_specific_init = {
    'spike_event':
"""
        // Spiking event and refractory
        refractory = std::vector<int>(size, 0);
        spiked = std::vector<int>(0, 0);
        last_spike = std::vector<long int>(size, -10000L);
        refractory_remaining = std::vector<int>(size, 0);
""",
    'rate_psp':
"""
        // Post-synaptic potential
        _sum_%(target)s = std::vector<double>(size, 0.0);"""
}