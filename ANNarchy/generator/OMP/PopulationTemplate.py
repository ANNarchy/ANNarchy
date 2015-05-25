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

extern double dt;
extern long int t;

struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

%(additional)s

    // Record parameter
    int record_period;
    long int record_offset;
    std::vector<int> record_ranks;

    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    bool set_active(bool val) { _active = val; }

    // Record
    void set_record_period(int period, long int t) { record_period = period; record_offset = t; }
    void set_record_ranks( std::vector<int> ranks) { record_ranks = ranks; }

    // Neuron specific
%(accessor)s

    void init_population() {
%(init)s
    }
    
    void update() {
%(update)s
    }
};
""",
    'spike':
"""#pragma once

extern double dt;
extern long int t;

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

    // Record parameter
    int record_period;
    long int record_offset;
    std::vector<int> record_ranks;

    // Access functions used by cython wrapper
    int get_size() { return size; }
    bool is_active() { return _active; }
    bool set_active(bool val) { _active = val; }

    // Record
    void set_record_period(int period, long int t) { record_period = period; record_offset = t; }
    void set_record_ranks( std::vector<int> ranks) { record_ranks = ranks; }

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

# c like definition of parameter members, whereas 'local' is used if values can vary across neurons,
# consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
parameter_decl = {
    'local':
"""
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'global':
"""
    // Global parameter %(name)s
    %(type)s  %(name)s ;
"""    
}

# c like definition of accessors for parameter members, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
parameter_acc = {
    'local':
"""
    // Local parameter %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; }
""",
    'global':
"""
    // Global parameter %(name)s
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
parameter_cpp_export = {
    'local':
"""
        # Local parameter %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_single_%(name)s(int rk)
        void set_%(name)s(vector[%(type)s])
        void set_single_%(name)s(int, %(type)s)
""",
    'global':
"""
        # Global parameter %(name)s
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
parameter_pyx_wrapper = {
    'local':
"""
    # Local parameter %(name)s
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
    # Global parameter %(name)s
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
parameter_cpp_init = {
    'local':
"""
        // Local parameter %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
""",
    'global':
"""
        // Global parameter %(name)s
        %(name)s = %(init)s;
"""
}

# c like definition of accessors for variable members, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons. As variables
# change over time, values can be recorded.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
variable_decl = {
    'local':
"""
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""",
    'global':
"""
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
"""
}

# c like definition of accessors for variable members, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons. As variables
# change over time, values can be recorded. By default recording is disabled, so we need functions
# to start, stop and get recorded values.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
variable_acc = {
    'local':
"""
    // Local variable %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; }
    std::vector< std::vector< %(type)s > > get_recorded_%(name)s() { return recorded_%(name)s; }
    bool is_%(name)s_recorded() { return record_%(name)s; }
    void set_record_%(name)s(bool val) { record_%(name)s = val; }
    void clear_recorded_%(name)s() { recorded_%(name)s.clear(); }
""",
    'global':
"""
    // Global variable %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s val) { %(name)s = val; }
    std::vector<%(type)s> get_recorded_%(name)s() { return recorded_%(name)s; }
    bool is_%(name)s_recorded() { return record_%(name)s; }
    void set_record_%(name)s(bool val) { record_%(name)s = val; }
    void clear_recorded_%(name)s() { recorded_%(name)s.clear(); }
"""
}

# export of accessors for variable members towards python, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
variable_cpp_export = {
    'local':
"""
        # Local variable %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_single_%(name)s(int rk)
        void set_%(name)s(vector[%(type)s])
        void set_single_%(name)s(int, %(type)s)
        vector[vector[%(type)s]] get_recorded_%(name)s()
        void set_record_%(name)s(bool)
        bool is_%(name)s_recorded()
        void clear_recorded_%(name)s()
""",
    'global':
"""
        # Global variable %(name)s
        %(type)s  get_%(name)s()
        void set_%(name)s(%(type)s)
        vector[%(type)s] get_recorded_%(name)s()
        void set_record_%(name)s(bool)
        bool is_%(name)s_recorded()
        void clear_recorded_%(name)s()
"""
}

# export of accessors for variable members towards python, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons. Functions marked as cpdef
# can be accessed from python as well as cython. Local parameters allows access to single as well as all values.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
variable_pyx_wrapper = {
    'local':
"""
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.get_%(name)s())
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.set_%(name)s(value)
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.get_single_%(name)s(rank)
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.set_single_%(name)s(rank, value)
    def start_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(True)
    def stop_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(False)
    def get_record_%(name)s(self):
        cdef vector[vector[%(type)s]] tmp = pop%(id)s.get_recorded_%(name)s()
        pop%(id)s.clear_recorded_%(name)s()
        return tmp
""",
    'global':
"""
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.get_%(name)s()
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.set_%(name)s(value)
    def start_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(True)
    def stop_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(False)
    def get_record_%(name)s(self):
        cdef vector[%(type)s] tmp = pop%(id)s.get_recorded_%(name)s()
        pop%(id)s.clear_recorded_%(name)s()
        return tmp
"""
}

# Initialization of parameters due to the init_population method.
#
# Parameters:
#
#    name: name of the variable
#    init: initial value
variable_cpp_init = {
    'local':
"""
        // Local variable %(name)s
        %(name)s = std::vector<%(type)s>(size, %(init)s);
        recorded_%(name)s = std::vector<std::vector<%(type)s> >(0, std::vector<%(type)s>(0,%(init)s));
        record_%(name)s = false;
""",
    'global':
"""
        // Global variable %(name)s
        %(name)s = %(init)s;
        recorded_%(name)s = std::vector<%(type)s>(0, %(init)s);
        record_%(name)s = false;
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
        record_spike = false;
        recorded_spike = std::vector<std::vector<long int> >(size, std::vector<long int>());
        spiked = std::vector<int>(0, 0);
        last_spike = std::vector<long int>(size, -10000L);
        refractory_remaining = std::vector<int>(size, 0);
""",
    'rate_psp':
"""
        // Post-synaptic potential
        _sum_%(target)s = std::vector<double>(size, 0.0);"""
}