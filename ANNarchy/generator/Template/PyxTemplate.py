"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

pyx_template = '''# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from libcpp.string cimport string
from math import ceil
import numpy as np
import sys
from tqdm import tqdm
cimport numpy as np
cimport cython

# Short names for unsigned integer types
ctypedef unsigned char _ann_uint8
ctypedef unsigned short _ann_uint16
ctypedef unsigned int _ann_uint32
ctypedef unsigned long _ann_uint64

import ANNarchy
from ANNarchy.cython_ext.Connector cimport LILConnectivity as LIL

cdef extern from "ANNarchy.hpp":

    # User-defined functions
%(custom_functions_export)s

    # User-defined constants
%(custom_constants_export)s

    # Data structures
%(pop_struct)s
%(proj_struct)s


    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_

%(monitor_struct)s

    # Instances
%(pop_ptr)s
%(proj_ptr)s

    # Methods
    void create_cpp_instances()
    void initialize(%(float_prec)s)
    void destroy_cpp_instances()
    void setSeed(long, int, bool)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    %(float_prec)s getDt()
    void setDt(%(float_prec)s dt_)

%(device_specific_export)s

# Profiling (if needed)
%(prof_class)s

# Population wrappers
%(pop_class)s

# Projection wrappers
%(proj_class)s

# Monitor wrappers
%(monitor_wrapper)s

# User-defined functions
%(functions_wrapper)s

# User-defined constants
%(constants_wrapper)s

# Initialize/Destroy the network
def pyx_create():
    create_cpp_instances()
def pyx_initialize(%(float_prec)s dt):
    initialize(dt)
def pyx_destroy():
    destroy_cpp_instances()

# Simulation for the given number of steps
def pyx_run(int nb_steps, progress_bar):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps %% batch

        if progress_bar:
            pbar = tqdm(total=nb_steps*getDt())
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if progress_bar:
                pbar.update(batch*getDt())
        if progress_bar:
            pbar.close()

        if rest > 0:
            run(rest)

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(%(float_prec)s dt):
    setDt(dt)
def get_dt():
    return getDt()

%(device_specific_wrapper)s

# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
'''

pyx_device_specific={
    'openmp': {
        'wrapper': """
# Set number of threads
def set_number_threads(int n, core_list):
    setNumberThreads(n, core_list)
""",
        'export': """
    # Number of threads
    void setNumberThreads(int, vector[int])
"""
    },
    'cuda': {
        'wrapper': """
# Set GPU device
def set_device(int device_id):
    setDevice(device_id)
""",
        'export': """
    # GPU device
    void setDevice(int)
"""
    }
}

pyx_default_pop_attribute_export = {
    'local': """
        # Local attributes
        vector[%(ctype)s] get_local_attribute_all_%(ctype_name)s(string)
        %(ctype)s get_local_attribute_%(ctype_name)s(string, int)
        void set_local_attribute_all_%(ctype_name)s(string, vector[%(ctype)s])
        void set_local_attribute_%(ctype_name)s(string, int, %(ctype)s)
""",
    'global': """
        # Global attributes
        %(ctype)s get_global_attribute_%(ctype_name)s(string)
        void set_global_attribute_%(ctype_name)s(string, %(ctype)s)
"""
}

pyx_default_pop_attribute_wrapper = {
    'local': """
    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')
%(get_local_all)s

    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')
%(get_local)s

    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_local_all)s

    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_local)s
""",
    'global': """
    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')
%(get_global)s

    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_global)s
"""
}

# Export for populations
pop_pyx_struct = """
    # Export Population %(id)s (%(name)s)
    cdef struct PopStruct%(id)s :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()
%(export_refractory)s
%(export_parameters_variables)s
%(export_functions)s
%(export_mean_fr)s
%(export_additional)s

        # memory management
        long int size_in_bytes()
        void clear()
"""

# Wrapper for populations
pop_pyx_wrapper = """
# Wrapper for population %(id)s (%(name)s)
@cython.auto_pickle(True)
cdef class pop%(id)s_wrapper :

    def __init__(self, %(wrapper_args)s):
%(wrapper_init)s
    # Number of neurons
    property size:
        def __get__(self):
            return pop%(id)s.get_size()
    # Reset the population
    def reset(self):
        pop%(id)s.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop%(id)s.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop%(id)s.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop%(id)s.set_active(val)

%(wrapper_access_parameters_variables)s
%(wrapper_access_functions)s
%(wrapper_access_refractory)s
%(wrapper_access_mean_fr)s
%(wrapper_access_additional)s

    # memory management
    def size_in_bytes(self):
        return pop%(id)s.size_in_bytes()

    def clear(self):
        return pop%(id)s.clear()
"""

# Export for projections
proj_pyx_struct = """
    # Export Projection %(id_proj)s
    cdef struct ProjStruct%(id_proj)s :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
%(export_connectivity)s

%(export_delay)s
%(export_event_driven)s
%(export_parameters_variables)s
%(export_functions)s
%(export_structural_plasticity)s
%(export_additional)s

        # cuda configuration
%(export_cuda_launch_config)s

        # memory management
        long int size_in_bytes()
        void clear()
"""

# Wrapper for projections
proj_pyx_wrapper = """
# Wrapper for projection %(id_proj)s
@cython.auto_pickle(True)
cdef class proj%(id_proj)s_wrapper :

    def __init__(self, %(wrapper_args)s):
        %(wrapper_init)s

%(wrapper_connector_call)s

    # Transmission flag
    def _get_transmission(self):
        return proj%(id_proj)s._transmission
    def _set_transmission(self, bool l):
        proj%(id_proj)s._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj%(id_proj)s._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj%(id_proj)s._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj%(id_proj)s._update
    def _set_update(self, bool l):
        proj%(id_proj)s._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj%(id_proj)s._plasticity
    def _set_plasticity(self, bool l):
        proj%(id_proj)s._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj%(id_proj)s._update_period
    def _set_update_period(self, int l):
        proj%(id_proj)s._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj%(id_proj)s._update_offset
    def _set_update_offset(self, long l):
        proj%(id_proj)s._update_offset = l

    # Access connectivity
%(wrapper_access_connectivity)s

%(wrapper_access_delay)s
%(wrapper_access_parameters_variables)s
%(wrapper_access_functions)s
%(wrapper_access_structural_plasticity)s
%(wrapper_access_additional)s

        # cuda configuration
%(wrapper_cuda_launch_config)s

    # memory management
    def size_in_bytes(self):
        return proj%(id_proj)s.size_in_bytes()

    def clear(self):
        return proj%(id_proj)s.clear()
"""

pyx_default_conn_export = """
        # Access connectivity
        vector[%(idx_type)s] get_post_rank()
        vector[ vector[%(idx_type)s] ] get_pre_ranks()
        vector[%(idx_type)s] get_dendrite_pre_rank(%(idx_type)s)
        %(size_type)s nb_synapses()
        %(idx_type)s nb_dendrites()
        %(idx_type)s dendrite_size(%(idx_type)s)
"""

pyx_default_conn_wrapper = """
    property size:
        def __get__(self):
            return proj%(id_proj)s.nb_dendrites()

    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_ranks()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj%(id_proj)s.nb_dendrites()
    def nb_synapses(self):
        return proj%(id_proj)s.nb_synapses()
    def dendrite_size(self, int n):
        return proj%(id_proj)s.dendrite_size(n)
"""

# The additional _%(ctype_name)s is required to resolve ambiguity for getter-methods.
pyx_proj_attribute_export = {
    'local': """
        # Local Attributes
        vector[vector[%(ctype)s]] get_local_attribute_all_%(ctype_name)s(string)
        vector[%(ctype)s] get_local_attribute_row_%(ctype_name)s(string, int)
        %(ctype)s get_local_attribute_%(ctype_name)s(string, int, int)
        void set_local_attribute_all_%(ctype_name)s(string, vector[vector[%(ctype)s]])
        void set_local_attribute_row_%(ctype_name)s(string, int, vector[%(ctype)s])
        void set_local_attribute_%(ctype_name)s(string, int, int, %(ctype)s)
""",
    'semiglobal': """
        # Semiglobal Attributes
        vector[%(ctype)s] get_semiglobal_attribute_all_%(ctype_name)s(string)
        %(ctype)s get_semiglobal_attribute_%(ctype_name)s(string, int)
        void set_semiglobal_attribute_all_%(ctype_name)s(string, vector[%(ctype)s])
        void set_semiglobal_attribute_%(ctype_name)s(string, int, %(ctype)s)
""",
    'global': """
        # Global Attributes
        %(ctype)s get_global_attribute_%(ctype_name)s(string)
        void set_global_attribute_%(ctype_name)s(string, %(ctype)s)
"""
}

pyx_proj_attribute_wrapper = {
    'local': """
    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')
%(get_local_all)s

    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')
%(get_local_row)s

    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')
%(get_local)s

    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_local_all)s

    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_local_row)s

    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_local)s
""",
    'semiglobal': """
    # Semiglobal Attributes
    def get_semiglobal_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')
%(get_semiglobal_all)s

    def get_semiglobal_attribute(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')
%(get_semiglobal)s

    def set_semiglobal_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_semiglobal_all)s

    def set_semiglobal_attribute(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_semiglobal)s
""",
    'global': """
    # Global Attributes
    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')
%(get_global)s

    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')
%(set_global)s
"""
}

pyx_profiler_template = """# Profiling
cdef extern from "Profiling.h":
    cdef cppclass Profiling:

        @staticmethod
        Profiling* get_instance()

        double get_avg_time(string, string)
        double get_std_time(string, string)

cdef class Profiling_wrapper:

    def get_timing(self, obj_name, func_name):
        cpp_string1 = obj_name.encode('utf-8')
        cpp_string2 = func_name.encode('utf-8')

        mean = (Profiling.get_instance()).get_avg_time(cpp_string1, cpp_string2)
        std = (Profiling.get_instance()).get_std_time(cpp_string1, cpp_string2)
        return mean, std
"""
