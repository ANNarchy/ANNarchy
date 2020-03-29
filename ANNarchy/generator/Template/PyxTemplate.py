pyx_template = '''# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from math import ceil
import numpy as np
import sys
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL
from ANNarchy.core.cython_ext.Connector cimport CSRConnectivity, CSRConnectivityPre1st

cdef extern from "ANNarchy.h":

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

    void addRecorder(Monitor*)
    void removeRecorder(Monitor*)
%(monitor_struct)s

    # Instances
%(pop_ptr)s
%(proj_ptr)s

    # Methods
    void initialize(%(float_prec)s, long)
    void init_rng_dist()
    void setSeed(long)
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

# Population wrappers
%(pop_class)s

# Projection wrappers
%(proj_class)s

# Monitor wrappers
cdef class Monitor_wrapper:
    cdef Monitor *thisptr
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        pass
    property ranks:
        def __get__(self): return self.thisptr.ranks
        def __set__(self, val): self.thisptr.ranks = val
    property period:
        def __get__(self): return self.thisptr.period_
        def __set__(self, val): self.thisptr.period_ = val
    property offset:
        def __get__(self): return self.thisptr.offset_
        def __set__(self, val): self.thisptr.offset_ = val
    property period_offset:
        def __get__(self): return self.thisptr.period_offset_
        def __set__(self, val): self.thisptr.period_offset_ = val

def add_recorder(Monitor_wrapper recorder):
    addRecorder(recorder.thisptr)
def remove_recorder(Monitor_wrapper recorder):
    removeRecorder(recorder.thisptr)

%(monitor_wrapper)s

# User-defined functions
%(functions_wrapper)s

# User-defined constants
%(constants_wrapper)s

# Initialize the network
def pyx_create(%(float_prec)s dt, long seed):
    initialize(dt, seed)

def pyx_init_rng_dist():
    init_rng_dist()

# Simple progressbar on the command line
def progress(count, total, status=''):
    """
    Prints a progress bar on the command line.

    adapted from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    Modification: The original code set the '\\r' at the end, so the bar disappears when finished.
    I moved it to the front, so the last status remains.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\\r[%%s] %%s%%s ...%%s' %% (bar, percents, '%%', status))
    sys.stdout.flush()

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
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if nb > 1 and progress_bar:
                progress(i+1, nb, 'simulate()')
        if rest > 0:
            run(rest)

        if (progress_bar):
            print('\\n')

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
def set_seed(long seed):
    setSeed(seed)
'''

pyx_device_specific={
    'openmp': {
        'wrapper': """
# Set number of threads
def set_number_threads(int n):
    setNumberThreads(n)
""",
        'export': """
    # Number of threads
    void setNumberThreads(int)
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
# export of accessors for parameter members towards python, whereas 'local' is used if values can vary
# across neurons, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (float, double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
pop_attribute_cpp_export = {
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
pop_attribute_pyx_wrapper = {
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
%(export_targets)s
%(export_mean_fr)s
%(export_additional)s

        # memory management
        long int size_in_bytes()
        void clear()
"""

# Wrapper for populations
pop_pyx_wrapper = """
# Wrapper for population %(id)s (%(name)s)
cdef class pop%(id)s_wrapper :

    def __cinit__(self, %(wrapper_args)s):
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
%(wrapper_access_targets)s
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

# export of accessors for synaptic attributes towards python, whereas 'local' is used if values can vary
# across synapses within a dendrite, consequently 'global' is used if values are common to all synapses within
# a single dendrite.
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
        vector[vector[%(type)s]] get_%(name)s()
        vector[%(type)s] get_dendrite_%(name)s(int)
        %(type)s get_synapse_%(name)s(int, int)
        void set_%(name)s(vector[vector[%(type)s]])
        void set_dendrite_%(name)s(int, vector[%(type)s])
        void set_synapse_%(name)s(int, int, %(type)s)
""",
    'semiglobal':
"""
        # Semiglobal %(attr_type)s %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_dendrite_%(name)s(int)
        void set_%(name)s(vector[%(type)s])
        void set_dendrite_%(name)s(int, %(type)s)
""",
    'global':
"""
        # Global %(attr_type)s %(name)s
        %(type)s get_%(name)s()
        void set_%(name)s(%(type)s)
"""
}

attribute_pyx_wrapper = {
    'local':
"""
    # Local %(attr_type)s %(name)s
    def get_%(name)s(self):
        return proj%(id)s.get_%(name)s()
    def set_%(name)s(self, value):
        proj%(id)s.set_%(name)s( value )
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.get_dendrite_%(name)s(rank)
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.set_dendrite_%(name)s(rank, value)
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.get_synapse_%(name)s(rank_post, rank_pre)
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.set_synapse_%(name)s(rank_post, rank_pre, value)
""",
    'semiglobal':
"""
    # Semiglobal %(attr_type)s %(name)s
    def get_%(name)s(self):
        return proj%(id)s.get_%(name)s()
    def set_%(name)s(self, value):
        proj%(id)s.set_%(name)s(value)
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.get_dendrite_%(name)s(rank)
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.set_dendrite_%(name)s(rank, value)
""",
    'global':
"""
    # Global %(attr_type)s %(name)s
    def get_%(name)s(self):
        return proj%(id)s.get_%(name)s()
    def set_%(name)s(self, value):
        proj%(id)s.set_%(name)s(value)
"""
}

# Export for projections
proj_pyx_struct = """
    # Export Projection %(id_proj)s
    cdef struct ProjStruct%(id_proj)s :
        # Flags
        bool _transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset
        # Size
        int get_size()
        int nb_synapses(int)
        void set_size(int)

%(export_connectivity)s
%(export_delay)s
%(export_event_driven)s
%(export_parameters_variables)s
%(export_functions)s
%(export_structural_plasticity)s
%(export_additional)s

        # memory management
        long int size_in_bytes()
        void clear()
"""

# Wrapper for projections
proj_pyx_wrapper = """
# Wrapper for projection %(id_proj)s
cdef class proj%(id_proj)s_wrapper :

    def __cinit__(self, %(wrapper_args)s):
%(wrapper_init_connectivity)s
%(wrapper_init_delay)s
%(wrapper_init_event_driven)s

    property size:
        def __get__(self):
            return proj%(id_proj)s.get_size()

    def nb_synapses(self, int n):
        return proj%(id_proj)s.nb_synapses(n)

    # Transmission flag
    def _get_transmission(self):
        return proj%(id_proj)s._transmission
    def _set_transmission(self, bool l):
        proj%(id_proj)s._transmission = l

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

%(wrapper_access_connectivity)s
%(wrapper_access_delay)s
%(wrapper_access_parameters_variables)s
%(wrapper_access_functions)s
%(wrapper_access_structural_plasticity)s
%(wrapper_access_additional)s

    # memory management
    def size_in_bytes(self):
        return proj%(id_proj)s.size_in_bytes()

    def clear(self):
        return proj%(id_proj)s.clear()
"""
