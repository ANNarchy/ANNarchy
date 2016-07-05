pyx_template = '''# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
import numpy as np
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport CSR as CSR

cdef extern from "ANNarchy.h":

    # Data structures
%(pop_struct)s
%(proj_struct)s

    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period
        long offset

    void addRecorder(Monitor*)
    void removeRecorder(Monitor*)
%(monitor_struct)s

    # Instances
%(pop_ptr)s
%(proj_ptr)s

    # Methods
    void initialize(double, long)
    void setSeed(long)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)

    # Number of threads
    void setNumberThreads(int)


# Population wrappers
%(pop_class)s

# Projection wrappers
%(proj_class)s

# Monitor wrappers
cdef class Monitor_wrapper:
    cdef Monitor *thisptr
    def __cinit__(self, list ranks, int period, long offset):
        pass
    property ranks:
        def __get__(self): return self.thisptr.ranks
        def __set__(self, val): self.thisptr.ranks = val
    property period:
        def __get__(self): return self.thisptr.period
        def __set__(self, val): self.thisptr.period = val
    property offset:
        def __get__(self): return self.thisptr.offset
        def __set__(self, val): self.thisptr.offset = val

def add_recorder(Monitor_wrapper recorder):
    addRecorder(recorder.thisptr)
def remove_recorder(Monitor_wrapper recorder):
    removeRecorder(recorder.thisptr)

%(monitor_wrapper)s

# Initialize the network
def pyx_create(double dt, long seed):
    initialize(dt, seed)

# Simulation for the given number of steps
def pyx_run(int nb_steps):
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
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()

# Set number of threads
def set_number_threads(int n):
    setNumberThreads(n)

# Set seed
def set_seed(long seed):
    setSeed(seed)
'''

# Export for populations
pop_pyx_struct = """
    # Export Population %(id)s (%(name)s)
    cdef struct PopStruct%(id)s :
        int get_size()
        bool is_active()
        void set_active(bool)
        void reset()
%(export_refractory)s
%(export_parameters_variables)s
%(export_targets)s
%(export_additional)s
"""

# Wrapper for populations
pop_pyx_wrapper = """
# Wrapper for population %(id)s (%(name)s)
cdef class pop%(id)s_wrapper :

    def __cinit__(self, %(wrapper_args)s):
%(wrapper_init)s

    property size:
        def __get__(self):
            return pop%(id)s.get_size()
    def reset(self):
        pop%(id)s.reset()
    def activate(self, bool val):
        pop%(id)s.set_active(val)

%(wrapper_access_parameters_variables)s
%(wrapper_access_targets)s
%(wrapper_access_refractory)s
%(wrapper_access_additional)s

"""

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
%(export_structural_plasticity)s
%(export_additional)s
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
%(wrapper_access_structural_plasticity)s
%(wrapper_access_additional)s

"""
