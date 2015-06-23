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
'''

proj_pyx_wrapper = """
cdef class proj%(id_proj)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()

        proj%(id_proj)s.set_size( size )
%(init_connectivity)s
        proj%(id_proj)s.set_w(syn.w)
%(init_delay)s
%(init_event_driven)s

    property size:
        def __get__(self):
            return proj%(id_proj)s.get_size()

    def nb_synapses(self, int n):
        return proj%(id_proj)s.nb_synapses(n)

    def _set_learning(self, bool l):
        proj%(id_proj)s._learning = l

%(accessor_declaration_connectivity)s
%(accessor_declaration_delay)s
%(accessor_declaration_parameters_variables)s
%(accessor_declaration_structural_plasticity)s
%(accessor_declaration_additional)s

"""

proj_pyx_struct = """
    cdef struct ProjStruct%(id_proj)s :
        bool _learning

        int get_size()
        int nb_synapses(int)
        void set_size(int)

%(accessor_wrapper_connectivity)s
%(accessor_wrapper_delay)s
%(accessor_wrapper_event_driven)s
%(accessor_wrapper_export)s
%(accessor_wrapper_structural_plasticity)s
%(accessor_wrapper_additional)s
"""