pyx_template = '''# cython: embedsignature=True

from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport CSR as CSR

cdef extern from "ANNarchy.h":

    # Data structures
%(pop_struct)s
%(proj_struct)s


    # Instances
%(pop_ptr)s
%(proj_ptr)s

    # Methods
    void initialize(double)
    void run(int nbSteps)
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

# Initialize the network
def pyx_create(double dt):
    initialize(dt)

# Simulation for the given numer of steps
def pyx_run(int nb_steps):
    run(nb_steps)

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