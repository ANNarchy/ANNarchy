import os
from datetime import datetime

_populations = []       # created population instances
_projections = []       # created projection instances
_proj_class_defs = []   # user defined projection classes

# predefined variables / parameters
_pre_def_synapse = ['dt', 'tau', 'value', 'rank', 'delay']
_pre_def_neuron = ['dt', 'tau', 'rank', 'rate']

# path to annarchy working directory
annarchy_dir = os.getcwd() + '/annarchy'

# version
version = 'ANNarchy-4.0.0'

# discretization timestamp
config = dict(
       { 'dt' : 1.0 }
)

def setup(**keyValueArgs):
    """
    takes various optional arguments: 

    * *dt*:    discretization constant
    
    **Note**: use this function before any other functions of ANNarchy
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]

def simulate(duration, show_time=False):
    """
    simulate #duration steps
    """    
    import ANNarchyCython
    t_start = datetime.now()
    ANNarchyCython.pyNetwork().Run(duration)
    t_stop = datetime.now()
    if show_time:
        print 'Simulation:\t', t_stop - t_start, '(', duration, 'steps)'