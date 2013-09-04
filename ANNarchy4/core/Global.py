import os
from datetime import datetime

# instances
_populations = []       # created populations
_projections = []       # created projections

# predefined variables / parameters
_pre_def_synapse = ['dt', 'tau', 'value', 'rank', 'delay', 'psp']
_pre_def_synapse_var = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_par = ['dt', 'tau']

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

    Parameter:
    
        * *dt*:    discretization constant
        
    **Note**: use this function before any other functions of ANNarchy
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]

def simulate(duration, show_time=False):
    """
    Run the simulation.
    
    Parameter:
        
        * *duration*: number of time steps simulated in ANNarchy ( 1 time steps is normally equal to 1 ms )
        * *show_time*: how long the simulation took (cpu-time). Might be used for an assumption of whole computation time.
    """    
    import ANNarchyCython
    t_start = datetime.now()
    ANNarchyCython.pyNetwork().Run(duration)
    t_stop = datetime.now()
    if show_time:
        print 'Simulation:\t', t_stop - t_start, '(', duration, 'steps)'

def get_population(name):
    """
    Returns population corresponding to *name*.
    
    Parameter:
    
        * *name*: population name
    """
    for pop in _populations:
        if pop.name == name:
            return pop
        
    print "Error: no population with the name '"+name+"' found."
    return None

def get_projection(pre, post, target):
    """
    Returns projection corresponding to the arguments.
    
    Parameter:
    
        * *pre*: presynaptic population
        * *post*: postsynaptic population
        * *target*: connection type
    """
    for proj in _projections:
        
        if proj.post == post:
            if proj.pre == pre:
                if proj.target == target:
                    return proj
    
    print "Error: no projection '"+pre.name+"'->'"+post.name+"' with target '"+target+"' found."
    return None
