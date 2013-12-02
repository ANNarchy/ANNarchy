"""

    Global.py
    
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
import os
from datetime import datetime
from math import ceil

# instances
_populations = []       # created populations
_projections = []       # created projections

# predefined variables / parameters
_pre_def_synapse = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_var = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_par = []

_pre_def_neuron = ['rank', 'rate']


# path to annarchy working directory
annarchy_dir = os.getcwd() + '/annarchy'

_compiled = False   #I know it's evil

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
    nb_steps = ceil(duration / config['dt'])
    ANNarchyCython.pyNetwork().Run(nb_steps)
    t_stop = datetime.now()
    if show_time:
        print 'Simulation:\t', t_stop - t_start, '(', nb_steps, 'steps, '+duration+' ms)'

def reset(populations=False, projections=False):
    """
    Reset the network to initial values.
    
    Parameter:
    
    * *populations*: reset the population values.
    
    * *projections*: reset the projection values.
    """
    if populations:
        for pop in _populations:
            pop.reset()
            
    if projections:
        print 'currently not implemented'
        
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

def record(to_record):
    
    for data_set in to_record:
        data_set['pop'].start_record(data_set['var'])

def get_record(to_record):
    data = {}
    
    for data_set in to_record:
        if data_set['pop'] in data:
            data[ data_set['pop'].name ].update( { data_set['var']: data_set['pop'].get_record(data_set['var']) } )
        else:
            data.update( { data_set['pop'].name: { data_set['var']: data_set['pop'].get_record(data_set['var']) } } )
    
    return data

def current_time():
    """
    Returns current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    return ANNarchyCython.pyNetwork().Time() * config['dt']

def current_step():
    """
    Returns current simulation time step.
    """
    import ANNarchyCython    
    return ANNarchyCython.pyNetwork().Time()    
