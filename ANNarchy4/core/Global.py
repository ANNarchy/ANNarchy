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
import sys, os
from datetime import datetime
from math import ceil
import __future__

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
   { 
    'dt' : 1.0,
    'verbose': False,
    'show_time': False,
    'suppress_warnings': False,
    'float_prec': 'single'
   }
)

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy4 simulation environment. It takes various optional arguments: 

    Parameter:
    
    * *dt*:         discretization constant
    * *verbose*:    shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    * *suppress_warnings*:  if set True warnings suppressed.
    * *show_time*:  if set True, initialization times are shown. ATTENTION: verbose should be set to True additionally.
    * *float_prec*: determine the used floating point precision. By default ANNarchy4 uses single floating point precision for computation. 
    
    **Note**: use this function before any other functions of ANNarchy4.
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]

def simulate(duration):
    """
    Run the simulation.
    
    Parameter:
        
    * *duration*: number of time steps simulated in ANNarchy ( 1 time steps is normally equal to 1 ms )
    """    
    import ANNarchyCython

    nb_steps = ceil(duration / config['dt'])
    ANNarchyCython.pyNetwork().Run(nb_steps)

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
        print('currently not implemented')
        
def get_population(name):
    """
    Returns population corresponding to *name*.
    
    Parameter:
    
    * *name*: population name
    """
    for pop in _populations:
        if pop.name == name:
            return pop
        
    print("Error: no population",name,"found.")
    return None

def get_projection(pre, post, target):
    """
    Returns projection corresponding to the arguments.
    
    Parameters:
    
    * *pre*: presynaptic population
    * *post*: postsynaptic population
    * *target*: connection type
    """
    for proj in _projections:
        
        if proj.post == post:
            if proj.pre == pre:
                if proj.target == target:
                    return proj
    
    print("Error: no projection",pre.name,"->",post.name,"with target ",target,"found.")
    return None

def record(to_record):
    """
    Record variables of one or more populations.
    
    Parameter:
    
    * *to_record*: a set of dictionaries containing population objects and variable names. Optionally you may append an as_1D key to get a different format. For more details check Population.get_record().
    
    Example:
    
        .. code-block:: python
        
            to_record = [
                { 'pop': Input, 'var': 'rate' }, 
                { 'pop': Input, 'var': 'mp' }        
            ]

    """
    for data_set in to_record:
        data_set['pop'].start_record(data_set['var'])

def get_record(to_record):
    """
    Retrieve recorded variables of one or more populations.
    
    Returns:
    
    * ...
    
    Example:
    
        .. code-block:: python
        
            ...
        
    """    
    data = {}
    
    for data_set in to_record:
        if data_set['pop'].name in data:
            if 'as_1D' in data_set.keys():
                data[ data_set['pop'].name ].update( { data_set['var']: data_set['pop'].get_record(data_set['var'], data_set['as_1D']) } )
            else:
                data[ data_set['pop'].name ].update( { data_set['var']: data_set['pop'].get_record(data_set['var']) } )
        else:
            if 'as_1D' in data_set.keys():
                data.update( { data_set['pop'].name: { data_set['var']: data_set['pop'].get_record(data_set['var'], data_set['as_1D']) } } )
            else:
                data.update( { data_set['pop'].name: { data_set['var']: data_set['pop'].get_record(data_set['var']) } } )
    
    return data

def current_time():
    """
    Returns current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    return ANNarchyCython.pyNetwork().get_time() * config['dt']

def current_step():
    """
    Returns current simulation step.
    """
    import ANNarchyCython    
    return ANNarchyCython.pyNetwork().get_time()    

def set_current_time(time):
    """
    Set current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    import ANNarchyCython
    ANNarchyCython.pyNetwork().set_time(int( time / config['dt']))

def set_current_step(time):
    """
    set current simulation step.
    """
    import ANNarchyCython    
    ANNarchyCython.pyNetwork().set_time( time )    

def _print(text):
    """
    Prints a message to standard out.
    """    
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        __future__.print_function(text)
    else:
        print(text)
        
def _warning(text):
    """
    Prints a warning message to standard out.
    """
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        __future__.print_function("WARNING",text)
    else:
        print(text)
        
def _error(text):
    """
    Prints an error message to standard out.
    """
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        __future__.print_function("ERROR:",text)
    else:
        print("ERROR",text)