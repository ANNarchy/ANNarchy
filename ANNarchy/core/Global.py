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
from __future__ import print_function

import sys, os
from datetime import datetime
from math import ceil

# Global instances
_network = None         # created network
_populations = []       # created populations
_projections = []       # created projections
_functions = []         # created functions

# Predefined variables / parameters
_pre_def_synapse = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_var = ['value', 'rank', 'delay', 'psp']
_pre_def_synapse_par = []

_pre_def_neuron = ['rank', 'rate']

# Aditional instances
_cy_instance = None
_visualizer = None

# Path to the annarchy working directory
annarchy_dir = os.getcwd() + '/annarchy'

# Flag to tell if the network has already been compiled
_compiled = False   #I know it's evil

# Configuration
config = dict(
   { 
    'dt' : 1.0,
    'verbose': False,
    'show_time': False,
    'suppress_warnings': False,
    'float_prec': 'single',
    'num_threads': None, #default by os
   }
)

# Authorized keywork for attributes
authorized_keywords = [
    'init',                   
    'min',
    'max',
    'population',
    'postsynaptic',
    'explicit',
    'implicit',
    'exponential',
    'int',
    'bool',
    'float'
]

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy simulation environment. It takes various optional arguments: 

    Parameters:
    
    * *dt*: discretization constant (default: 1.0 ms).
    
    * *num_threads*: number of treads used by openMP (overrides the environment variable OMP_NUM_THREADS when set, default = None).
    
    * *float_prec*: determines the floating point precision to be used ('single' or 'double'). By default ANNarchy uses single floating point precision. 
    
    The following parameters are mainly for debugging and profiling, and should be ignored by most users:
    
    * *verbose*: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    
    * *suppress_warnings*: if True, warnings (e. g. from mathematical parser) are suppressed.
    
    * *show_time*: if True, initialization times are shown. ATTENTION: verbose should be set to True additionally.
    
    
    **Note**: this function should be used before any other functions of ANNarchy, right after ``from ANNarchy import *``.
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]
        else:
            _print('unknown key:', key)
    
def reset(states=True, connections=False):
    """
    Reinitialises the network, runs each object's reset() method (resetting them to 0).

    Parameter:

    * *states*: if set to True then it will reinitialise the neuron state variables.
    * *connections*: if set to True then it will reinitialise the connection variables.
    """
    if states:
        for pop in _populations:
            pop.reset()
            
    if connections:
        print('currently not implemented')
        
def get_population(name):
    """
    Returns population corresponding to *name*.
    
    Parameter:
    
    * *name*: population name

    Returns:
    
    the requested population if existing otherwise None is returned.
    """
    for pop in _populations:
        if pop.name == name:
            return pop
        
    print("Error: no population",name,"found.")
    return None

def get_projection(pre, post, target, suppress_error=False):
    """
    Returns projection corresponding to the arguments.
    
    Parameters:
    
    * *pre*: presynaptic population
    * *post*: postsynaptic population
    * *target*: connection type
    * *suppress_error*: if suppress_error is True the potential error will not prompted.
    
    Returns:
    
    the requested projection if existing otherwise None is returned.
    """
    for proj in _projections:
        
        if proj.post == post:
            if proj.pre == pre:
                if proj.target == target:
                    return proj
    
    if not suppress_error:
        _error("No projection " + pre.name + " -> " + post.name + " with target " + target + " found.")
    
    return None
    
def add_function(function):
    """
    Defines a global function which can be used by all neurons and synapses.
    
    The function must have only one return value and use only the passed arguments.
    
    Examples of valid functions:
    
        logistic(x) = 1 / (1 + exp(-x))
        
        piecewise(x, a, b) =    if x < a:
                                    a
                                else:
                                    if x > b :
                                        b
                                    else:
                                        x
    
    Please refer to the manual to know the allowed mathematical functions.
    """  
    _functions.append(function)
    
def simulate(duration):
    """

    Runs the network for the given duration. 
    

    If an integer is given, the argument represents the number of time steps.
    
    If a floating point value is given, it represents a duration in milliseconds computed relative to the discretization step declared in setup() (default: 1ms). 

    """
    if isinstance(duration, int):
        nb_steps = duration
    elif isinstance(duration, float):
        nb_steps = ceil(duration / config['dt'])
    else:
        _error('simulate() require either integers or floats.')
        return

    if _network:      
        _network.run(nb_steps)
    else:
        _error('simulate(): the network is not compiled yet.')
        return
    
def step():
    """

    Performs a single simulation step. 

    """
    if _network:      
        _network.run(1)

    
def current_time():
    """
    Returns current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    try:
        return _network.get_time() * config['dt']
    except:
        return 0.0

def current_step():
    """
    Returns current simulation step.
    """
    try:
        return _network.get_time()
    except:
        return 0

def set_current_time(time):
    """
    Set current simulation time in ms.
    
    **Note**: computed as number of simulation steps times dt
    """
    try:
        _network.set_time(int( time / config['dt']))
    except:
        _warning('the network is not compiled yet')
    
def set_current_step(time):
    """
    set current simulation step.
    """
    try:
        _network.set_time( time )
    except:
        _warning('the network is not compiled yet')
        
def record(to_record):
    """
    Record variables of one or more populations. For more detailed information please refer to the Population.record method.
    
    Parameter:
    
    * *to_record*: a set of dictionaries containing population objects and variable names. Optionally you may append an as_1D key to get a different format.
    
    Example:
    
        .. code-block:: python
        
            to_record = [
                { 'pop': Input, 'var': 'rate' }, 
                { 'pop': Input, 'var': 'mp', 'as_1D': True }        
            ]
            
        By default the variable data is always handled in population geometry shape. In this example, we force the storage of ``mp`` as one dimensional array.

    """
    for data_set in to_record:
        data_set['pop'].start_record(data_set['var'])

def get_record(to_record):
    """
    Retrieve recorded variables of one or more populations. For more detailed information please refer to the Population.get_record method.
  
    Parameter:
    
    * *to_record*: a set of dictionaries containing population objects and variable names. Optionally you may append an as_1D key to get a different format. For more details check Population.record().
    
    Returns:
    
    * A dictionary containing all recorded values. The dictionary is empty if no recorded data is available.
    
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

def _print(*var_text):
    """
    Prints a message to standard out.
    """    
    text = ''
    for var in var_text:
        text += str(var) + ' '
        
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)

def _debug(*var_text):
    """
    Prints a message to standard out, if verbose mode set True.
    """    
    if not config['verbose']:
        return
    
    text = ''
    for var in var_text:
        text += str(var) + ' '
        
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)
        
def _warning(*var_text):
    """
    Prints a warning message to standard out.
    """
    text = 'WARNING: '
    for var in var_text:
        text += str(var) + ' '

    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)
        
def _error(*var_text):
    """
    Prints an error message to standard out.
    """
    text = 'ERROR: '
    for var in var_text:
        text += str(var) + ' '
    
    if sys.version_info[:2] >= (2, 6) and sys.version_info[:2] < (3, 0):
        p = print        
        p(text)
    else:
        print(text)
