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
import time
from math import ceil
import numpy as np

# Dictionaries of  instances
_populations = []       # created populations
_projections = []       # created projections
_functions = []         # created functions
_neurons = []           # created neurons
_synapses = []          # created synapses

# Global Cython instance
_network = None

# Path to the annarchy working directory
annarchy_dir = os.getcwd() + '/annarchy'

# Flag to tell if the network has already been compiled
_compiled = False   # I know it's evil
def set_compiled(): # called by the generator
    global _compiled
    _compiled = True

# Configuration
config = dict(
   { 
    'dt' : 1.0,
    'verbose': False,
    'show_time': False,
    'suppress_warnings': False,
    'num_threads': None,
    'paradigm': "openmp",
    'method': "explicit",
    'seed': -1,
    'structural_plasticity': False,
    'profiling': False
   }
)

cuda_config = dict(
    {
     'device': 0
    }
)

# Minimum number of neurons to apply OMP parallel regions
OMP_MIN_NB_NEURONS = 10

# Authorized keywork for attributes
authorized_keywords = [
    # Init
    'init',      
    # Bounds             
    'min',
    'max',
    # Locality
    'population',
    'postsynaptic',
    # Numerical methods
    'explicit',
    'implicit',
    'semiimplicit',
    'exponential',
    'midpoint',
    'exact',
    # Refractory
    'unless_refractory',
    # Type
    'int',
    'bool',
    'float'
]

# Dictionary of population variables being currently recorded
_recorded_populations = {}

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy simulation environment. It takes various optional arguments: 

    *Parameters*:
    
    * **dt**: discretization time step (default: 1.0 ms).

    * **method**: default method to numerize ODEs. Default is the explicit forward Euler method ('explicit').
    
    * **num_threads**: number of treads used by openMP (overrides the environment variable ``OMP_NUM_THREADS`` when set, default = None).

    * **structural_plasticity**: allows synapses to be dynamically added/removed during the simulation (default: False).

    * **seed**: the seed (integer) to be used in the random number generators (default = -1 is equivalent to time(NULL)). 
    
    The following parameters are mainly for debugging and profiling, and should be ignored by most users:
    
    * **verbose**: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    
    * **suppress_warnings**: if True, warnings (e. g. from the mathematical parser) are suppressed.
    
    * **show_time**: if True, initialization times are shown. Attention: verbose should be set to True additionally.
    
    
    .. note::

        This function should be used before any other functions of ANNarchy, right after ``from ANNarchy import *``::

            from ANNarchy import *
            setup(dt=1.0, method='midpoint', num_threads=2)
            ...
    """
    for key in keyValueArgs:

        if key in config.keys():
            config[key] = keyValueArgs[key]
        else:
            _print('Unknown key:', key)

        if key == 'seed':
            np.random.seed(keyValueArgs[key])

def set_cuda_config(config):
    """
    setup cuda config, whereas the config is a dictionary containing the device id where to compute on (default 0) 
    and for each population and projection an amount of threads. If not specified, we assume 32 threads for 
    populations and 192 threads for projections. ATTENTION: need to be set before compilation.

    Example:

    config = { 'device': 0, Input: 64, Output: 32, Input_Output: 64 }
    set_cuda_config(config)
    compile()

    Warning:

    setting this config, will overwrite completely existing configurations.
    """
    global cuda_config
    cuda_config = config
    
def reset(populations=True, projections=False, synapses = False):
    """
    Reinitialises the network to its state before the call to compile.

    *Parameters*:

    * **populations**: if True (default), the neural parameters and variables will be reset to their initial value.
    * **projections**: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
    * **synapses**: if True, the synaptic weights will be erased and recreated (default=False).
    """
    if populations:
        for pop in _populations:
            pop.reset()
            
    if projections:
        for proj in _projections:
            proj.reset(synapses)

    _network.set_time(0)
        
def get_population(name):
    """
    Returns the population with the given *name*.
    
    *Parameter*:
    
    * **name**: name of the population

    Returns:
    
    * The requested ``Population`` object if existing, ``None`` otherwise.
    """
    for pop in _populations:
        if pop.name == name:
            return pop
        
    print("Error: no population",name,"found.")
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
    
def simulate(duration, measure_time = False):
    """
    Runs the network for the given duration in milliseconds. The number of simulation steps is  computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms)::

        simulate(1000.0)

    *Parameters*:

    * **duration**: the duration in milliseconds.
    * **measure_time**: defines whether the simulation time should be printed (default=False).
    """
    nb_steps = ceil(float(duration) / config['dt'])

    if _network:      
        if measure_time:
            tstart = time.time() 
        _network.pyx_run(nb_steps)
        if measure_time:
            print('Simulating', duration/1000.0, 'seconds of the network took', time.time() - tstart, 'seconds.')
    else:
        _error('simulate(): the network is not compiled yet.')
        return
    
def simulate_until(max_duration, population, operator='and', measure_time = False):
    """
    Runs the network for the maximal duration in milliseconds. If the ``stop_condition`` defined in the population becomes true during the simulation, it is stopped.

    One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

    Example::

        pop1 = Population( ..., stop_condition = "r > 1.0 : any")
        compile()
        simulate_until(max_duration=1000.0. population=pop1)

    *Parameters*:

    * **duration**: the maximum duration of the simulation in milliseconds.
    * **population**: the (list of) population whose ``stop_condition`` should be checked to stop the simulation.
    * **operator**: operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
    * **measure_time**: defines whether the simulation time should be printed (default=False).

    *Returns*:

    * the actual duration of the simulation in milliseconds.
    """
    nb_steps = ceil(float(max_duration) / config['dt'])
    if not isinstance(population, list):
        population = [population]
    if _network:      
        if measure_time:
            tstart = time.time() 
        nb = _network.pyx_run_until(nb_steps, [pop.id for pop in population], True if operator=='and' else False)
        sim_time = float(nb) / config['dt']
        if measure_time:
            print('Simulating', nb/config['dt']/1000.0, 'seconds of the network took', time.time() - tstart, 'seconds.')
        return sim_time
    else:
        _error('simulate(): the network is not compiled yet.')
        return 0.0

def step():
    """
    Performs a single simulation step (duration = ``dt``). 

    """
    if _network:      
        _network.pyx_step()
    else:
        _error('simulate(): the network is not compiled yet.')
        return 0.0

################################
## Learning flags
################################
def enable_learning(projections=None):
    """
    Enables learning for all projections.
    
    *Parameter*:
    
    * **projections**: the projections whose learning should be enabled. By default, all the existing projections are enabled.
    """
    if not projections:
        projections = _projections
    for proj in projections:
        proj.enable_learning()
        
def disable_learning(projections=None):
    """
    Disables learning for all projections.
    
    *Parameter*:
    
    * **projections**: the projections whose learning should be disabled. By default, all the existing projections are disabled.
    """
    if not projections:
        projections = _projections
    for proj in projections:
        proj.disable_learning()
    
################################
## Time
################################
def get_time():
    return _network.get_time()/config['dt']
def set_time(t):
    return _network.set_time(int(t*config['dt']))
def get_current_step():
    return _network.get_time()
def set_current_step(t):
    return _network.set_time(int(t))
def dt():
    return config['dt']

################################
## Recording
################################
def start_record(to_record, period = {}):
    """
    Starts recording of variables in different populations. 
    
    *Parameter*:
    
    * **to_record**: a dictionary with population objects (or names) as keys and variable names as values (either a single string or a list of strings). 

    * **period**: a dictionary with population objects as keys and record periods (in ms) as values (default: dt for each).

    Example::
    
        to_record = { 
            pop1 : ['mp', 'r'], 
            pop2 : 'mp'    
        }
        start_record(to_record)
    """
    global _recorded_populations
    _recorded_populations = to_record

    for pop, variables in to_record.iteritems():
        # get name and object
        pop_obj = pop if not isinstance(pop, str) else get_population(pop)
        pop_name = pop if isinstance(pop, str) else pop.name

        try: # key == obj
            period = record_period[pop_obj]
        except:
            try: # key == name
                period = record_period[pop_name]
            except: # nothing set
                period = config['dt']

        # start recording
        pop_obj.start_record(variables, period)

def get_record(to_record=None, reshape=False):
    """
    Retrieves the recorded variables of one or more populations since the last call. 
  
    *Parameter*:
    
    * **to_record**: a dictionary containing population objects (or names) as keys and variable names as values. For more details check Population.start_record(). When omitted, the dictionary provided in the last call to start_record() is used.

    * **reshape**: defines if the recorded variables should be reshaped to match the population geometry (default: False).
    
    Returns:
    
    * A dictionary containing all recorded values. The dictionary is empty if no recorded data is available.
    
    Example::
    
        to_record = { 
            pop1 : ['mp', 'r'], 
            pop2: 'mp'    
        }
        start_record(to_record)
        simulate(1000.0)
        data = get_record()
        
    """   
    if not to_record:
        to_record = _recorded_populations

    data = {}
    
    for pop, variables in to_record.iteritems():
        if not isinstance(pop, str):
            pop_object = pop
        else:
            pop_object = get_population(pop)

        data[pop] = pop_object.get_record(variables, reshape)
    
    return data  

        
def stop_record(to_record=None):
    """
    Stops the recording of variables in different populations. 
    
    *Parameter*:
    
    * **to_record**: a dictionary with population objects (or names) as keys and variable names as values (either a single string or a list of strings). For more details check Population.stop_record(). When omitted, the dictionary provided in the last call to start_record() is used.
    """
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.iteritems():
        if not isinstance(pop, str):
            pop.stop_record(variables)
        else:
            get_population(pop).stop_record(variables)

        
def pause_record(to_record=None):
    """
    Pauses the recording of variables in different populations. 
    
    *Parameter*:
    
    * **to_record**: a dictionary with population objects (or names) as keys and variable names as values (either a single string or a list of strings). For more details check Population.pause_record(). When omitted, the dictionary provided in the last call to start_record() is used.
    """
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.iteritems():
        if not isinstance(pop, str):
            pop.pause_record(variables)
        else:
            get_population(pop).pause_record(variables)

        
def resume_record(to_record=None):
    """
    Resumes the recording of variables in different populations. 
    
    *Parameter*:
    
    * **to_record**: a dictionary with population objects (or names) as keys and variable names as values (either a single string or a list of strings). For more details check Population.resume_record(). When omitted, the dictionary provided in the last call to start_record() is used.
    """
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.iteritems():
        if not isinstance(pop, str):
            pop.resume_record(variables)
        else:
            get_population(pop).resume_record(variables)

################################
## Printing
################################

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
    if not config['suppress_warnings']:
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

