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
import time
import traceback
import numpy as np

# High-level structures
_objects = {
    'functions': [],
    'neurons': [],
    'synapses': [],
}

# Data for the different networks
_network = [
    {
    'populations': [],
    'projections': [],
    'monitors': [],
    'instance': None,
    'compiled': False
    }
]

# Configuration
config = dict(
   {
    'dt' : 1.0,
    'verbose': False,
    'show_time': False,
    'suppress_warnings': False,
    'num_threads': 1,
    'paradigm': "openmp",
    'method': "explicit",
    'seed': -1,
    'structural_plasticity': False,
    'profiling': False
   }
)

# Configuration for CUDA
cuda_config = dict(
    {
     'device': 0
    }
)

# Minimum number of neurons to apply OMP parallel regions
OMP_MIN_NB_NEURONS = 100

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
    'event-driven',
    # Refractory
    'unless_refractory',
    # Type
    'int',
    'bool',
    'float',
    # Event-based
    'unless_post',
]

# Dictionary of population variables being currently recorded
_recorded_populations = {}

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy simulation environment. It takes various optional arguments:

    *Parameters*:

    * **dt**: simulation step size (default: 1.0 ms).

    * **paradigm**: parallel framework for code generation. Accepted values: "openmp" or "cuda" (default: "openmp").

    * **method**: default method to numerize ODEs. Default is the explicit forward Euler method ('explicit').

    * **num_threads**: number of treads used by openMP (overrides the environment variable ``OMP_NUM_THREADS`` when set, default = None).

    * **structural_plasticity**: allows synapses to be dynamically added/removed during the simulation (default: False).

    * **seed**: the seed (integer) to be used in the random number generators (default = -1 is equivalent to time(NULL)).

    The following parameters are mainly for debugging and profiling, and should be ignored by most users:

    * **verbose**: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.

    * **suppress_warnings**: if True, warnings (e. g. from the mathematical parser) are suppressed.

    * **show_time**: if True, initialization times are shown. Attention: verbose should be set to True additionally.


    .. note::

        This function should be used before any other functions of ANNarchy (including importing a network definition), right after ``from ANNarchy import *``::

            from ANNarchy import *
            setup(dt=1.0, method='midpoint', num_threads=2)
            ...

    """
    if len(_network[0]['populations']) > 0 or len(_network[0]['projections']) > 0 or len(_network[0]['monitors']) > 0:
        _warning('setup(): populations or projections have already been defined. Changing a setup parameter now might lead to strange behaviors...')
        _print('In particular, changing dt after a projection has been created might create problems with the synaptic delays (internally generated in steps, not ms).')

    for key in keyValueArgs:
        if key in config.keys():
            config[key] = keyValueArgs[key]
        else:
            _warning('setup(): unknown key:', key)

        if key == 'seed': # also seed numpy
            np.random.seed(keyValueArgs[key])


def reset(populations=True, projections=False, synapses = False, net_id=0):
    """
    Reinitialises the network to its state before the call to compile.

    *Parameters*:

    * **populations**: if True (default), the neural parameters and variables will be reset to their initial value.
    * **projections**: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
    * **synapses**: if True, the synaptic weights will be erased and recreated (default=False).
    """
    if populations:
        for pop in _network[net_id]['populations']:
            pop.reset()

    if projections:
        for proj in _network[net_id]['projections']:
            proj.reset(synapses)

    _network[net_id]['instance'].set_time(0)

def get_population(name, net_id=0):
    """
    Returns the population with the given *name*.

    *Parameter*:

    * **name**: name of the population

    Returns:

    * The requested ``Population`` object if existing, ``None`` otherwise.
    """
    for pop in _network[net_id]['populations']:
        if pop.name == name:
            return pop

    _error("get_population(): the population", name, "does not exist.")

def populations(net_id=0):
    """
    Returns a list of all declared populations.
    """
    return _network[net_id]['populations']

def projections(net_id=0):
    """
    Returns a list of all declared projections.
    """
    return _network[net_id]['projections']

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
    _objects['functions'].append(function)


################################
## Learning flags
################################
def enable_learning(projections=None, net_id=0):
    """
    Enables learning for all projections.

    *Parameter*:

    * **projections**: the projections whose learning should be enabled. By default, all the existing projections are enabled.
    """
    if not projections:
        projections = _network[net_id]['projections']
    for proj in projections:
        proj.enable_learning()

def disable_learning(projections=None, net_id=0):
    """
    Disables learning for all projections.

    *Parameter*:

    * **projections**: the projections whose learning should be disabled. By default, all the existing projections are disabled.
    """
    if not projections:
        projections = _network[net_id]['projections']
    for proj in projections:
        proj.disable_learning()

################################
## Time
################################
def get_time(net_id=0):
    "Returns the current time in ms."
    try:
        t = _network[net_id]['instance'].get_time()*config['dt']
    except:
        t = 0.0
    return t

def set_time(t, net_id=0):
    """Sets the current time in ms.

    .. warning::

        Can be dangerous for some spiking models.
    """
    try:
        _network[net_id]['instance'].set_time(int(t/config['dt']))
    except:
        _warning('Time can only be set when the network is compiled.')

def get_current_step(net_id=0):
    "Returns the current simulation step."
    try:
        t = _network[net_id]['instance'].get_time()
    except:
        t = 0
    return t

def set_current_step(t, net_id=0):
    """Sets the current simulation step.

    .. warning::

        Can be dangerous for some spiking models.
    """
    try:
        _network[net_id]['instance'].set_time(int(t))
    except:
        _warning('Time can only be set when the network is compiled.')

def dt():
    "Returns the simulation step size ``dt`` used in the simulation."
    return config['dt']

################################
## Seed
################################
def set_seed(seed, net_id=0):
    "Sets the seed of the random number generators, both in numpy.random and in the C++ library when it is created."
    config['seed'] = seed
    if seed > -1:
        np.random.seed(seed)
    try:
        _network[net_id]['instance'].set_seed(seed)
    except:
        _warning('The seed will only be set in the simulated network when it is compiled.')



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
    _warning('Recording through global methods is deprecated. Use the Monitor object instead.')
    global _recorded_populations
    _recorded_populations = to_record

    for pop, variables in to_record.items():
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
    _warning('Recording through global methods is deprecated. Use the Monitor object instead.')
    if not to_record:
        to_record = _recorded_populations

    data = {}

    for pop, variables in to_record.items():
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
    _warning('Recording through global methods is deprecated. Use the Monitor object instead.')
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.items():
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
    _warning('Recording through global methods is deprecated. Use the Monitor object instead.')
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.items():
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
    _warning('Recording through global methods is deprecated. Use the Monitor object instead.')
    if not to_record:
        to_record = _recorded_populations
    for pop, variables in to_record.items():
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
    print(text)

def _warning(*var_text):
    """
    Prints a warning message to standard out.
    """
    text = 'WARNING: '
    for var in var_text:
        text += str(var) + ' '
    if not config['suppress_warnings']:
        print(text)

def _error(*var_text, **args):
    """
    Prints an error message to standard out and exits.

    When passing exit=False, the program will not exit.
    """
    text = 'ERROR: '
    for var in var_text:
        text += str(var) + ' '

    print(text)

    # tb = traceback.print_stack()
    tb = traceback.format_stack()
    for line in tb:
        if not '/ANNarchy/core/' in line and \
           not '/ANNarchy/parser/' in line and \
           not '/ANNarchy/generator/' in line :
            print(line)

    if 'exit' in args.keys():
        if args['exit']:
            sys.exit(1)
    else:
        sys.exit(1)
