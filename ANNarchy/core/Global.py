#===============================================================================
#
#     Global.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
import sys, os
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
        'compiled': False,
        'directory': os.getcwd() + "/annarchy/"
    },
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
    'precision': "double",
    'seed': -1,
    'structural_plasticity': False,
    'profiling': False,
    'profile_out': None
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
    'projection',
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

def setup(**keyValueArgs):
    """
    The setup function is used to configure ANNarchy simulation environment. It takes various optional arguments:

    *Parameters*:

    * **dt**: simulation step size (default: 1.0 ms).

    * **paradigm**: parallel framework for code generation. Accepted values: "openmp" or "cuda" (default: "openmp").

    * **method**: default method to numerize ODEs. Default is the explicit forward Euler method ('explicit').

    * **precision**: default floating precision for variables in ANNarchy. Accepted values: "float" or "double" (default: "double")

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

def clear():
    """
    Clears all variables (erasing already defined populations and variables), as if you had just imported ANNarchy.

    Useful when re-running Jupyter/IPython notebooks multiple times::

        from ANNarchy import *
        clear()
        ...
        compile()
    """
    # Reset objects 
    _objects = {
        'functions': [],
        'neurons': [],
        'synapses': [],
    }

    # Data for the different networks
    global _network
    for net in _network:
        for pop in net['populations']:
            del pop
        for proj in net['projections']:
            del proj
        for m in net['monitors']:
            del m
    _network.clear()
    _add_network()

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
        'precision': "double",
        'seed': -1,
        'structural_plasticity': False,
        'profiling': False,
        'profile_out': None
       }
    )


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

    * **name**: name of the population.

    Returns:

    * The requested ``Population`` object if existing, ``None`` otherwise.
    """
    for pop in _network[net_id]['populations']:
        if pop.name == name:
            return pop

    _warning("get_population(): the population", name, "does not exist.")
    return None

def get_projection(name, net_id=0):
    """
    Returns the projection with the given *name*.

    *Parameter*:

    * **name**: name of the projection.

    Returns:

    * The requested ``Projection`` object if existing, ``None`` otherwise.
    """
    for proj in _network[net_id]['projections']:
        if proj.name == name:
            return proj

    _warning("get_projection(): the projection", name, "does not exist.")
    return None

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

    .. code-block:: python

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
    name = function.split('(')[0]
    _objects['functions'].append( (name, function))

def functions(name, net_id=0):
    """
    Allows to access a global function defined with ``add_function`` and use it from Python using arrays **after compilation**.

    The name of the function is not added to the global namespace to avoid overloading.

    .. code-block:: python
    
        add_function("logistic(x) = 1. / (1. + exp(-x))") 

        compile()  

        result = functions('logistic')([0., 1., 2., 3., 4.])
 
    Only lists or 1D Numpy arrays can be passed as arguments, not single values nor multidimensional arrays.

    When passing several arguments, make sure they have the same size.

    """
    try:
        func = getattr(_network[net_id]['instance'], 'func_' + name)
    except:
        _error('call to', name, ': the function is not compiled yet.')

    return func

################################
## Networks
################################
def _add_network():
    """
    Adds an empty structure for a new network.
    """
    _network.append(
        {
            'populations': [],
            'projections': [],
            'monitors': [],
            'instance': None,
            'compiled': False,
            'directory': os.getcwd() + "/annarchy/"
        }
    )

################################
## Learning flags
################################
def enable_learning(projections=None, period=None, offset=None, net_id=0):
    """
    Enables learning for all projections. Optionally *period* and *offset* can be changed for all projections.

    *Parameter*:

    * **projections**: the projections whose learning should be enabled. By default, all the existing projections are enabled.
    * **period** determines how often the synaptic variables will be updated.
    * **offset** determines the offset at which the synaptic variables will be updated relative to the current time.

    """
    if not projections:
        projections = _network[net_id]['projections']
    for proj in projections:
        proj.enable_learning(period, offset)

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
## Paradigm
################################
def _check_paradigm(paradigm):
    """
    Returns True when the provided paradigm is currently used.

    Possible values:

    1. "openmp"
    2. "cuda"
    """
    try:
        return paradigm == config['paradigm']
    except KeyError:
        _error("Unknown paradigm")


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
    #     # Print the trace
    #     tb = traceback.format_stack()
    #     for line in tb:
    #         if not '/ANNarchy/core/' in line and \
    #            not '/ANNarchy/parser/' in line and \
    #            not '/ANNarchy/generator/' in line :
    #             print(line)

def _error(*var_text, **args):
    """
    Prints an error message to standard out and exits.

    When passing exit=False, the program will not exit.
    """
    text = ''
    for var in var_text:
        text += str(var) + ' '

    exit = False
    if 'exit' in args.keys():
        if args['exit']:
            exit = True
    else:
        exit = True

    if exit:
        raise ANNarchyException(text, exit)
    else:
        print('ERROR:' + text)

class ANNarchyException(Exception):
    """
    Custom exception that can be ctached in some cases (IO) instead of quitting.
    """
    def __init__(self, message, exit):
        super(ANNarchyException, self).__init__(message)

        # # Print the error message
        # print('ERROR: ' + message)

        # # Print the trace
        # # tb = traceback.print_stack()
        # tb = traceback.format_stack()
        # for line in tb:
        #     if not '/ANNarchy/core/' in line and \
        #        not '/ANNarchy/parser/' in line and \
        #        not '/ANNarchy/generator/' in line :
        #         print(line)