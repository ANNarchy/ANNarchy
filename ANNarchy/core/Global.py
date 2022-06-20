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
import inspect
import traceback
import numpy as np
import argparse

from ANNarchy.core.NetworkManager import NetworkManager

# High-level structures
_objects = {
    'functions': [],
    'neurons': [],
    'synapses': [],
    'constants': [],
}

# Data for the different networks
_network = NetworkManager()

# Configuration
config = dict(
   {
    'dt' : 1.0,
    'verbose': False,
    'debug': False,
    'show_time': False,
    'suppress_warnings': False,
    'num_threads': 1,
    'visible_cores': [],
    'paradigm': "openmp",
    'method': "explicit",
    'precision': "double",
    'only_int_idx_type': True,
    'seed': -1,
    'structural_plasticity': False,
    'profiling': False,
    'profile_out': None,
    'disable_parallel_rng': True,
    'use_seed_seq': True,
    'use_cpp_connectors': False,
    'disable_split_matrix': True,
    'disable_SIMD_SpMV': False
   }
)

# Profiling instance
_profiler = None

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

    * dt: simulation step size (default: 1.0 ms).
    * paradigm: parallel framework for code generation. Accepted values: "openmp" or "cuda" (default: "openmp").
    * method: default method to numerize ODEs. Default is the explicit forward Euler method ('explicit').
    * precision: default floating precision for variables in ANNarchy. Accepted values: "float" or "double" (default: "double")
    * only_int_idx_type: if set to True (default) only signed integers are used to store pre-/post-synaptic ranks which was default until 4.7.
                         If set to False, the index type used in a single projection is selected based on the size of the corresponding populations.
    * num_threads: number of treads used by openMP (overrides the environment variable ``OMP_NUM_THREADS`` when set, default = None).
    * visible_cores: allows a fine-grained control which cores are useable for the created threads (default = [] for no limitation).
                     It can be used to limit created openMP threads to a physical socket.
    * structural_plasticity: allows synapses to be dynamically added/removed during the simulation (default: False).
    * seed: the seed (integer) to be used in the random number generators (default = -1 is equivalent to time(NULL)).
    * disable_parallel_rng: determines if random numbers drawn from distributions are generated from a single source (default: True). 
                            If this flag is set to true only one RNG source is used und the values are drawn by one thread which 
                            reduces parallel performance (this is the behavior of all ANNarchy versions prior to 4.7). 
                            If set to false a seed sequence is generated to allow usage of one RNG per thread. Please note, that this
                            flag won't effect the GPUs which draw from multiple sources anyways.
    * use_seed_seq: If parallel RNGs are used the single generators need to be initialized. By default (use_seed_seq == True) we use
                    the STL seed sequence to generate a list of seeds from the given master seed (*seed* argument). If set to False,
                    we use a simpler initialization strategy adapted from NEST.
    * use_cpp_connectors:   For some of the default connectivity methods of ANNarchy we offer a CPP-side construction of the pattern to improve the
                            initialization time (default=False). For maximum performance the disable_parallel_rng should be set to False to allow
                            a parallel construction of the pattern.
    * disable_split_matrix: determines if projections can use thread-local allocation. If set to *True* (default) no thread local allocation is allowed.
                            This equals the behavior of ANNarchy until 4.7. If set to *False* the code generator can use sliced versions if they
                            are available.
    * disable_SIMD_SpMV: determines if the hand-written implementation is used (by default True) if the current hardware platform and used sparse matrix
                         format does support the vectorization). Disabling is intended for performance analysis.


    The following parameters are mainly for debugging and profiling, and should be ignored by most users:

    * verbose: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
    * suppress_warnings: if True, warnings (e. g. from the mathematical parser) are suppressed.
    * show_time: if True, initialization times are shown. Attention: verbose should be set to True additionally.


    **Note:**

    This function should be used before any other functions of ANNarchy (including importing a network definition), right after `from ANNarchy import *`:

    ```python
    from ANNarchy import *
    setup(dt=1.0, method='midpoint', cores=2)
    ```

    """
    if len(_network[0]['populations']) > 0 or len(_network[0]['projections']) > 0 or len(_network[0]['monitors']) > 0:
        if 'dt' in keyValueArgs:
            _warning('setup(): populations or projections have already been created. Changing dt now might lead to strange behaviors with the synaptic delays (internally generated in steps, not ms)...')
        if 'precision' in keyValueArgs:
            _warning('setup(): populations or projections have already been created. Changing precision now might lead to strange behaviors...')

    for key in keyValueArgs:
        if key in config.keys():
            config[key] = keyValueArgs[key]

            if key == "use_cpp_connectors":
                _warning("use_cpp_connectors=True is currently disabled, will be enabled soon.")
                config["use_cpp_connectors"] = False

        else:
            _warning('setup(): unknown key:', key)

        if key == 'seed': # also seed numpy
            np.random.seed(keyValueArgs[key])

def clear():
    """
    Clears all variables (erasing already defined populations, projections, monitors and constants), as if you had just imported ANNarchy.

    Useful when re-running Jupyter/IPython notebooks multiple times:

    ```python
    from ANNarchy import *
    clear()
    ```
    """
    # Reset objects
    global _objects
    _objects = {
        'functions': [],
        'neurons': [],
        'synapses': [],
        'constants': [],
    }

    # Reinitialize initial state
    global _network
    _network.clear()

    # # Configuration
    # config = dict(
    #    {
    #     'dt' : 1.0,
    #     'verbose': False,
    #     'show_time': False,
    #     'suppress_warnings': False,
    #     'num_threads': 1,
    #     'paradigm': "openmp",
    #     'method': "explicit",
    #     'precision': "double",
    #     'seed': -1,
    #     'structural_plasticity': False,
    #     'profiling': False,
    #     'profile_out': None
    #    }
    # )


def reset(populations=True, projections=False, synapses=False, monitors=True, net_id=0):
    """
    Reinitialises the network to its state before the call to compile. The network time will be set to 0ms.

    All monitors are emptied.

    :param populations: if True (default), the neural parameters and variables will be reset to their initial value.
    :param projections: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
    :param synapses: if True, the synaptic weights will be erased and recreated (default=False).
    :param monitors: if True, the monitors will be emptied and reset (default=True).
    """

    _network[net_id]['instance'].set_time(0)
    
    if populations:
        for pop in _network[net_id]['populations']:
            pop.reset()

        # pop.reset only clears spike container with no or uniform delay
        for proj in _network[net_id]['projections']:
            if hasattr(proj.cyInstance, 'reset_ring_buffer'):
                proj.cyInstance.reset_ring_buffer()

    if synapses and not projections:
        _warning("reset(): if synapses is set to true this automatically enables projections==true")
        projections = True

    if projections:
        for proj in _network[net_id]['projections']:
            proj.reset(attributes=-1, synapses=synapses)

    if monitors:
        for monitor in _network[net_id]['monitors']:
            monitor.reset()


def get_population(name, net_id=0):
    """
    Returns the population with the given ``name``.

    :param name: name of the population.
    :return: The requested ``Population`` object if existing, ``None`` otherwise.
    """
    for pop in _network[net_id]['populations']:
        if pop.name == name:
            return pop

    _warning("get_population(): the population", name, "does not exist.")
    return None

def get_projection(name, net_id=0):
    """
    Returns the projection with the given *name*.

    :param name: name of the projection.
    :return: The requested ``Projection`` object if existing, ``None`` otherwise.
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

def projections(net_id=0, post=None, pre=None, target=None, suppress_error=False):
    """
    Returns a list of all declared populations. By default, the method returns all connections which were defined.
    By setting *one* of the arguments, post, pre and target one can select a subset accordingly.

    :param post: all returned projections should have this population as post.
    :param pre: all returned projections should have this population as pre.
    :param target: all returned projections should have this target.
    :param suppress_error: by default, ANNarchy throws an error if the list of assigned projections is empty. If this flag is set to True, the error message is suppressed.
    :return: A list of all assigned projections in this network. Or a subset
    according to the arguments.
    """
    if post is None and pre is None and target is None:
        return _network[net_id]['projections']
    else:
        res = []
        if isinstance(post, str):
            post = get_population(post, net_id)
        if isinstance(pre, str):
            pre = get_population(pre, net_id)

        # post is the criteria
        if (post is not None) and (pre is None) and (target is None) :
            for proj in _network[net_id]['projections']:
                if proj.post == post:
                    res.append(proj)

        # pre is the criteria
        elif (pre is not None) and (post is None) and (target is None):
            for proj in _network[net_id]['projections']:
                if proj.pre == pre:
                    res.append(proj)

        # post is the criteria
        elif target is not None and (post is None) and (pre is None):
            for proj in _network[net_id]['projections']:
                if proj.target == target:
                    res.append(proj)

        else:
            raise ValueError("ANNarchy.core.Global.projections(): either none or one of the arguments post, pre, target must be set.")

        return res


################################
## Functions
################################
def add_function(function):
    """
    Defines a global function which can be used by all neurons and synapses.

    The function must have only one return value and use only the passed arguments.

    Examples of valid functions:

    ```python
    logistic(x) = 1 / (1 + exp(-x))

    piecewise(x, a, b) =    if x < a:
                                a
                            else:
                                if x > b :
                                    b
                                else:
                                    x
    ```

    Please refer to the manual to know the allowed mathematical functions.
    """
    name = function.split('(')[0]
    _objects['functions'].append( (name, function))


def functions(name, net_id=0):
    """
    Allows to access a global function defined with ``add_function`` and use it from Python using arrays **after compilation**.

    The name of the function is not added to the global namespace to avoid overloading.
    
    ```python
    add_function("logistic(x) = 1. / (1. + exp(-x))") 

    compile()  

    result = functions('logistic')([0., 1., 2., 3., 4.])
    ```
 
    Only lists or 1D Numpy arrays can be passed as arguments, not single values nor multidimensional arrays.

    When passing several arguments, make sure they have the same size.

    """
    try:
        func = getattr(_network[net_id]['instance'], 'func_' + name)
    except:
        _error('call to', name, ': the function is not compiled yet.')

    return func

################################
## Constants
################################
class Constant(float):
    """
    Constant parameter that can be used by all neurons and synapses.

    The class ``Constant`` derives from ``float``, so any legal operation on floats (addition, multiplication) can be used.

    If a Neuron/Synapse defines a parameter with the same name, the constant parameters will not be visible.

    Example:

    ```python

    tau = Constant('tau', 20)
    factor = Constant('factor', 0.1)
    real_tau = Constant('real_tau', tau*factor)

    neuron = Neuron(
        equations='''
            real_tau*dr/dt + r =1.0
        '''
    )
    ```

    The value of the constant can be changed anytime with the ``set()`` method. Assignments will have no effect (e.g. ``tau = 10.0`` only creates a new float).

    The value of constants defined as combination of other constants (``real_tau``) is not updated if the value of these constants changes (changing ``tau`` with ``tau.set(10.0)`` will not modify the value of ``real_tau``).

    """
    def __new__(cls, name, value, net_id=0):
        return float.__new__(cls, value)
        
    def __init__(self, name, value, net_id=0):
        """
        :param name: name of the constant (unique), which can be used in equations.
        :param value: the value of the constant, which must be a float, or a combination of Constants.
        """

        self.name = name
        self.value = value
        self.net_id = net_id
        for obj in _objects['constants']:
            if obj.name == name:
                _error('the constant', name, 'is already defined.')
        _objects['constants'].append(self)
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return self.__str__()
    def set(self, value):
        "Changes the value of the constant."
        self.value = value
        if _network[self.net_id]['compiled']:
            getattr(_network[self.net_id]['instance'], '_set_'+self.name)(self.value)

def list_constants(net_id=0):
    """
    Returns a list of all constants declared with ``Constant(name, value)``.
    """
    l = []
    for obj in _objects['constants']:
        l.append(obj.name)
    return l

def get_constant(name, net_id=0):
    """
    Returns the ``Constant`` object with the given name, ``None`` otherwise.
    """
    for obj in _objects['constants']:
        if obj.name == name:
            return obj
    return None


################################
## Memory management
################################
def _bytes_human_readable(size_in_bytes):
    """ Transforms given size in GB/MB/KB or bytes dependent on the value. """
    if size_in_bytes > (1024*1024*1024):
        return "{:.2f} GB".format(float(size_in_bytes)/(1024.0*1024.0*1024.0))
    elif size_in_bytes > (1024*1024):
        return "{:.2f} MB".format(float(size_in_bytes)/(1024.0*1024.0))
    elif size_in_bytes > (1024):
        return "{:.2f} KB".format(float(size_in_bytes)/(1024.0))
    else:
        return str(size_in_bytes) + " bytes"

def _cpp_memory_footprint(net_id=0):
    """
    Print the C++ memory consumption for populations, projections on the console.

    :param net_id: net_id of the requested network.
    """
    print("Memory consumption of C++ objects: ")

    for pop in populations(net_id):
        print(pop.name, _bytes_human_readable(pop.size_in_bytes()))

    for proj in projections(net_id):
        print(proj.name, _bytes_human_readable(proj.size_in_bytes()))

    for mon in _network[net_id]['monitors']:
        print(mon.name, _bytes_human_readable(mon.size_in_bytes()))

def _python_current_max_rusage():
    """
    Prints the current max residen size for the current process and the children.
    """
    import resource
    size_kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(_bytes_human_readable(size_kilobytes*1024))

################################
## Learning flags
################################
def enable_learning(projections=None, period=None, offset=None, net_id=0):
    """
    Enables learning for all projections. Optionally *period* and *offset* can be changed for all projections.

    :param projections: the projections whose learning should be enabled. By default, all the existing projections are enabled.
    :param period: determines how often the synaptic variables will be updated.
    :param offset: determines the offset at which the synaptic variables will be updated relative to the current time.

    """
    if not projections:
        projections = _network[net_id]['projections']
    for proj in projections:
        proj.enable_learning(period, offset)

def disable_learning(projections=None, net_id=0):
    """
    Disables learning for all projections.

    :param projections: the projections whose learning should be disabled. By default, all the existing projections are disabled.
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
    """
    Sets the current time in ms.

    **Warning:** can be dangerous for some spiking models.
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
    """
    Sets the current simulation step (integer).

    **Warning:** can be dangerous for some spiking models.
    """
    try:
        _network[net_id]['instance'].set_time(int(t))
    except:
        _warning('Time can only be set when the network is compiled.')

def dt():
    "Returns the simulation step size `dt` used in the simulation."
    return config['dt']

################################
## Seed
################################
def set_seed(seed, use_seed_seq=True, net_id=0):
    "Sets the seed of the random number generators, both in numpy.random and in the C++ library when it is created."
    config['seed'] = seed
    config['use_seed_seq'] = use_seed_seq
    if seed > -1:
        np.random.seed(seed)
    
    try:
        if config['disable_parallel_rng']:
            _network[net_id]['instance'].set_seed(seed, 1, use_seed_seq)
        else:
            _network[net_id]['instance'].set_seed(seed, config['num_threads'], use_seed_seq)
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

def _check_precision(precision):
    """
    Returns True when the provided precision is currently used.

    Possible values:

    1. "float"
    2. "double"
    """
    try:
        return precision == config['precision']
    except KeyError:
        _error("Unknown precision")



################################
## Printing
################################

def _print(*var_text, end="\n", flush=False):
    """
    Prints a message to standard out.
    """
    text = ''
    for var in var_text:
        text += str(var) + ' '

    if sys.version_info.major == 3:
        print(text, end=end, flush=flush)
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
    print(text)

def _warning(*var_text):
    """
    Prints a warning message to standard out. Can be suppressed by configuration.
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

def _info(*var_text):
    """
    Prints a information message to standard out. Can be suppressed by configuration.
    """
    text = 'INFO: '
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
    Custom exception that can be catched in some cases (IO) instead of quitting.
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

class CodeGeneratorException(Exception):
    def __init__(self, msg):
        print("An error in the code generation occured:")
        sys.exit(self)

class InvalidConfiguration(Exception):
    def __init__(self, msg):
        print("The configuration you requested is not implemented in ANNarchy.")
        sys.exit(self)
