"""
Contains global available functions and state variables,
e.g., the config dictionary and holds a reference to 
network instances.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from typing import Any

from ANNarchy.intern.ConfigManagement import get_global_config, _update_global_config
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern import Messages

# Minimum number of neurons to apply OMP parallel regions
OMP_MIN_NB_NEURONS = 100

def clear(functions:bool=True, neurons:bool=True, synapses:bool=True, constants:bool=True):
    """
    Clears all variables (erasing already defined populations, projections, monitors and constants), as if you had just imported ANNarchy.

    Useful when re-running Jupyter/IPython notebooks multiple times:

    ```python
    import ANNarchy as ann
    ann.clear()
    ```

    :param functions: if True (default), all functions defined with ``add_function`` are erased.
    :param neurons: if True (default), all neurons defined with ``Neuron`` are erased.
    :param synapses: if True (default), all synapses defined with ``Synapse`` are erased.
    :param constants: if True (default), all constants defined with ``Constant`` are erased.
    """
    # Reset globally defined objects
    GlobalObjectManager().clear(functions=functions, neurons=neurons, synapses=synapses, constants=constants)

    # Remove the present profiler
    if Profiler().enabled:
        check_profile_results()
        Profiler().disable_profiling()

    # Reinitialize initial state
    NetworkManager().clear()

def check_profile_results():
    """
    If the user enabled profiling, we here check if we recorded some results.
    """
    if Profiler().enabled:
        Profiler().print_profile()

        Profiler().store_cpp_time_as_csv()

def reset(populations:bool=True, projections:bool=False, synapses:bool=False, monitors:bool=True, net_id:int=0):
    """
    Reinitialises the network to its state before the call to compile. The network time will be set to 0ms.

    All monitors are emptied.

    :param populations: if True (default), the neural parameters and variables will be reset to their initial value.
    :param projections: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
    :param synapses: if True, the synaptic weights will be erased and recreated (default=False).
    :param monitors: if True, the monitors will be emptied and reset (default=True).
    """

    NetworkManager().cy_instance(net_id=net_id).set_time(0)
    
    if populations:
        for pop in NetworkManager().get_populations(net_id=net_id):
            pop.reset()

        # pop.reset only clears spike container with no or uniform delay
        for proj in NetworkManager().get_projections(net_id=net_id):
            if hasattr(proj.cyInstance, 'reset_ring_buffer'):
                proj.cyInstance.reset_ring_buffer()

    if synapses and not projections:
        Messages._warning("reset(): if synapses is set to true this automatically enables projections==true")
        projections = True

    if projections:
        for proj in NetworkManager().get_projections(net_id=net_id):
            proj.reset(attributes=-1, synapses=synapses)

    if monitors:
        for monitor in NetworkManager().get_monitors(net_id=net_id):
            monitor.reset()

################################
## Accessing shadow network
################################
def get_population(name:str, net_id:int=0) -> "Population":
    """
    Returns the population with the given name.

    :param name: name of the population.
    :return: The requested ``Population`` object if existing, ``None`` otherwise.
    """
    for pop in NetworkManager().get_populations(net_id=net_id):
        if pop.name == name:
            return pop

    Messages._warning("get_population(): the population", name, "does not exist.")
    return None

def get_projection(name:str, net_id:int=0) -> "Projection":
    """
    Returns the projection with the given name.

    :param name: name of the projection.
    :return: The requested ``Projection`` object if existing, ``None`` otherwise.
    """
    for proj in NetworkManager().get_projections(net_id=net_id):
        if proj.name == name:
            return proj

    Messages._warning("get_projection(): the projection", name, "does not exist.")
    return None

def populations(net_id:int=0) -> list["Population"]:
    """
    Returns a list of all declared populations.

    :retruns: a list of all populations.
    """
    return NetworkManager().get_populations(net_id=net_id)

def projections(net_id:int=0, 
                post:"Population"=None, pre:"Population"=None, target:str=None, suppress_error:bool=False) -> list["Projection"]:
    """
    Returns a list of all declared populations. By default, the method returns all connections which were defined.
    By setting *one* of the arguments, post, pre and target one can select a subset accordingly.

    :param post: all returned projections should have this population as post.
    :param pre: all returned projections should have this population as pre.
    :param target: all returned projections should have this target.
    :param suppress_error: by default, ANNarchy throws an error if the list of assigned projections is empty. If this flag is set to True, the error message is suppressed.
    :return: A list of all assigned projections in this network, or a subset
    according to the arguments.
    """
    return NetworkManager().get_projections(net_id=net_id, pre=pre, post=post, target=target, suppress_error=suppress_error)

def monitors(net_id:int=0, obj: Any=None) -> list["Monitor"]:
    """
    Returns a list of declared monitors. By default, all monitors are returned.
    By setting *obj*, only monitors recording from this object, either *Population* or *Projection*, will be returned.
    """
    if obj is None:
        return NetworkManager().get_monitors(net_id=net_id)

    else:
        mon_list = []
        for monitor in NetworkManager().get_monitors(net_id=net_id):
            if monitor.object == obj:
                mon_list.append(monitor)

        return mon_list

################################
## Functions
################################
def add_function(function:str):
    """
    Defines a global function which can be used by all neurons and synapses.

    The function must have only one return value and use only the passed arguments.

    Examples of valid functions:

    ```python
    ann.add_function('logistic(x) = 1 / (1 + exp(-x))')

    ann.add_function('''
        piecewise(x, a, b) = if x < a:
                                a
                             else: 
                                if x > b :
                                    b
                                else:
                                    x
    ''')
    ```

    Please refer to the manual to know the allowed mathematical functions.

    :param function: (multi)string representing the function.
    """
    GlobalObjectManager().add_function(function)

def functions(name:str, network=None):
    """
    Allows to access a global function declared with ``add_function`` and use it from Python using arrays **after compilation**.

    The name of the function is not added to the global namespace to avoid overloading.
    
    ```python
    add_function("logistic(x) = 1. / (1. + exp(-x))") 

    ann.compile()  

    result = functions('logistic')([0., 1., 2., 3., 4.])
    ```

    Only lists or 1D Numpy arrays can be passed as arguments, not single values nor multidimensional arrays.

    When passing several arguments, make sure they have the same size.

    :param name: name of the function.
    """
    return GlobalObjectManager().functions(name, network)

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
    NetworkManager()._cpp_memory_footprint(net_id=net_id)

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
def enable_learning(projections:list=None, period:list=None, offset:float=None, net_id:int=0):
    """
    Enables learning for all projections. Optionally *period* and *offset* can be changed for all projections.

    :param projections: the projections whose learning should be enabled. By default, all the existing projections are enabled.
    :param period: determines how often the synaptic variables will be updated.
    :param offset: determines the offset at which the synaptic variables will be updated relative to the current time.

    """
    if not projections:
        projections = NetworkManager().get_projections(net_id=net_id)
    for proj in projections:
        proj.enable_learning(period, offset)

def disable_learning(projections=None, net_id=0):
    """
    Disables learning for all projections.

    :param projections: the projections whose learning should be disabled. By default, all the existing projections are disabled.
    """
    if not projections:
        projections = NetworkManager().get_projections(net_id=net_id)
    for proj in projections:
        proj.disable_learning()

################################
## Time
################################
def get_time(net_id=0) -> float:
    "Returns the current time in ms."
    try:
        t = NetworkManager().cy_instance(net_id=net_id).get_time() * get_global_config('dt')
    except:
        t = 0.0
    return t

def set_time(t:float, net_id=0):
    """
    Sets the current time in ms.

    **Warning:** can be dangerous for some spiking models.
    """
    try:
        NetworkManager().cy_instance(net_id=net_id).set_time(int(t / get_global_config('dt')))
    except:
        Messages._warning('Time can only be set when the network is compiled.')

def get_current_step(net_id=0) -> int:
    "Returns the current simulation step."
    try:
        t = NetworkManager().cy_instance(net_id=net_id).get_time()
    except:
        t = 0
    return t

def set_current_step(t:int, net_id=0):
    """
    Sets the current simulation step (integer).

    **Warning:** can be dangerous for some spiking models.
    """
    try:
        NetworkManager().cy_instance(net_id=net_id).set_time(int(t))
    except:
        Messages._warning('Time can only be set when the network is compiled.')

def dt() -> float:
    "Returns the simulation step size `dt` used in the simulation."
    return get_global_config('dt')

################################
## Seed
################################
def set_seed(seed:int, use_seed_seq:bool=True, net_id:int=0):
    """
    Sets the seed of the random number generators, both in ANNarchy.RandomDistributions and in the C++ library when it is created.

    Numpy still has to be seeded explicitly when using the default RNG, for example:

    ```python
    ann.set_seed(seed=42)
    rng = np.random.default_rng(seed=42)
    A = rng.uniform(0.0, 1.0, (10, 10))
    ```
    
    :param seed: integer value used to seed the C++ and Numpy RNG
    :param use_seed_seq: for openMP and parallel RNGs, we use either the STL SeedSequence (True, default) or a specialized implementation proposed by Melissa O'Neil (False, see _optimization_flags for more details).
    """
    _update_global_config('seed', seed)
    _update_global_config('use_seed_seq', use_seed_seq)
    
    try:
        if get_global_config('disable_parallel_rng'):
            NetworkManager().cy_instance(net_id=net_id).set_seed(seed, 1, use_seed_seq)
        else:
            NetworkManager().cy_instance(net_id=net_id).set_seed(seed, get_global_config('num_threads'), use_seed_seq)
    except:
        Messages._warning('The seed will only be set in the simulated network when it is compiled.')
