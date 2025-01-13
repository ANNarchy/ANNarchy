"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from .Network import Network
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages

import ANNarchy.core.Global as Global
import ANNarchy.generator.Compiler as Compiler
import numpy as np

def parallel_run(
        method, 
        networks:list=None, 
        number:int=0, 
        max_processes:int=-1, 
        measure_time:bool=False, 
        sequential:bool=False, 
        same_seed:bool=False, 
        annarchy_json:str="", 
        visible_cores:list=[], 
        **args) -> list:
    """
    Allows to run multiple networks in parallel using multiprocessing.

    If the ``networks`` argument is provided as a list of Network objects, the given method will be executed for each of these networks.

    If ``number`` is given instead, the same number of networks will be created and the method is applied.

    If ``number`` is used, the created networks are not returned, you should return what you need to analyse.

    Example:

    ```python
    pop1 = ann.PoissonPopulation(100, rates=10.0)
    pop2 = ann.Population(100, ann.Izhikevich)
    proj = ann.Projection(pop1, pop2, 'exc')
    proj.connect_fixed_probability(weights=5.0, probability=0.2)
    m = ann.Monitor(pop2, 'spike')

    ann.compile()

    def simulation(idx, net):
        net.get(pop1).rates = 10. * idx
        net.simulate(1000.)
        return net.get(m).raster_plot()

    results = ann.parallel_run(method=simulation, number = 3)

    t1, n1 = results[0]
    t2, n2 = results[1]
    t3, n3 = results[2]
    ```


    :param method: a Python method which will be executed for each network. This function must accept an integer as first argument (id of the simulation) and a Network object as second argument.
    :param networks: a list of networks to simulate in parallel.
    :param number: the number of identical networks to run in parallel.
    :param max_processes: maximal number of processes to start concurrently (default: the available number of cores on the machine).
    :param measure_time: if the total simulation time should be printed out.
    :param sequential: if True, runs the simulations sequentially instead of in parallel (default: False).
    :param same_seed: if True, all networks will use the same seed. If not, the seed will be randomly initialized with time(0) for each network (default). It has no influence when the ``networks`` argument is set (the seed has to be set individually for each network using ``net.set_seed()``), only when ``number`` is used.
    :param annarchy_json: path to a different configuration file if needed (default "").
    :param visible_cores: a list of CPU core ids to simulate on (must have max_processes entries and max_processes must be != -1)
    :param args: other named arguments you want to pass to the simulation method.
    :returns: a list of the values returned by each call to `method`.

    """
    # Check inputs
    if not networks and number < 1:
        Messages._error('parallel_run(): the networks or number arguments must be set.', exit=True)

    if len(visible_cores) > 0 and max_processes == -1:
        Messages._error('parallel_run(): when using visible cores the number of max_processes must be set.', exit=True)

    if (len(visible_cores) > 0) and (len(visible_cores) != max_processes):
        Messages._error('parallel_run(): the number of entries in visible_cores must be equal to max_processes.', exit=True)

    import types
    if not isinstance(method, types.FunctionType):
        Messages._error('parallel_run(): the method argument must be a method.', exit=True)

    if not networks: # The magic network will run N times
        return _parallel_multi(method, number, max_processes, measure_time, sequential, same_seed, annarchy_json, visible_cores, args)

    if not isinstance(networks, list):
        Messages._error('parallel_run(): the networks argument must be a list.', exit=True)

    # Simulate the different networks
    return _parallel_networks(method, networks, max_processes, measure_time, sequential, args)


def _parallel_networks(method, networks, max_processes, measure_time, sequential, args):
    " Method when different networks are provided"
    import multiprocessing
    from multiprocessing import Pool

    # Time measurement
    from time import time
    if measure_time:
        ts = time()

    # Number of processes to create depends on number of
    # available CPUs or GPUs
    if max_processes < 0:
        if get_global_config('paradigm') == "openmp":
            max_processes = min(len(networks), multiprocessing.cpu_count())
        elif get_global_config('paradigm') == "cuda":
            Messages._warning("In the present ANNarchy version the usage of parallel networks and multi-GPUs is disabled.")
            max_processes = 1
        else:
            raise NotImplementedError

    # Number of networks
    number = len(networks)

    # Build arguments list
    arguments = [[method, n, networks[n]] for n in range(number)]
    if len(args) != method.__code__.co_argcount-2:  # idx, net are default
        Messages._error('the method', method.__name__, 'takes', method.__code__.co_argcount-2,
                      'arguments (in addition to idx and net) which have to be passed to parallel_run:', method.__code__.co_varnames[2:method.__code__.co_argcount])
    for arg in range(2, method.__code__.co_argcount):
        varname = method.__code__.co_varnames[arg]
        data = args[varname]
        if not len(data) == number:
            Messages._error('parallel_run(): the argument', varname, 'must be a list of values for each of the', number, 'networks.')
        for n in range(number):
            arguments[n].append(data[n])

    # Simulation
    if not sequential:
        pool = Pool(max_processes)
        try:
            results = pool.map(_only_run_method, arguments)
        except Exception as e:
            Messages._print(e)
            Messages._error('parallel_run(): running multiple networks failed.', exit=True)
        pool.close()
        pool.join()
    else:
        results = []
        for idx, net in enumerate(networks):
            try:
                results.append(method(*arguments[idx][1:]))
            except Exception as e:
                Messages._print(e)
                Messages._error('parallel_run(): running network ' + str(net.id) + ' failed.', exit=True)

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(len(networks)) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Messages._print(msg)

    return results


def _parallel_multi(method, number, max_processes, measure_time, sequential, same_seed, annarchy_json, visible_cores, args):
    "Method when the same network must be simulated multiple times."
    import multiprocessing
    from multiprocessing import Pool

    # Time measurement
    from time import time
    if measure_time:
        ts = time()

    # Make sure the magic network is compiled
    if not NetworkManager().is_compiled(net_id=0):
        Messages._warning('parallel_run(): the network is not compiled yet, doing it now...')
        Compiler.compile(annarchy_json=annarchy_json)

    # Number of processes to create
    if max_processes < 0:
        if get_global_config('paradigm') == "openmp":
            max_processes = min(number, multiprocessing.cpu_count())
        elif get_global_config('paradigm') == "cuda":
            Messages._warning("In the present ANNarchy version the usage of parallel networks and multi-GPUs is disabled.")
            max_processes = 1
        else:
            raise NotImplementedError

    # Seed
    if same_seed and get_global_config('seed') is not None: # use the global seed
        seed = get_global_config('seed')
    else: # draw it everytime with time(0)
        seed = np.random.get_state()[1][0] # old api < 1.17

    # Build arguments list for each instance with the following structure:
    # [ net_id, arguments for method, seed ]
    arguments = [[n, method] for n in range(number)]
    if len(args) != method.__code__.co_argcount-2:  # idx, net are default
        Messages._error('the method', method.__name__, 'takes', method.__code__.co_argcount-2,
                      'arguments (in addition to idx and net) which have to be passed to parallel_run:', method.__code__.co_varnames[2:method.__code__.co_argcount])
    for arg in range(2, method.__code__.co_argcount):
        varname = method.__code__.co_varnames[arg]
        data = args[varname]
        if not len(data) == number:
            Messages._error('parallel_run(): the argument', varname, 'must be a list of values for each of the', number, 'networks.')
        for n in range(number):
            arguments[n].append(data[n])
    for n in range(number): # Add the seed at the end. Increment the seed if the seeds should be different
        arguments[n].append(seed + n if not same_seed else 0)

    # Thread placement is optional
    if len(visible_cores) == 0:
        for n in range(number):
            arguments[n].append([])
    else:
        for n in range(number):
            arguments[n].append([visible_cores[np.mod(n,max_processes)]])

    # Simulation
    if not sequential and len(visible_cores) == 0:
        try:
            pool = Pool(max_processes)
            results = pool.map(_create_and_run_method, arguments)
            pool.close()
            pool.join()
        except Exception as e:
            Messages._print(e)
            Messages._error('parallel_run(): running ' + str(number) + ' networks failed.', exit=True)

    elif not sequential and len(visible_cores) > 0:
        # Thread placement requires some more fine-grained control
        # on the execution
        try:
            n_iter = int(np.ceil(number / max_processes))
            pool = Pool(max_processes)
            for idx in range(n_iter):
                beg = int(idx * max_processes)
                end = int(min((idx+1) * max_processes, number))
                results = pool.map(_create_and_run_method, arguments[beg:end])
            pool.close()
            pool.join()
        except Exception as e:
            Messages._print(e)
            Messages._error('parallel_run(): running ' + str(number) + ' networks failed.', exit=True)

    else:
        results = []
        try:
            for n in range(number):
                results.append(_create_and_run_method(arguments[0]))
        except Exception as e:
            Messages._print(e)
            Messages._error('parallel_run(): running ' + str(number) + ' networks failed.', exit=True)

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(number) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Messages._print(msg)

    return results


def _create_and_run_method(args):
    """
    Method called to wrap the user-defined method when different networks are created.
    """
    # Get arguments
    n = args[0]
    method = args[1]
    visible_cores = args[-1]
    seed = args[-2]
    # Create and instantiate the network 0, not compile it!
    net = Network(True)
    Compiler._instantiate(net_id=net.id, import_id=0, core_list=visible_cores)
    # Set the seed
    net.set_seed(seed)
    # Create the arguments
    arguments = args[:-2] # all arguments except seed and visible_cores
    arguments[1] = net # replace the second argument method with net
    # Call the method
    res = method(*arguments)
    del net
    return res


def _only_run_method(args):
    """
    Method called to wrap the user-defined method when a single network is already instantiated.
    """
    method = args[0]
    arguments = args[1:]
    res = method(*arguments)
    return res
