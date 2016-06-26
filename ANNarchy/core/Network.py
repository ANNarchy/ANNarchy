from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Record import Monitor
import ANNarchy.core.Global as Global
import ANNarchy.core.Simulate as Simulate
import ANNarchy.generator.Compiler as Compiler

import os, shutil, sys
import numpy as np

class Network(object):
    """
    A network gathers already defined populations, projections and monitors in order to run them independently.

    This is particularly useful when varying single parameters of a network and comparing the results (see the ``parallel_run()`` method).

    Only objects declared before the creation of the network can be used. Global methods such as ``simulate()`` must be used on the network object.
    The objects must be accessed through the ``get()`` method, as the original ones will not be part of the network (a copy is made).

    Each network must be individually compiled, but it does not matter if the original objects were already compiled.

    When passing ``everything=True`` to the constructor, all populations/projections/monitors already defined at the global level will be added to the network.

    If not, you can select which object will be added to network with the ``add()`` method.

    Example with ``everything=True``::

        pop = Population(100, Izhikevich)
        proj = Projection(pop, pop, 'exc')
        proj.connect_all_to_all(1.0)
        m = Monitor(pop, 'spike')

        compile() # Optional

        net = Network(everything=True)
        net.get(pop).a = 0.02
        net.compile()
        net.simulate(1000.)

        net2 = Network(everything=True)
        net2.get(pop).a = 0.05
        net2.compile()
        net2.simulate(1000.)

        t, n = net.get(m).raster_plot()
        t2, n2 = net2.get(m).raster_plot()

    Example with ``everything=False`` (the default)::

        pop = Population(100, Izhikevich)
        proj1 = Projection(pop, pop, 'exc')
        proj1.connect_all_to_all(1.0)
        proj2 = Projection(pop, pop, 'exc')
        proj2.connect_all_to_all(2.0)
        m = Monitor(pop, 'spike')

        net = Network()
        net.add([pop, proj1, m])
        net.compile()
        net.simulate(1000.)

        net2 = Network()
        net2.add([pop, proj2, m])
        net2.compile()
        net2.simulate(1000.)

        t, n = net.get(m).raster_plot()
        t2, n2 = net2.get(m).raster_plot()

    """
    def __init__(self, everything=False):
        """
        *Parameters:*

        * **everything**: defines if all existing populations and projections should be automatically added (default: False).
        """
        self.id = len(Global._network)
        self.everything = everything
        Global._network.append(
            {
            'populations': [],
            'projections': [],
            'monitors': [],
            'instance': None,
            'compiled': False
            }
        )
        Simulate._callbacks.append([])
        Simulate._callbacks_enabled.append(True)
        self.populations = []
        self.projections = []
        self.monitors = []
        if everything:
            self.add(Global._network[0]['populations'])
            self.add(Global._network[0]['projections'])
            self.add(Global._network[0]['monitors'])

    def add(self, objects):
        """
        Adds a Population, Projection or Monitor to the network.

        *Parameters:*

        * **objects**: A single object or a list to add to the network.
        """
        if isinstance(objects, list):
            for item in objects:
                self._add_object(item)
        else:
            self._add_object(objects)

    def _add_object(self, obj):
        if isinstance(obj, Population):
            # Create a copy
            pop = Population(geometry=obj.geometry, neuron=obj.neuron_type, name=obj.name, stop_condition=obj.stop_condition)
            # Remove the copy from the global network
            Global._network[0]['populations'].pop(-1)
            # Copy import properties
            pop.id = obj.id
            pop.name = obj.name
            pop.class_name = obj.class_name
            pop.init = obj.init
            pop.enabled = obj.enabled
            if not obj.enabled: # Also copy the enabled state:
                pop.disable()
            # Add the copy to the local network
            Global._network[self.id]['populations'].append(pop)
            self.populations.append(pop)

        elif isinstance(obj, Projection):
            # Check the pre- or post- populations
            try:
                pre_pop = self.get(obj.pre)
                if isinstance(obj.pre, PopulationView):
                    pre = PopulationView(pre_pop, obj.pre.ranks)
                else:
                    pre = pre_pop
                post_pop = self.get(obj.post)
                if isinstance(obj.post, PopulationView):
                    post = PopulationView(post_pop, obj.post.ranks)
                else:
                    post = post_pop
            except:
                Global._error('Network.add(): The pre- or post-synaptic population of this projection are not in the network.')

            target = obj.target
            synapse = obj.synapse_type
            # Create the projection
            proj = Projection(pre=pre, post=post, target=target, synapse=synapse)
            # Remove the copy from the global network
            Global._network[0]['projections'].pop(-1)
            # Copy import properties
            proj.id = obj.id
            proj.name = obj.name
            proj.init = obj.init
            # Copy the synapses if they are already created
            proj._store_connectivity(obj._connection_method, obj._connection_args, obj._connection_delay)
            # Add the copy to the local network
            Global._network[self.id]['projections'].append(proj)
            self.projections.append(proj)

        elif isinstance(obj, Monitor):
            m = Monitor(obj.object, variables=obj.variables, period=obj._period, start=obj._start, net_id=self.id)
            # Add the copy to the local network (the monitor writes itself already in the right network)
            self.monitors.append(m)


    def get(self, obj):
        """
        Returns the local Population, Projection or Monitor identical to the provided argument.

        *Parameters:*

        * **obj**: A single object or a list of objects.

        *Returns:*

        * The corresponding object or list of objects.

        **Example**::

            pop = Population(100, Izhikevich)
            net = Network()
            net.add(pop)
            net.compile()
            net.simulate(100.)
            print net.get(pop).v
        """
        if isinstance(obj, list):
            return [self._get_object(o) for o in obj]
        else:
            return self._get_object(obj)

    def _get_object(self, obj):
        "Retrieves the corresponding object."
        if isinstance(obj, (Population, PopulationView)):
            for pop in self.populations:
                if pop.id == obj.id:
                    return pop
        elif isinstance(obj, Projection):
            for proj in self.projections:
                if proj.id == obj.id:
                    return proj
        elif isinstance(obj, Monitor):
            for m in self.monitors:
                if m.id == obj.id:
                    return m
        Global._error('The network has no such object:', obj.name, obj)

    def compile(self, directory='annarchy', silent=False):
        """ Compiles the network.

        *Parameters*:

        * **directory**: name of the subdirectory where the code will be generated and compiled.
        * **silent**: defines if the "Compiling... OK" should be printed.
        """
        Compiler.compile(directory=directory, silent=silent, net_id=self.id)

    def simulate(self, duration, measure_time = False):
        """
        Runs the network for the given duration in milliseconds. The number of simulation steps is  computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms)::

            simulate(1000.0)

        *Parameters*:

        * **duration**: the duration in milliseconds.
        * **measure_time**: defines whether the simulation time should be printed (default=False).

        """
        Simulate.simulate(duration, measure_time, net_id=self.id)

    def simulate_until(self, max_duration, population, operator='and', measure_time = False):
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
        return Simulate.simulate_until(max_duration, population, operator, measure_time, net_id=self.id)

    def step(self):
        """
        Performs a single simulation step (duration = ``dt``).
        """
        Simulate.step(self.id)


    def reset(self, populations = True, projections = False, synapses = False):
        """
        Reinitialises the network to its state before the call to compile.

        *Parameters*:

        * **populations**: if True (default), the neural parameters and variables will be reset to their initial value.
        * **projections**: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
        * **synapses**: if True, the synaptic weights will be erased and recreated (default=False).
        """
        Global.reset(populations,  projections, synapses, self.id)

    def get_time(self):
        "Returns the current time in ms."
        return Global.get_time(self.id)

    def set_time(self, t, net_id=0):
        """Sets the current time in ms.

        .. warning::

            Can be dangerous for some spiking models.
        """
        Global.set_time(t, self.id)

    def get_current_step(self):
        "Returns the current simulation step."
        return Global.get_current_step(self.id)

    def set_current_step(self, t):
        """Sets the current simulation step.

        .. warning::

            Can be dangerous for some spiking models.
        """
        Global.set_current_step(t, self.id)

    def set_seed(self, seed):
        """
        Sets the seed of the random number generators for this network.
        """
        Global.set_seed(seed, self.id)

    def enable_learning(self, projections=None):
        """
        Enables learning for all projections.

        *Parameter*:

        * **projections**: the projections whose learning should be enabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self.projections
        for proj in projections:
            proj.enable_learning()

    def disable_learning(self, projections=None):
        """
        Disables learning for all projections.

        *Parameter*:

        * **projections**: the projections whose learning should be disabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self.projections
        for proj in projections:
            proj.disable_learning()

    def get_population(self, name):
        """
        Returns the population with the given *name*.

        *Parameter*:

        * **name**: name of the population

        Returns:

        * The requested ``Population`` object if existing, ``None`` otherwise.
        """
        for pop in self.populations:
            if pop.name == name:
                return pop
        return None

def parallel_run(method, networks=None, number=0, max_processes=-1, measure_time=False, sequential=False, same_seed=False, **args):
    """
    Allows to run multiple networks in parallel using multiprocessing.

    If the ``networks`` argument is provided as a list of Network objects, the given method will be executed for each of these networks.

    If ``number`` is given instead, the same number of networks will be created and the method is applied.

    **Returns**:

    * a list of the values returned by ``method``.

    If ``number`` is used, the created networks are not returned, you should return what you need to analyse.

    *Parameters*:

    * **method**: a Python method which will be executed for each network. This function must accept an integer as first argument (id of the simulation) and a Network object as second argument.
    * **networks**: a list of networks to simulate in parallel.
    * **number**: the number of odentical networks to run in parallel.
    * **max_processes**: maximal number of processes to start concurrently (default: the available number of cores on the machine).
    * **measure_time**: if the total simulation time should be printed out.
    * **sequential**: if True, runs the simulations sequentially instead of in parallel (default: False).
    * **same_seed**: if True, all networks will use the same seed. If not, the seed will be randomly initialized with time(0) for each network (default).
    * **args**: other named arguments you want to pass to the simulation method.

    *Example:*::

        pop1 = PoissonPopulation(100, rates=10.0)
        pop2 = Population(100, Izhikevich)
        proj = Projection(pop1, pop2, 'exc')
        proj.connect_fixed_probability(weights=5.0, probability=0.2)
        m = Monitor(pop2, 'spike')

        compile()

        def simulation(idx, net):
            net.get(pop1).rates = 10. * idx
            net.simulate(1000.)
            return net.get(m).raster_plot()

        results = parallel_run(method=simulation, number = 3)

        t1, n1 = results[0]
        t2, n2 = results[1]
        t3, n3 = results[2]

    """
    # Check inputs
    if not networks and number < 1:
        Global._error('parallel_run(): the networks or number arguments must be set.', exit=True)

    import types
    if not isinstance(method, types.FunctionType):
        Global._error('parallel_run(): the method argument must be a method.', exit=True)

    if not networks: # The magic network will run N times
        return _parallel_multi(method, number, max_processes, measure_time, sequential, same_seed, args)

    if not isinstance(networks, list):
        Global._error('parallel_run(): the networks argument must be a list.', exit=True)

    # Simulate the different networks
    return _parallel_networks(method, networks, max_processes, measure_time, sequential, same_seed, args)

def _parallel_networks(method, networks, max_processes, measure_time, sequential, same_seed, args):
    " Method when different networks are provided"
    import multiprocessing
    from multiprocessing.dummy import Pool

    # Time measurement
    from time import time
    if measure_time:
        ts = time()

    # Number of processes to create
    if max_processes < 0:
        max_processes = min(len(networks), multiprocessing.cpu_count())

    # Number of networks
    number = len(networks)

    # Build arguments list
    arguments = [[method, n, networks[n]] for n in range(number)]
    if len(args) != method.__code__.co_argcount-2:  # idx, net are default
        Global._error('the method', method.__name__, 'takes', method.__code__.co_argcount-2,
                      'arguments (in addition to idx and net) which have to be passed to parallel_run:', method.__code__.co_varnames[2:method.__code__.co_argcount])
    for arg in range(2, method.__code__.co_argcount):
        varname = method.__code__.co_varnames[arg]
        data = args[varname]
        if not len(data) == number:
            Global._error('parallel_run(): the argument', varname, 'must be a list of values for each of the', number, 'networks.')
        for n in range(number):
            arguments[n].append(data[n])

    # Simulation
    if not sequential:
        pool = Pool(max_processes)
        try:
            results = pool.map(_only_run_method, arguments)
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running multiple networks failed.', exit=True)
        pool.close()
        pool.join()
    else:
        results = []
        for idx, net in enumerate(networks):
            try:
                results.append(method(*arguments[idx][1:]))
            except Exception as e:
                Global._print(e)
                Global._error('parallel_run(): running network ' + str(net.id) + ' failed.', exit=True)

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(len(networks)) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Global._print(msg)

    return results


def _parallel_multi(method, number, max_processes, measure_time, sequential, same_seed, args):
    "Method when the same network must be simulated multiple times."
    import multiprocessing
    from multiprocessing import Pool

    # Time measurement
    from time import time
    if measure_time:
        ts = time()

    # Make sure the magic network is compiled
    if not Global._network[0]['compiled']:
        Global._warning('parallel_run(): the magic network is not compiled yet, doing it...')
        Compiler.compile()

    # Number of processes to create
    if max_processes < 0:
        max_processes = min(number, multiprocessing.cpu_count())

    # Seed
    if same_seed and Global.config['seed'] > -1: # use the global seed
        seed =  Global.config['seed']
    elif same_seed and Global.config['seed'] == -1: # not defined, but should be the same for all networks
        seed = np.random.get_state()[1][0] # not the current seed, but close enough...
    else: # draw it everytime with time(0)
        seed = -1

    # Build arguments list
    arguments = [[n, method] for n in range(number)]
    if len(args) != method.__code__.co_argcount-2:  # idx, net are default
        Global._error('the method', method.__name__, 'takes', method.__code__.co_argcount-2,
                      'arguments (in addition to idx and net) which have to be passed to parallel_run:', method.__code__.co_varnames[2:method.__code__.co_argcount])
    for arg in range(2, method.__code__.co_argcount):
        varname = method.__code__.co_varnames[arg]
        data = args[varname]
        if not len(data) == number:
            Global._error('parallel_run(): the argument', varname, 'must be a list of values for each of the', number, 'networks.')
        for n in range(number):
            arguments[n].append(data[n])
    for n in range(number): # Add the seed at the end
        arguments[n].append(seed)


    # Simulation
    if not sequential:
        try:
            pool = Pool(max_processes)
            results = pool.map(_create_and_run_method, arguments)
            pool.close()
            pool.join()
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running ' + str(number) + ' networks failed.', exit=True)
    else:
        results = []
        try:
            for n in range(number):
                results.append(_create_and_run_method(arguments))
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running ' + str(number) + ' networks failed.', exit=True)

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(number) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Global._print(msg)

    return results

def _create_and_run_method(args):
    "method called to wrap the user-defined method"
    n = args[0]
    method = args[1]
    seed = args[-1]
    net = Network(True)
    Compiler._instantiate(net.id, 0)
    if seed == -1:
        net.set_seed(seed)
        np.random.seed() # Would set -1 otherwise
    else:
        net.set_seed(seed)
    # Create the arguments
    arguments = args[:-1] # all arguments except seed
    arguments[1] = net # replace the second argument method with net
    # Call the method
    res = method(*arguments)
    del net
    return res

def _only_run_method(args):
    "method called to wrap the user-defined method"
    method = args[0]
    arguments = args[1:]
    res = method(*arguments)
    return res
