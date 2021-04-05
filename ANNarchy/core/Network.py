#===============================================================================
#
#     Network.py
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
from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Monitor import Monitor
from ANNarchy.extensions.bold import BoldMonitor

import ANNarchy.core.Global as Global
import ANNarchy.core.Simulate as Simulate
import ANNarchy.core.IO as IO
import ANNarchy.core.SpecificPopulation as SpecificPopulation
import ANNarchy.generator.Compiler as Compiler
import numpy as np
import os

class Network(object):
    """
    A network gathers already defined populations, projections and monitors in order to run them independently.

    This is particularly useful when varying single parameters of a network and comparing the results (see the ``parallel_run()`` method).

    Only objects declared before the creation of the network can be used. Global methods such as ``simulate()`` must be used on the network object.
    The objects must be accessed through the ``get()`` method, as the original ones will not be part of the network (a copy is made).

    Each network must be individually compiled, but it does not matter if the original objects were already compiled.

    When passing ``everything=True`` to the constructor, all populations/projections/monitors already defined at the global level will be added to the network.

    If not, you can select which object will be added to network with the ``add()`` method.

    Example with ``everything=True``:

    ```python
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
    ```

    Example with ``everything=False`` (the default):

    ```python
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
    ```

    """
    def __init__(self, everything=False):
        """
        :param everything: defines if all existing populations and projections should be automatically added (default: False).
        """
        self.id = Global._network.add_network(self)
        self.everything = everything

        Simulate._callbacks.append([])
        Simulate._callbacks_enabled.append(True)
        self.populations = []
        self.projections = []
        self.monitors = []

        if everything:
            self.add(Global._network[0]['populations'])
            self.add(Global._network[0]['projections'])
            self.add(Global._network[0]['monitors'])

    def __del__(self):
        
        # Overridden destructor for two reasons:
        # 
        # a) track destruction of objects
        # b) manually deallocate C++ container data
        # 
        # Hint: this function can be called explicitly (which is not recommended in many cases) or as
        #       finalizer from the garbage collection. If called explicitely, one should take in mind,
        #       that the function will be called twice. The better approach is to trigger this function
        #       by del on the network object.
        
        for pop in self.get_populations():
            pop._clear()
            del pop

        for proj in self.get_projections(suppress_error=True):
            proj._clear()
            del proj

        for mon in self.monitors:
            mon._clear()
            del mon

        Global._network._remove_network(self)

    def _cpp_memory_footprint(self):
        """
        Print the C++ memory consumption for populations, projections on the console.
        """
        for pop in self.get_populations():
            print(pop.name, pop.size_in_bytes())

        for proj in self.get_projections():
            print(proj.name, proj.size_in_bytes())

        for mon in self.monitors:
            print(type(mon), mon.size_in_bytes())

    def add(self, objects):
        """
        Adds a Population, Projection or Monitor to the network.

        :param objects: A single object or a list to add to the network.
        """
        if isinstance(objects, list):
            for item in objects:
                self._add_object(item)
        else:
            self._add_object(objects)

    def _add_object(self, obj):
        """
        Add the object *obj* to the network.

        TODO: instead of creating copies by object construction, one should check if deepcopy works ...
        """
        if isinstance(obj, Population):
            # Create a copy
            pop = obj._copy()

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
            proj = obj._copy(pre=pre, post=post)
            # Remove the copy from the global network
            Global._network[0]['projections'].pop(-1)

            # Copy import properties
            proj.id = obj.id
            proj.name = obj.name
            proj.init = obj.init

            # Copy the synapses if they are already created
            proj._store_connectivity(obj._connection_method, obj._connection_args, obj._connection_delay, obj._storage_format)

            # Add the copy to the local network
            Global._network[self.id]['projections'].append(proj)
            self.projections.append(proj)

        elif isinstance(obj, BoldMonitor):

            # Create a copy of the monitor
            m = BoldMonitor(obj.object,  variables=obj.variables, epsilon=obj._epsilon, alpha=obj._alpha, kappa=obj._kappa, gamma=obj._gamma, E_0=obj._E_0, V_0=obj._V_0, tau_0=obj._tau_0, record_all_variables=obj._record_all_variables, period=obj._period, start=obj._start, net_id=self.id)

            # there is a bad mismatch between object ids:
            #
            # m.id     is dependent on len(_network[net_id].monitors)
            # obj.id   is dependent on len(_network[0].monitors)
            m.id = obj.id # TODO: check this !!!!

            # Add the copy to the local network (the monitor writes itself already in the right network)
            self.monitors.append(m)

        elif isinstance(obj, Monitor):
            # Get the copied reference of the object monitored
            # try:
            #     obj_copy = self.get(obj.object)
            # except:
            #     Global._error('Network.add(): The monitor does not exist.')

            # Stop the master monitor, otherwise it gets data.
            for var in obj.variables:
                try:
                    setattr(obj.cyInstance, 'record_'+var, False)
                except:
                    pass
            # Create a copy of the monitor
            m = Monitor(obj.object, variables=obj.variables, period=obj._period, start=obj._start, net_id=self.id)

            # there is a bad mismatch between object ids:
            #
            # m.id     is dependent on len(_network[net_id].monitors)
            # obj.id   is dependent on len(_network[0].monitors)
            m.id = obj.id # TODO: check this !!!!

            # Add the copy to the local network (the monitor writes itself already in the right network)
            self.monitors.append(m)

    def get(self, obj):
        """
        Returns the local Population, Projection or Monitor identical to the provided argument.

        Example:

        ```python
        pop = Population(100, Izhikevich)
        net = Network()
        net.add(pop)
        net.compile()
        net.simulate(100.)
        print net.get(pop).v
        ```

        :param obj: A single object or a list of objects.
        :return: The corresponding object or list of objects.
        """
        if isinstance(obj, list):
            return [self._get_object(o) for o in obj]
        else:
            return self._get_object(obj)

    def _get_object(self, obj):
        "Retrieves the corresponding object."
        if isinstance(obj, Population):
            for pop in self.populations:
                if pop.id == obj.id:
                    return pop
        elif isinstance(obj, PopulationView):
            for pop in self.populations:
                if pop.id == obj.id:
                    return PopulationView(pop, obj.ranks) # Create on the fly?
        elif isinstance(obj, Projection):
            for proj in self.projections:
                if proj.id == obj.id:
                    return proj
        elif isinstance(obj, Monitor):
            for m in self.monitors:
                if m.id == obj.id:
                    return m
        Global._error('The network has no such object:', obj.name, obj)

    def compile(self,
                directory='annarchy',
                clean=False,
                compiler="default",
                compiler_flags="-march=native -O2",
                cuda_config=None,
                annarchy_json="",
                silent=False):
        """
        Compiles the network.

        :param directory: name of the subdirectory where the code will be generated and compiled. Must be a relative path. Default: "annarchy/".
        :param clean: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
        :param compiler: C++ compiler to use. Default: g++ on GNU/Linux, clang++ on OS X. Valid compilers are [g++, clang++].
        :param compiler_flags: platform-specific flags to pass to the compiler. Default: "-march=native -O2". Warning: -O3 often generates slower code and can cause linking problems, so it is not recommended.
        :param cuda_config: dictionary defining the CUDA configuration for each population and projection.
        :param annarchy_json: compiler flags etc are stored in a .json file normally placed in the home directory. With this flag one can directly assign a file location.
        :param silent: defines if the "Compiling... OK" should be printed.

        """
        Compiler.compile(directory=directory, silent=silent, clean=clean, compiler=compiler, compiler_flags=compiler_flags, cuda_config=cuda_config, annarchy_json=annarchy_json, net_id=self.id)

    def simulate(self, duration, measure_time = False):
        """
        Runs the network for the given duration in milliseconds. 
        
        The number of simulation steps is  computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms):

        ```python
        simulate(1000.0)
        ```

        :param duration: the duration in milliseconds.
        :param measure_time: defines whether the simulation time should be printed (default=False).

        """
        Simulate.simulate(duration, measure_time, net_id=self.id)

    def simulate_until(self, max_duration, population, operator='and', measure_time = False):
        """
        Runs the network for the maximal duration in milliseconds. If the ``stop_condition`` defined in the population becomes true during the simulation, it is stopped.

        One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

        Example:

        ```python
        pop1 = Population( ..., stop_condition = "r > 1.0 : any")
        compile()
        simulate_until(max_duration=1000.0. population=pop1)
        ```

        :param max_duration: the maximum duration of the simulation in milliseconds.
        :param population: the (list of) population whose ``stop_condition`` should be checked to stop the simulation.
        :param operator: operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
        :param measure_time: defines whether the simulation time should be printed (default=False).
        :return: the actual duration of the simulation in milliseconds.
        """
        return Simulate.simulate_until(max_duration, population, operator, measure_time, net_id=self.id)

    def step(self):
        """
        Performs a single simulation step (duration = ``dt``).
        """
        Simulate.step(self.id)

    def reset(self, populations=True, projections=False, synapses=False):
        """
        Reinitialises the network to its state before the call to compile.

        :param populations: if True (default), the neural parameters and variables will be reset to their initial value.
        :param projections: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
        :param synapses: if True, the synaptic weights will be erased and recreated (default=False).
        """
        Global.reset(populations,  projections, synapses, self.id)

    def get_time(self):
        "Returns the current time in ms."
        return Global.get_time(self.id)

    def set_time(self, t, net_id=0):
        """
        Sets the current time in ms.

        **Warning:** can be dangerous for some spiking models.
        """
        Global.set_time(t, self.id)

    def get_current_step(self):
        "Returns the current simulation step."
        return Global.get_current_step(self.id)

    def set_current_step(self, t):
        """
        Sets the current simulation step.

        **Warning:** can be dangerous for some spiking models.
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

        :param projections: the projections whose learning should be enabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self.projections
        for proj in projections:
            proj.enable_learning()

    def disable_learning(self, projections=None):
        """
        Disables learning for all projections.

        :param projections: the projections whose learning should be disabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self.projections
        for proj in projections:
            proj.disable_learning()

    def get_population(self, name):
        """
        Returns the population with the given *name*.

        :param name: name of the population
        :return: The requested ``Population`` object if existing, ``None`` otherwise.
        """
        for pop in self.populations:
            if pop.name == name:
                return pop
        Global._print('get_population(): the population', name, 'does not exist in this network.')
        return None

    def get_projection(self, name):
        """
        Returns the projection with the given *name*.

        :param name: name of the projection
        :return: The requested ``Projection`` object if existing, ``None`` otherwise.
        """
        for proj in self.projections:
            if proj.name == name:
                return proj
        Global._print('get_projection(): the projection', name, 'does not exist in this network.')
        return None

    def get_populations(self):
        """
        Returns a list of all declared populations in this network.
        """
        if self.populations == []:
            Global._warning("Network.get_populations(): no populations attached to this network.")
        return self.populations

    def get_projections(self, post=None, pre=None, target=None, suppress_error=False):
        """
        Get a list of declared projections for the current network. By default,
        the method returns all connections within the network.

        By setting the arguments, post, pre and target one can select a subset.

        :param post: all returned projections should have this population as post.
        :param pre: all returned projections should have this population as pre.
        :param target: all returned projections should have this target.
        :param suppress_error: by default, ANNarchy throws an error if the list of assigned projections is empty. If this flag is set to True, the error message is suppressed.
        :return: A list of all assigned projections in this network or a subset according to the arguments.

        """
        if self.projections == []:
            if not suppress_error:
                Global._error("Network.get_projections(): no projections attached to this network.")

        if post is None and pre is None and target is None:
            return self.projections
        else:
            res = []
            if isinstance(post, str):
                post = self.get_population(post)
            if isinstance(pre, str):
                pre = self.get_population(pre)

            for proj in self.projections:
                if post is not None:
                    # post is exclusionary
                    if proj.post == post:
                        res.append(proj)
                
                if pre is not None:
                    raise NotImplementedError

                if target is not None:
                    raise NotImplementedError

            return res

    def load(self, filename, populations=True, projections=True):
        """
        Loads a saved state of the current network by calling ANNarchy.core.IO.load().

        :param filename: filename, may contain relative or absolute path.
        :param populations: if True, population data will be saved (by default True)
        :param projections: if True, projection data will be saved (by default True)
        """
        IO.load(filename, populations, projections, self.id)

    def save(self, filename, populations=True, projections=True):
        """
        Saves the current network by calling ANNarchy.core.IO.save().

        :param filename: filename, may contain relative or absolute path.
        :param populations: if True, population data will be saved (by default True)
        :param projections: if True, projection data will be saved (by default True)
        """
        IO.save(filename, populations, projections, self.id)

def parallel_run(method, networks=None, number=0, max_processes=-1, measure_time=False, sequential=False, same_seed=False, **args):
    """
    Allows to run multiple networks in parallel using multiprocessing.

    If the ``networks`` argument is provided as a list of Network objects, the given method will be executed for each of these networks.

    If ``number`` is given instead, the same number of networks will be created and the method is applied.

    If ``number`` is used, the created networks are not returned, you should return what you need to analyse.

    Example:

    ```python
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
    ```


    :param method: a Python method which will be executed for each network. This function must accept an integer as first argument (id of the simulation) and a Network object as second argument.
    :param networks: a list of networks to simulate in parallel.
    :param number: the number of odentical networks to run in parallel.
    :param max_processes: maximal number of processes to start concurrently (default: the available number of cores on the machine).
    :param measure_time: if the total simulation time should be printed out.
    :param sequential: if True, runs the simulations sequentially instead of in parallel (default: False).
    :param same_seed: if True, all networks will use the same seed. If not, the seed will be randomly initialized with time(0) for each network (default). It has no influence when the ``networks`` argument is set (the seed has to be set individually for each network using ``net.set_seed()``), only when ``number`` is used.
    :param args: other named arguments you want to pass to the simulation method.
    :return: a list of the values returned by ``method``.

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
    return _parallel_networks(method, networks, max_processes, measure_time, sequential, args)


def _parallel_networks(method, networks, max_processes, measure_time, sequential, args):
    " Method when different networks are provided"
    import multiprocessing
    from multiprocessing.dummy import Pool

    # Time measurement
    from time import time
    if measure_time:
        ts = time()

    # Number of processes to create depends on number of
    # available CPUs or GPUs
    if max_processes < 0:
        if Global.config['paradigm'] == "openmp":
            max_processes = min(len(networks), multiprocessing.cpu_count())
        elif Global.config['paradigm'] == "cuda":
            from ANNarchy.generator.CudaCheck import CudaCheck
            max_processes = min(len(networks), CudaCheck().gpu_count())
        else:
            raise NotImplementedError

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
        Global._warning('parallel_run(): the network is not compiled yet, doing it now...')
        Compiler.compile()

    # Number of processes to create
    if max_processes < 0:
        max_processes = min(number, multiprocessing.cpu_count())

    # Seed
    if same_seed and Global.config['seed'] > -1: # use the global seed
        seed =  Global.config['seed']
    else: # draw it everytime with time(0)
        seed = np.random.get_state()[1][0]

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
    for n in range(number): # Add the seed at the end. Increment the seed if the seeds should be different
        arguments[n].append(seed + n if not same_seed else 0)


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
    """
    Method called to wrap the user-defined method when different networks are created.
    """
    # Get arguments
    n = args[0]
    method = args[1]
    seed = args[-1]
    # Create and instantiate the network 0, not compile it!
    net = Network(True)
    Compiler._instantiate(net_id=net.id, import_id=0)
    # Set the seed
    net.set_seed(seed)
    # Create the arguments
    arguments = args[:-1] # all arguments except seed
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
