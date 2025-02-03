"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import copy

from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Monitor import Monitor
from .Neuron import Neuron
from .Synapse import Synapse

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages
from ANNarchy.extensions.bold import BoldMonitor, BoldModel, balloon_RN


import ANNarchy.core.Global as Global
import ANNarchy.core.Simulate as Simulate
import ANNarchy.core.IO as IO
import ANNarchy.generator.Compiler as Compiler

# Meta class to avoid forcing the constructor 
class NetworkMeta(type):
    def __call__(cls, *args, **kwargs):

        # Create an instance without calling __init__
        instance = cls.__new__(cls, *args, **kwargs)

        # Call the parent class's __init__ methods first
        if isinstance(instance, Network):
            Network.__init__(instance, *args, **kwargs)

        # Call the child's __init__ method
        if hasattr(cls, '__init__'):
            cls.__init__(instance, *args, **kwargs)

        return instance

class Network (metaclass=NetworkMeta):
    """
    A network gathers already defined populations, projections and monitors in order to run them independently.

    This is particularly useful when varying single parameters of a network and comparing the results (see the `parallel_run()` method).

    Only objects declared before the creation of the network can be used. Global methods such as `simulate()` must be used on the network object.
    The objects must be accessed through the `get()` method, as the original ones will not be part of the network (a copy is made).

    Each network must be individually compiled, but it does not matter if the original objects were already compiled.

    When passing `everything=True` to the constructor, all populations/projections/monitors already defined at the global level will be added to the network.

    If not, you can select which object will be added to network with the ``add()`` method.

    Example with ``everything=True``:

    ```python
    pop = ann.Population(100, Izhikevich)
    proj = ann.Projection(pop, pop, 'exc')
    proj.connect_all_to_all(1.0)
    m = ann.Monitor(pop, 'spike')

    ann.compile() # Optional

    net = ann.Network(everything=True)
    net.get(pop).a = 0.02
    net.compile()
    net.simulate(1000.)

    net2 = ann.Network(everything=True)
    net2.get(pop).a = 0.05
    net2.compile()
    net2.simulate(1000.)

    t, n = net.get(m).raster_plot()
    t2, n2 = net2.get(m).raster_plot()
    ```

    Example with ``everything=False`` (the default):

    ```python
    pop = ann.Population(100, Izhikevich)
    proj1 = ann.Projection(pop, pop, 'exc')
    proj1.connect_all_to_all(1.0)
    proj2 = ann.Projection(pop, pop, 'exc')
    proj2.connect_all_to_all(2.0)
    m = ann.Monitor(pop, 'spike')

    net = ann.Network()
    net.add([pop, proj1, m])
    net.compile()
    net.simulate(1000.)

    net2 = ann.Network()
    net2.add([pop, proj2, m])
    net2.compile()
    net2.simulate(1000.)

    t, n = net.get(m).raster_plot()
    t2, n2 = net2.get(m).raster_plot()
    ```
    
    :param everything: defines if all existing populations and projections of the magic network should be automatically added (default: False).   
    """

    # Data
    _populations = []
    _projections = []
    _monitors = []
    _extensions = []

    def __init__(self, everything:bool=False, *args, **kwargs):

        # Constructor should only be called once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Store the arguments for parallel_run
        self._arguments_dict = {}
        for index, value in enumerate(args):
            print(index, value)
            self._arguments_dict[f'arg{index}'] = value
        self._arguments_dict.update(kwargs)


        # Register the network
        self.id = NetworkManager().add_network(self)

        # Callbacks
        Simulate._callbacks.append([])
        Simulate._callbacks_enabled.append(True)
        
        # Get all objects of the magic network
        if everything:
            self.add(NetworkManager().get_populations(net_id=0))
            self.add(NetworkManager().get_projections(net_id=0))
            self.add(NetworkManager().get_monitors(net_id=0))
            self.add(NetworkManager().get_extensions(net_id=0))

    def __del__(self):
        
        # Overridden destructor for two reasons:
        # 
        # a) track destruction of objects
        # b) manually deallocate C++ container data
        # 
        # Hint: this function can be called explicitly (which is not recommended in many cases) or as
        #       finalizer from the garbage collection. If called explicitely, one should take in mind,
        #       that the function will be called twice. The better approach is to trigger this function
        #       by del on the network object

        for pop in self._populations:
            pop._clear()
            del pop

        for proj in self.get_projections(suppress_error=True):
            proj._clear()
            del proj

        for mon in self._monitors:
            mon._clear()
            del mon

        for ext in self._extensions:
            ext._clear()
            del ext

        NetworkManager()._remove_network(self)

    def clear(self):
        """
        Empties the network to prevent a memory leak until the grabage collector wakes up.
        """
        for pop in self._populations:
            pop._clear()

        for proj in self.get_projections(suppress_error=True):
            proj._clear()

        for mon in self._monitors:
            mon._clear()

        for ext in self._extensions:
            ext._clear()

        NetworkManager()._network_desc[self.id].instance = None

    def create(
            self,
            geometry: tuple | int, 
            neuron: Neuron = None, 
            stop_condition:str = None, 
            name:str = None,
            population: Population = None,
            # Internal use only
            storage_order:str = 'post_to_pre', 
            ) -> Population:
        """
        Adds a population of neurons to the network.

        TODO
        """
        if isinstance(geometry, Population): # trick if one does use population=
            pop = geometry._copy(self.id)
        elif population is not None:
            if not isinstance(population, Population):
                Messages._error("Network.create(population=pop) only accepts instances of ann.Population and its subclasses.")
            # Population is already created
            pop = population._copy(self.id)
        else:
            # Create the population
            pop = Population(
                geometry=geometry, 
                neuron=neuron, 
                name=name, 
                stop_condition=stop_condition,
                storage_order=storage_order, 
                copied=False, 
                net_id=self.id
            )
        
        # Add the population to the list
        self._populations.append(pop)

        return pop
    
    def connect(
            self,
            pre: str | Population, 
            post: str | Population = None, 
            target: str = "", 
            synapse: Synapse = None, 
            name:str = None, 
            projection:Projection = None,
            # Internal
            disable_omp:bool = True, 
        ) -> "Projection":

        # Check the pre- or post- populations, they must be in the same network
        # TODO
        
        # Create the projection
        if isinstance(pre, Projection): # trick if one does not use projection=
            proj = pre._copy(pre.pre, pre.post, self.id)
        elif projection is not None:
            if not isinstance(projection, Projection):
                Messages._error("Network.connect(projection=proj) only accepts instances of ann.Projection and its subclasses.")
            # Population is already created
            proj = projection ._copy(projection.pre, projection.post, self.id)
        else:
            if post is None:
                Messages._error("Network.connect(): the post population must be provided.")
            if target == "":
                Messages._error("Network.connect(): the target must be specified.")

            proj = Projection(
                pre = pre, 
                post = post, 
                target = target, 
                synapse = synapse, 
                name = name, 
                # Internal
                disable_omp = disable_omp, 
                copied = False,
                net_id = self.id,
            )
        
        # Add the projection to the list
        self._projections.append(proj)
        
        return proj
    
    def monitor(
            self,
            obj: Population | PopulationView | Projection, 
            variables:list=[], 
            period:float=None, 
            period_offset:float=None, 
            start:bool=True, 
            ) -> Monitor:

        monitor = Monitor(
            obj=obj, variables=variables, period=period, period_offset=period_offset, start=start, net_id=self.id)
        
        self._monitors.append(monitor)

        return monitor
    
    def boldmonitor(
            self,
            populations: list=None,
            bold_model: BoldModel = balloon_RN,
            mapping: dict={'I_CBF': 'r'},
            scale_factor: list[float]=None,
            normalize_input: list[int]=None,
            recorded_variables: list[str]=None,
            start:bool=False,
            ) -> "BoldMonitor":

        boldmonitor = BoldMonitor(
                populations=populations,
                bold_model=bold_model,
                mapping=mapping,
                scale_factor=scale_factor,
                normalize_input=normalize_input,
                recorded_variables=recorded_variables,
                start=start,
                net_id=self.id,
            )
        
        self._extensions.append(boldmonitor)

        return boldmonitor
    
    ###################################
    # Compile
    ###################################

    def compile(self,
                directory:str='annarchy',
                clean:bool=False,
                compiler:str="default",
                compiler_flags:list[str]="default",
                add_sources:str="",
                extra_libs:str="",
                cuda_config:dict={'device': 0},
                annarchy_json:str="",
                silent:bool=False,
                debug_build:bool=False,
                profile_enabled:bool=False):
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
        Compiler.compile(
            directory=directory, 
            clean=clean, 
            silent=silent, 
            debug_build=debug_build, 
            add_sources=add_sources, 
            extra_libs=extra_libs, 
            compiler=compiler, 
            compiler_flags=compiler_flags, 
            cuda_config=cuda_config, 
            annarchy_json=annarchy_json, 
            profile_enabled=profile_enabled, 
            net_id=self.id)
        

    ###################################
    # Parallel run
    ###################################
    def copy(self, *args, **kwargs):
        """
        Returns a copy of the Network instance.
        """
        # Create an instance of the child class
        net = self.__class__(*args, **kwargs)

        # Instantiate the network with the current id.
        net.instantiate(self.id)

        return net

    def instantiate(self, import_id=-1):
        # Instantiate the network (but do not compile it) as if it had the provided id.
        Compiler._instantiate(self.id, import_id=import_id, cuda_config=None, user_config=None, core_list=None)

    def parallel_run(
            self, 
            method,
            number:int, 
            max_processes:int=-1, 
            measure_time:bool=False, 
            *args, **kwargs
        ):
        """
        Runs the provided method for multiple copies of the network.
        """

        import multiprocessing as mp

        with mp.Pool(processes=min(number, max_processes if max_processes > 0 else mp.cpu_count())) as pool:
            results = pool.map(self._worker, [(self.id, self.__class__, method, args, kwargs)] * number)

        return results

    @staticmethod
    def _worker(params):
        id, classname, method, args, kwargs = params
        net = classname()
        net.instantiate(import_id=id)
        result = method(net, *args, **kwargs)
        del net
        return result



    ###################################
    # Simulation
    ###################################
    def simulate(self, duration:float, measure_time:bool=False):
        """
        Runs the network for the given duration in milliseconds. 
        
        The number of simulation steps is  computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms):

        ```python
        net.simulate(1000.0)
        ```

        :param duration: the duration in milliseconds.
        :param measure_time: defines whether the simulation time should be printed (default=False).

        """
        Simulate.simulate(duration, measure_time, net_id=self.id)

    def simulate_until(self, max_duration:float, population:"Population", operator:str='and', measure_time:bool=False) -> float:
        """
        Runs the network for the maximal duration in milliseconds. If the `stop_condition` defined in the population becomes true during the simulation, it is stopped.

        One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

        Example:

        ```python
        pop1 = ann.Population( ..., stop_condition = "r > 1.0 : any")
        ...
        net.compile()
        net.simulate_until(max_duration=1000.0. population=pop1)
        ```

        :param max_duration: the maximum duration of the simulation in milliseconds.
        :param population: the (list of) population whose ``stop_condition`` should be checked to stop the simulation.
        :param operator: operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
        :param measure_time: defines whether the simulation time should be printed (default=False).
        :returns: the actual duration of the simulation in milliseconds.
        """
        return Simulate.simulate_until(max_duration=max_duration, population=population, operator=operator, measure_time=measure_time, net_id=self.id)

    def step(self) -> None:
        """
        Performs a single simulation step (duration = ``dt``).
        """
        Simulate.step(self.id)

    def reset(self, populations:bool=True, projections:bool=False, monitors:bool=True, synapses:bool=False) -> None:
        """
        Reinitialises the network to its state before the call to compile.

        :param populations: if True (default), the neural parameters and variables will be reset to their initial value.
        :param projections: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
        :param synapses: if True, the synaptic weights will be erased and recreated (default=False).
        """
        Global.reset(populations=populations, projections=projections, synapses=synapses, monitors=monitors, net_id=self.id)

    def get_time(self) -> float:
        "Returns the current time in ms."
        return Global.get_time(self.id)

    def set_time(self, t:float) -> None:
        """
        Sets the current time in ms.

        **Warning:** can be dangerous for some spiking models.
        """
        Global.set_time(t, self.id)

    def get_current_step(self) -> int:
        "Returns the current simulation step."
        return Global.get_current_step(self.id)

    def set_current_step(self, t:int):
        """
        Sets the current simulation step.

        **Warning:** can be dangerous for some spiking models.
        """
        Global.set_current_step(t, self.id)

    def set_seed(self, seed:int, use_seed_seq:bool=True) -> None:
        """
        Sets the seed of the random number generators for this network.
        """
        Global.set_seed(seed=seed, use_seed_seq=use_seed_seq, net_id=self.id)

    def enable_learning(self, projections:list=None, period:float=None, offset:float=None) -> None:
        """
        Enables learning for all projections.

        :param projections: the projections whose learning should be enabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self._projections
        for proj in projections:
            proj.enable_learning(period=period, offset=offset)

    def disable_learning(self, projections:list=None) -> None:
        """
        Disables learning for all projections.

        :param projections: the projections whose learning should be disabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self._projections
        for proj in projections:
            proj.disable_learning()

    ###################################
    # IO
    ###################################
    def load(self, filename:str, populations:bool=True, projections:bool=True, pickle_encoding:str=None):
        """
        Loads a saved state of the current network by calling ANNarchy.core.IO.load().

        :param filename: filename, may contain relative or absolute path.
        :param populations: if True, population data will be saved (by default True)
        :param projections: if True, projection data will be saved (by default True)
        :param pickle_encoding: optional parameter provided to the pickle.load() method. If set to None the default is used.
        """
        IO.load(filename=filename, populations=populations, projections=projections, pickle_encoding=pickle_encoding, net_id=self.id)

    def save(self, filename:str, populations:bool=True, projections:bool=True):
        """
        Saves the current network by calling ANNarchy.core.IO.save().

        :param filename: filename, may contain relative or absolute path.
        :param populations: if True, population data will be saved (by default True)
        :param projections: if True, projection data will be saved (by default True)
        """
        IO.save(filename, populations, projections, self.id)

    ###################################
    # Memory
    ###################################
    def _cpp_memory_footprint(self):
        """
        Print the C++ memory consumption for populations, projections on the console.
        """
        for pop in self.get_populations():
            print(pop.name, pop.size_in_bytes())

        for proj in self.get_projections():
            print(proj.name, proj.size_in_bytes())

        for mon in self._monitors:
            print(type(mon), mon.size_in_bytes())

    ###################################
    # Access methods using names
    ###################################

    def get_population(self, name:str) -> "Population":
        """
        Returns the population with the given name.

        :param name: name of the population
        :returns: The requested ``Population`` object if existing, ``None`` otherwise.
        """
        for pop in self._populations:
            if pop.name == name:
                return pop
        Messages._print('get_population(): the population', name, 'does not exist in this network.')
        return None

    def get_projection(self, name:str) -> "Projection":
        """
        Returns the projection with the given name.

        :param name: name of the projection
        :returns: The requested ``Projection`` object if existing, ``None`` otherwise.
        """
        for proj in self._projections:
            if proj.name == name:
                return proj
        Messages._print('get_projection(): the projection', name, 'does not exist in this network.')
        return None
    
    ###################################
    # Access methods for everything
    ###################################

    def get_populations(self) -> list["Population"]:
        """
        Returns a list of all declared populations in this network.

        :returns: the list of all populations in the network.
        """
        if self._populations == []:

            Messages._warning("Network.get_populations(): no populations attached to this network.")

        return self._populations

    def get_projections(self, post=None, pre=None, target=None, suppress_error=False) -> list["Projection"]:
        """
        Get a list of declared projections for the current network. By default, the method returns all connections within the network.

        By setting the arguments, post, pre and target one can select a subset.

        :param post: all returned projections should have this population as post.
        :param pre: all returned projections should have this population as pre.
        :param target: all returned projections should have this target.
        :param suppress_error: by default, ANNarchy throws an error if the list of assigned projections is empty. If this flag is set to True, the error message is suppressed.
        :returns: the list of all assigned projections in this network or a subset according to the arguments.

        """
        if len(self._projections) == 0:
            if not suppress_error:
                Messages._error("Network.get_projections(): no projections attached to this network.")

        return NetworkManager().get_projections(net_id=self.id, pre=pre, post=post, target=target, suppress_error=suppress_error)

    def get_monitors(self, obj=None) -> list["Monitor"]:
        """
        Returns a list of declared monitors. By default, all monitors are returned.
        By setting *obj*, only monitors recording from this object, either *Population* or *Projection* will be returned.
        """
        if obj is None:
            return self._monitors

        else:
            mon_list = []
            for monitor in self._monitors:
                if monitor.object == obj:
                    mon_list.append(monitor)

            return mon_list

    ###################################
    ### Deprecated interface
    ###################################

    def add(self, objects:list) -> None:
        """
        Adds a Population, Projection or Monitor to the network.

        :param objects: A single object or a list to add to the network.
        """
        if isinstance(objects, list):
            for item in objects:
                self._add_object(item)
        else:
            self._add_object(objects)


    def get(self, obj):
        """
        Returns the local Population, Projection or Monitor corresponding to the provided argument.

        `obj` is for example a top-level poopulation, while `net.get(pop)`is the copy local to the network.

        Example:

        ```python
        pop = ann.Population(100, Izhikevich)
        net = ann.Network()
        net.add(pop)
        net.compile()
        
        print(net.get(pop).v)
        ```

        :param obj: A single object or a list of objects.
        :returns: The corresponding object or list of objects.
        """
        if isinstance(obj, list):
            return [self._get_object(o) for o in obj]
        else:
            return self._get_object(obj)

    def _get_object(self, obj):
        "Retrieves the corresponding object."
        if isinstance(obj, Population):
            for pop in self._populations:
                if pop.id == obj.id:
                    return pop
        elif isinstance(obj, PopulationView):
            for pop in self._populations:
                if pop.id == obj.id:
                    return PopulationView(pop, obj.ranks) # Create on the fly?
        elif isinstance(obj, Projection):
            for proj in self._projections:
                if proj.id == obj.id:
                    return proj
        elif isinstance(obj, Monitor):
            for m in self._monitors:
                if m.id == obj.id:
                    return m
        elif isinstance(obj, BoldMonitor):
            for m in self._extensions:
                if m.id == obj.id:
                    return m
        else:
            Messages._error('The network has no such object:', obj.name, obj)




    def _add_object(self, obj):
        """
        Add the object *obj* to the network.

        TODO: instead of creating copies by object construction, one should check if deepcopy works ...
        """
        if isinstance(obj, Population):
            # Create a copy
            pop = obj._copy()

            # Remove the object created by _copy from the global network
            NetworkManager()._remove_last_item_from_list(net_id=0, list_name='populations')

            # Copy import properties
            pop.id = obj.id
            pop.name = obj.name
            pop.class_name = obj.class_name
            pop.init = obj.init
            pop.enabled = obj.enabled
            if not obj.enabled: # Also copy the enabled state:
                pop.disable()

            # Add the copy to the local network
            NetworkManager().add_population(net_id=self.id, population=pop)
            self._populations.append(pop)

            # Check whether the computation of mean-firing rate is requested
            if obj._compute_mean_fr > 0:
                pop.compute_firing_rate(obj._compute_mean_fr)

        elif isinstance(obj, Projection):
            # Check the pre- or post- populations
            try:
                pre_pop = self.get(obj.pre)
                if isinstance(obj.pre, PopulationView):
                    pre = PopulationView(population=pre_pop.population, ranks=obj.pre.ranks)
                else:
                    pre = pre_pop
                post_pop = self.get(obj.post)
                if isinstance(obj.post, PopulationView):
                    post = PopulationView(population=post_pop.population, ranks=obj.post.ranks)
                else:
                    post = post_pop
            except:
                Messages._error('Network.add(): The pre- or post-synaptic population of this projection are not in the network.')

            # Create the projection
            proj = obj._copy(pre=pre, post=post)

            # Remove the object created by _copy from the global network
            NetworkManager()._remove_last_item_from_list(net_id=0, list_name='projections')

            # Copy import properties
            proj.id = obj.id
            proj.name = obj.name
            proj.init = obj.init

            # Copy the connectivity properties if the projection is not already set
            if proj._connection_method is None:
                proj._store_connectivity(method=obj._connection_method, args=obj._connection_args, delay=obj._connection_delay, storage_format=obj._storage_format, storage_order=obj._storage_order)

            # Add the copy to the local network
            NetworkManager().add_projection(net_id=self.id, projection=proj)
            self._projections.append(proj)

        elif isinstance(obj, BoldMonitor):
            # Create a copy of the monitor
            m = BoldMonitor(
                populations=obj._populations,
                bold_model=obj._bold_model,
                mapping=obj._mapping,
                scale_factor=obj._scale_factor,
                normalize_input=obj._normalize_input,
                recorded_variables=obj._recorded_variables,
                start=obj._start,
                net_id=self.id,
                copied=True
            )

            # there is a bad mismatch between object ids:
            #
            # m.id     is dependent on len(_network[net_id].monitors)
            # obj.id   is dependent on len(_network[0].monitors)
            m.id = obj.id # TODO: check this !!!!

            # Stop the master monitor, otherwise it gets data.
            for var in obj._monitor.variables:
                try:
                    setattr(obj._monitor.cyInstance, 'record_'+var, False)
                except:
                    pass

            # assign contained objects
            m._monitor = self._get_object(obj._monitor)
            m._bold_pop = self._get_object(obj._bold_pop)
            m._acc_proj = []
            for tmp in obj._acc_proj:
                m._acc_proj.append(self._get_object(tmp))

            # need to be done manually for copied instances
            m._initialized = True

            # Add the copy to the local network (the monitor writes itself already in the right network)
            self._extensions.append(m)

        elif isinstance(obj, Monitor):
            # Get the copied reference of the object monitored
            # try:
            #     obj_copy = self.get(obj.object)
            # except:
            #     Messages._error('Network.add(): The monitor does not exist.')

            # Stop the master monitor, otherwise it gets data.
            for var in obj.variables:
                try:
                    setattr(obj.cyInstance, 'record_'+var, False)
                except:
                    pass
            # Create a copy of the monitor
            m = Monitor(obj=self._get_object(obj.object), variables=obj.variables, period=obj._period, period_offset=obj._period_offset, start=obj._start, net_id=self.id)

            # there is a bad mismatch between object ids:
            #
            # m.id     is dependent on len(_network[net_id].monitors)
            # obj.id   is dependent on len(_network[0].monitors)
            m.id = obj.id # TODO: check this !!!!

            # Add the copy to the local network (the monitor writes itself already in the right network)
            self._monitors.append(m)