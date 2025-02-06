"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import copy
import numpy as np
import time

from typing import List
from dataclasses import dataclass, field

from ANNarchy.core.Population import Population
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core.Neuron import Neuron
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.Constant import Constant

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import ConfigManager, get_global_config, _update_global_config
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

        instance._init_args = args
        instance._init_kwargs = kwargs

        return instance


@dataclass
class NetworkData :
    "Container for all the data of the network."
    populations: List = field(default_factory=list)
    projections: List = field(default_factory=list)
    monitors: List = field(default_factory=list)
    extensions: List = field(default_factory=list)
    constants: List = field(default_factory=list)
    instance = None
    compiled:bool = False
    directory:str = None

class Network (metaclass=NetworkMeta):
    """
    A network is a collection of populations, projections and monitors that runs the simulation.

    TODO
    """

    def __init__(self, *args, **kwargs):

        # Constructor should only be called once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Register the network in the NetworkManager()
        self.id = NetworkManager().add_network(self)

        # Create the data structure to store populations and projections
        self._data = NetworkData()

        # Get the default config
        self._config = copy.deepcopy(ConfigManager()._config)

        # Overwrite config
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
            self._config['seed'] = seed
            np.random.seed(seed)
            _update_global_config('seed', seed)

        # Callbacks
        self._callbacks = []
        self._callbacks_enabled = True


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

        for pop in self._data.populations:
            pop._clear()
            del pop

        for proj in self._data.projections:
            proj._clear()
            del proj

        for mon in self._data.monitors:
            mon._clear()
            del mon

        for ext in self._data.extensions:
            ext._clear()
            del ext

        for const in self._data.constants:
            del const

        NetworkManager().remove_network(self)

    def clear(self):
        """
        Empties the network to prevent a memory leak until the garbage collector wakes up.
        """
        for pop in self._data.populations:
            pop._clear()

        for proj in self._data.projections:
            proj._clear()

        for mon in self._data.monitors:
            mon._clear()

        for ext in self._data.extensions:
            ext._clear()

        NetworkManager().get_network(self.id).instance = None

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
        """
        TODO
        """

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
        
        return proj
    
    def monitor(
            self,
            obj: Population | PopulationView | Projection, 
            variables:list=[], 
            period:float=None, 
            period_offset:float=None, 
            start:bool=True, 
            ) -> Monitor:
        """
        TODO
        """
        
        if isinstance(obj, Monitor): # trick if one does not use obj=
            monitor = obj._copy(self.id)
        else:
            monitor = Monitor(
                obj=obj, 
                variables=variables, 
                period=period, 
                period_offset=period_offset, 
                start=start, 
                net_id=self.id
            )
            
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
        """
        TODO
        """

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

        return boldmonitor
    
    
    def _import_constants(self):
        """
        Tells the network to use the constants defined at the global level.
        """
        constants = NetworkManager().magic_network().get_constants()

        for c in constants:
            # Create the constant inside this network
            c._set_child_network(self.id)
            c._copy(self.id)
    
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
                profile_enabled:bool=False
        ) -> None:
        """
        Compiles the network.

        :param directory: name of the subdirectory where the code will be generated and compiled. Default: "./annarchy/".
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
        Returns a new instance of the Network class, using the provided arguments to the constructor. 
        
        Beware, `Network.compile()` is not called, only the instantiation of the data structures. Nothing in the constructor should induce a recompilation.
        """
        # Create an instance of the child class
        net = self.__class__(*args, **kwargs)

        # Instantiate the network with the current id.
        net.instantiate(self.id)

        return net

    def instantiate(self, import_id=-1):
        """
        Instantiates the network using the code generated for the network of id `import_id`.
        """
        # Instantiate the network (but do not compile it) as if it had the provided id.
        Compiler._instantiate(self.id, import_id=import_id, cuda_config=None, user_config=None, core_list=None)

    def parallel_run(
            self, 
            method,
            number:int, 
            max_processes:int=-1, 
            seed: int | str=None,
            measure_time:bool=False, 
            *args, **kwargs
        ):
        """
        Runs the provided method for multiple copies of the network.

        TODO
        """
        Messages._debug("Network was created with ", self._init_args, "and", self._init_kwargs)

        if measure_time:
            tstart = time.time()

        # Seed
        if seed is None: # 
            seeds = [None for _ in range(number)]

        # Import multiprocessing here to avoid warnings when setting the start method
        import multiprocessing as mp

        # Create and run the processes
        with mp.Pool(processes=min(number, max_processes if max_processes > 0 else mp.cpu_count())) as pool:
            results = pool.map(
                        self._worker, # method to call
                        [
                            (self.id, 
                             self.__class__, 
                             method, 
                             self._init_args, 
                             self._init_kwargs)
                        ] * number # arguments
                    )

        # Time measurement
        if measure_time:
            Messages._print('Simulating', number, 'networks in parallel took', time.time() - tstart, 'seconds.')

        return results

    @staticmethod
    def _worker(params):
        "Worker method called by parallel_run()."

        # Get the parameters
        id, classname, method, args, kwargs = params
        
        # Create an instance with the same parameters as the original instance
        net = classname(*args, **kwargs)
        
        # Instantiate the network but not compile
        net.instantiate(import_id=id)
        
        # Run the simulation
        result = method(net)
        
        # Delete the network
        net.__del__()
        
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
        Sets the seed of the random number generator.
        """
        if self.compiled(): # Send the seed to the cython instance
            if get_global_config('disable_parallel_rng'):
                self.cy_instance.set_seed(seed, 1, use_seed_seq)
            else:
                self.cy_instance.set_seed(seed, get_global_config('num_threads'), use_seed_seq)
        else: # Store it in the config
            self._config['seed'] = seed
            self._config['use_seed_seq'] = use_seed_seq
            np.random.seed(seed)
            _update_global_config('seed', seed) # only option for RandomDistributions


    def enable_learning(self, projections:list=None, period:float=None, offset:float=None) -> None:
        """
        Enables learning for all projections.

        :param projections: the projections whose learning should be enabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self._data.projections
        for proj in projections:
            proj.enable_learning(period=period, offset=offset)

    def disable_learning(self, projections:list=None) -> None:
        """
        Disables learning for all projections.

        :param projections: the projections whose learning should be disabled. By default, all the existing projections are disabled.
        """
        if not projections:
            projections = self._data.projections
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
        for pop in self._data.populations:
            print(pop.name, pop.size_in_bytes())

        for proj in self._data.projections:
            print(proj.name, proj.size_in_bytes())

        for mon in self._data.monitors:
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
        for pop in self._data.populations:
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
        for proj in self._data.projections:
            if proj.name == name:
                return proj
        Messages._print('get_projection(): the projection', name, 'does not exist in this network.')
        return None

    def get_constant(self, name:str) -> "Constant":
        """
        Returns the constant with the given name.

        :param name: name of the constant
        :returns: The requested `Constant` object if existing, `None` otherwise.
        """
        for constant in self._data.constants:
            if constant.name == name:
                return constant
        Messages._print('get_constant(): the constant', name, 'does not exist in this network.')
        return None
    
    ###################################
    # Access methods for everything
    ###################################

    def get_populations(self) -> list["Population"]:
        """
        Returns a list of all declared populations in this network.

        :returns: the list of all populations in the network.
        """
        return self._data.populations

    def get_projections(self) -> list["Projection"]:
        """
        Returns a list of all declared projections for the current network. 

        :returns: the list of all assigned projections in this network or a subset according to the arguments.

        """
        return self._data.projections

    def get_monitors(self) -> list["Monitor"]:
        """
        Returns a list of declared monitors. 
        """
        return self._data.monitors

    def get_extensions(self) -> list:
        """
        Returns a list of declared extensions (e.g. BOLD monitors). 
        """
        return self._data.extensions

    def get_constants(self) -> list:
        """
        Returns a list of declared constants. 
        """
        return self._data.constants


    
    ###################################
    # Store objects in the data structure
    ###################################
    def _add_population(self, population:Population) -> int :
        pop_id = len(self._data.populations)
        self._data.populations.append(population)
        return pop_id
    
    def _add_projection(self, projection:Projection) -> int :
        proj_id = len(self._data.projections)
        self._data.projections.append(projection)
        return proj_id
    
    def _add_monitor(self, monitor:Monitor) -> int :
        mon_id = len(self._data.monitors)
        self._data.monitors.append(monitor)
        return mon_id
    
    def _add_extension(self, extension) -> int :
        ext_id = len(self._data.extensions)
        self._data.extensions.append(extension)
        return ext_id
    
    def _add_constant(self, constant) -> int :
        const_id = len(self._data.constants)
        self._data.constants.append(constant)
        return const_id
    

    
    ###################################
    # Properties
    ###################################
    @property
    def instance(self):
        """
        C++ instance.
        """
        return self._data.instance
    
    @instance.setter
    def instance(self, instance) -> None:
        self._data.instance = instance

    @property
    def compiled(self) -> bool:
        """
        Whether the network has been compiled.
        """
        return self._data.compiled
    
    @compiled.setter
    def compiled(self, value) -> None:
        self._data.compiled = True
    
    @property
    def directory(self) -> str:
        """
        Directory in which the network has been compiled.
        """
        return self._data.directory
    
    @directory.setter
    def directory(self, directory) -> None :
        self._data.directory = directory


    ###################################
    ### Deprecated interface
    ###################################

    def add(self, objects:list) -> None:
        """
        REMOVED
        """
        Messages._error("Network.add(): adding populations or projections defined at the top level to a network is removed since 5.0. Use Network.create() and Network.connect() instead.")


    def get(self, obj):
        """
        REMOVED
        """
        Messages._error("Network.add(): adding populations or projections defined at the top level to a network is removed since 5.0. Use Network.create() and Network.connect() instead.")