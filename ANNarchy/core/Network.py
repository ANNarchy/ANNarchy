"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
import secrets
import inspect

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
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

import ANNarchy.extensions.bold as bold
import ANNarchy.core.Global as Global
import ANNarchy.core.Simulate as Simulate
import ANNarchy.core.IO as IO
import ANNarchy.generator.Compiler as Compiler

# Meta class to avoid forcing the constructor 
class NetworkMeta(type):

    def __call__(cls, *args, **kwargs):
        
        # Extract the seed and dt from kwargs if provided
        seed = kwargs.pop('seed', None)
        dt = kwargs.pop('dt', None)

        # Create an instance without calling __init__
        instance = cls.__new__(cls)

        # Call the parent class's __init__ methods first
        if isinstance(instance, Network):
            Network.__init__(instance, dt=dt, seed=seed)

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
    A network creates the populations, projections and monitors, and controls the simulation.

    ```python
    net = ann.Network(dt=1.0, seed=42)
    ```

    To create a population:

    ```python
    pop = net.create(100, neuron=ann.Izhikevich)
    ```

    To connect two populations:

    ```python
    proj = net.connect(pre=pop1, post=pop2, target='exc', synapse=ann.STDP)
    ```

    To monitor a population or projection:

    ```python
    m = net.monitor(pop, ['spike', 'v'])
    ```

    To compile the network:

    ```python
    net.compile()
    ```

    To simulate for one second:

    ```python
    net.simulate(1000.)
    ```
    
    Refer to the manual for more functionalities. `dt` and `seed` must be passed with keywords.

    :param dt: step size in milliseconds.
    :param seed: seed for the random number generators.
    """

    def __init__(self, dt:float=None, seed:int=None):
        # Constructor should only be called once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Object tracking
        Messages._debug("Instantiate Network object", self)

        # Register the network in the NetworkManager()
        self.id = NetworkManager().add_network(self)

        # Create the data structure to store populations and projections
        self._data = NetworkData()

        # Get the default config
        ConfigManager().register_network(self.id)

        # dt
        if dt is not None:
            self.dt = dt

        # Draw a value for seed if not provided by user
        if seed is None:
            seed = secrets.randbits(32)  # Generates a random 32-bit integer

        # Store the seed value
        self._set_config('seed', seed)

        # initialize one RNG instance
        self._default_rng = np.random.default_rng(self.seed)
        # Store RNG state for Network.reset(reseed=True)
        self._default_rng_state = self._default_rng.bit_generator.state

        # Callbacks
        self._callbacks = []
        self._callbacks_enabled = True

        # Import constants from the global level
        self._import_constants()


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

        # clear attached objects
        self.clear()

        # de-register the network instance
        NetworkManager().remove_network(self)

    def create(
            self,
            geometry: tuple | int = None, 
            neuron: Neuron = None, 
            stop_condition:str = None, 
            name:str = None,
            population: Population = None,

            # Internal use only
            storage_order:str = 'post_to_pre', 
            ) -> Population:
        """
        Adds a population of neurons to the network.

        ```python
        net = ann.Network()
        pop = net.create(geometry=100, neuron=ann.Izhikevich, name="Excitatory population")
        ```

        Specific populations (e.g. `PoissonPopulation()`) can also be passed to the `population` argument, or simply as the first argument, in both cases do not provide other arguments.:

        ```python
        pop = net.create(population=ann.PoissonPopulation(100, rates=20.))
        # or
        pop = net.create(ann.PoissonPopulation(100, rates=20.))
        ```

        :param geometry: population geometry as tuple. If an integer is given, it is the size of the population. If an instance of `Population` is given, do not provide other arguments.
        :param neuron: `Neuron` instance. It can be user-defined or a built-in model.
        :param name: unique name of the population (optional).
        :param stop_condition: a single condition on a neural variable which can stop the simulation whenever it is true.
        :param population: instance of a `Population`. If given, do not provide other arguments.
        """
        # if both population and geometry are None, error
        if geometry is None and population is None:
            Messages._error("Network.create(): either 'geometry' or 'population' argument must be provided.")
        
        # if both are given as population instances, error
        if isinstance(geometry, Population) and population is not None:
            Messages._error("Network.create(): cannot provide both 'geometry' and 'population' as Population instances.")
        
        # if population is given, check its type
        if population is not None and not isinstance(population, Population):
            Messages._error("Network.create(): 'population' argument only accepts instances of ann.Population and its subclasses.")
        
        # if a population instance is given, check that the other arguments are None
        if isinstance(geometry, Population) or population is not None:
            if neuron is not None or name is not None or stop_condition is not None:
                Messages._error("Network.create(): do not give other arguments when a `Population` instance is provided. Define them when creating the given `Population` instance.")
        if population is not None:
            if geometry is not None:
                Messages._error("Network.create(): do not give other arguments when a `Population` instance is provided. Define them when creating the given `Population` instance.")

        if isinstance(geometry, Population):  # trick if one does provide `Population` as first argument
            # Population is already created
            pop = geometry._copy(self.id)
        elif population is not None:
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
            pre: str | Population = None, 
            post: str | Population = None, 
            target: str = "", 
            synapse: Synapse = None, 
            name:str = None, 
            projection:Projection = None,
            # Internal
            disable_omp:bool = True, 
        ) -> "Projection":
        """
        Connects two populations by creating a projection.

        ```python
        net.connect(pre=pop1, post=pop2, target="exc", synapse=STDP)
        ```

        If the `synapse` argument is omitted, defaults synapses without plastivity will be used (`psp = "w * pre.r"` for rate-coded projections, `pre_spike="g_target += w"` for spiking ones.).

        Specific projections can be passed to the `projection` argument, or as the first unnamed argument. In both cases, do not provide other arguments.

        ```python
        net.connect(projection=ann.DecodingProjection(pre, post, 'exc'))
        ```

        :param pre: pre-synaptic population. If an instance of `Projection` is given, do not provide other arguments.
        :param post: post-synaptic population.
        :param target: type of the connection.
        :param synapse: `Synapse` class or instance.
        :param name: (optional) name of the Projection.
        :param projection: specific projection. If given, do not provide other arguments.
        """
        # if both projection and pre are None, error
        if pre is None and projection is None:
            Messages._error("Network.connect(): either 'pre' or 'projection' argument must be provided.")

        # if both are given as projection instances, error
        if isinstance(pre, Projection) and projection is not None:
            Messages._error("Network.connect(): cannot provide both 'pre' and 'projection' arguments.")

        # if projection is given, check its type
        if projection is not None and not isinstance(projection, Projection):
            Messages._error("Network.connect(projection=proj) only accepts instances of ann.Projection and its subclasses.")

        # if a projection instance is given, check that the other arguments are not provided
        if isinstance(pre, Projection) or projection is not None:
            if post is not None or target != "" or synapse is not None or name is not None:
                Messages._error("Network.connect(): do not give other arguments when a `Projection` instance is provided. Define them when creating the given `Projection` instance.")
        if projection is not None:
            if pre is not None:
                Messages._error("Network.connect(): do not give other arguments when a `Projection` instance is provided. Define them when creating the given `Projection` instance.")

        if isinstance(pre, Projection):  # trick if one does provide `Projection` as first argument
            # Projection is already created
            proj = pre._copy(pre.pre, pre.post, self.id)
        elif projection is not None:
            # Projection is already created
            proj = projection ._copy(projection.pre, projection.post, self.id)
        else:
            # Create the projection
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
            name:str=None,
            ) -> Monitor:
        """
        Creates a Monitor on variables of the specified object.

        The object can be an instance of either `Population`, `PopulationView`, `Dendrite` or `Projection`.

        The variables must be declared by the neuron or synapse type. For spiking neurons, `'spike'` can be also recorded.

        ```python
        m = net.monitor(pop, ['v', 'spike'])
        m = net.monitor(proj.dendrite(0), 'w')
        ```

        By default, the monitors start recording right after `Network.compile()`, but you can hold them with `start=False`. Starting the recordings necessitates then to call the ``start()``mehtod. Pausing/resuming the recordings is cheived through the `pause()` and `resume()`.

        ```python
        m.start() # Start recording
        net.simulate(T)
        m.pause() # Pause recording
        net.simulate(T)
        m.resume() # Resume recording
        net.simulate(T)

        data = m.get() # Get the data
        ```

        :param obj: object to monitor. Must be a `Population`, `PopulationView`, `Dendrite` or `Projection` object.
        :param variables: single variable name or list of variable names to record (default: []).
        :param period:
        :param period_offset:
        :param start:
        :param name:
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
                name=name,
                net_id=self.id
            )
            
        return monitor
    
    def boldmonitor(
            self,
            populations: list=None,
            bold_model: bold.BoldModel = None,
            mapping: dict={'I_CBF': 'r'},
            scale_factor: list[float]=None,
            normalize_input: list[int]=None,
            recorded_variables: list[str]=None,
            start:bool=False,
            ) -> "bold.BoldMonitor":
        """
        Monitors the BOLD signal of several populations using a computational model.

        The BOLD monitor transforms one or two input population variables (such as the mean firing rate) into a recordable BOLD signal according to a computational model (for example a variation of the Balloon model).

        Several models are available or can be created with a `bold.BoldModel` instance. These models must be explicitly imported:

        ```python
        import ANNarchy.extensions.bold as bold

        m_bold = net.boldmonitor(
            # Recorded populations
            populations = [pop1, pop2], 
            # BOLD model to use (default is balloon_RN)
            bold_model = bold.balloon_RN(), 
            # Mapping from pop.r to I_CBF
            mapping = {'I_CBF': 'r'}, 
            # Time window to compute the baseline.
            normalize_input = 2000,  
            # Variables to be recorded
            recorded_variables = ["I_CBF", "BOLD"]  
        )

        # Unlike regular monitors, BOLD monitors must be explicitly started
        m_bold.start()
        net.simulate(5000) 
        bold_data = m_bold.get("BOLD")
        ```

        :param populations: list of recorded populations.
        :param bold_model: computational model for BOLD signal defined as a BoldModel object. Default is `bold.balloon_RN`.
        :param mapping: mapping dictionary between the inputs of the BOLD model (`I_CBF` for single inputs, `I_CBF` and `I_CMRO2` for double inputs in the provided examples) and the variables of the input populations. By default, `{'I_CBF': 'r'}` maps the firing rate `r` of the input population(s) to the variable `I_CBF` of the BOLD model. 
        :param scale_factor: list of float values to allow a weighting of signals between populations. By default, the input signal is weighted by the ratio of the population size to all populations within the recorded region.
        :param normalize_input: list of integer values which represent a optional baseline per population. The input signals will require an additional normalization using a baseline value. A value different from 0 represents the time period for determing this baseline in milliseconds (biological time).
        :param recorded_variables: which variables of the BOLD model should be recorded? (by default, the output variable of the BOLD model is added, e.g. ["BOLD"] for the provided examples).
        :param start: whether to start recording directly.
        """

        if bold_model is None:
            bold_model = bold.balloon_RN

        boldmonitor = bold.BoldMonitor(
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
    
    def constant(self, name, value):
        """
        Adds a constant to the network.

        ```python
        c = net.constant('c', 2.0)

        # `c` can be used in a neuron/synapse definition
        neuron = ann.Neuron(equations="r = c * sum(exc)")

        # Change the value of the constant in the network.
        c.set(10.0) 
        ```

        :param name: name of the constant.
        :param value: initial value of the constant.
        """
        constant = Constant(name, value, net_id=self.id)
        return constant
    
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
    # Simulation
    ###################################
    def simulate(self, duration:float, measure_time:bool=False):
        """
        Runs the network for the given duration in milliseconds. 
        
        The number of simulation steps is  computed relative to the discretization step `dt` declared in the constructor (default: 1 ms):

        ```python
        net.simulate(1000.0)
        ```

        :param duration: the duration in milliseconds.
        :param measure_time: defines whether the simulation time should be printed.

        """
        Simulate.simulate(duration, measure_time, net_id=self.id)

    def simulate_until(self, max_duration:float, population:"Population", operator:str='and', measure_time:bool=False) -> float:
        """
        Runs the network for the maximal duration in milliseconds until a `stop_condition` is met.
        
        Whenever the `stop_condition` defined in `population` becomes true, the simulation is stopped.

        The method returns the actual duration of the simulation in milliseconds.

        One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

        Example:

        ```python
        pop1 = net.create( ..., stop_condition = "r > 1.0 : any")
        
        net.compile()
        
        net.simulate_until(max_duration=1000.0. population=pop1)
        ```

        :param max_duration: the maximum duration of the simulation in milliseconds.
        :param population: the (list of) population whose ``stop_condition`` should be checked to stop the simulation.
        :param operator: operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
        :param measure_time: defines whether the simulation time should be printed (default=False).
        """
        return Simulate.simulate_until(max_duration=max_duration, population=population, operator=operator, measure_time=measure_time, net_id=self.id)

    def step(self) -> None:
        """
        Performs a single simulation step (duration = `dt`).
        """
        Simulate.step(self.id)

    def reset(self, populations:bool=True, projections:bool=False, synapses:bool=False, monitors:bool=True, reseed_rng:bool=True) -> None:
        """
        Reinitialises the network to its state before the call to `compile()`.  The network time will be set to 0ms.

        :param populations: if True (default), the neural parameters and variables will be reset to their initial value.
        :param projections: if True, the synaptic parameters and variables (except the connections) will be reset (default=False).
        :param synapses: if True, the synaptic weights will be erased and recreated (default=False).
        :param monitors: if True, the monitors will be emptied and reset (default=True).
        ;param reseed_rng: if True, RNG generators will be reset using the stored seed (default=True).
        """

        self.instance.set_time(0)

        if populations:
            for pop in self.get_populations():
                pop.reset()

            # pop.reset only clears spike container with no or uniform delay
            for proj in self.get_projections():
                if hasattr(proj.cyInstance, 'reset_ring_buffer'):
                    proj.cyInstance.reset_ring_buffer()

        if synapses and not projections:
            Messages._warning("reset(): if synapses is set to true this automatically enables projections==true")
            projections = True

        if projections:
            for proj in self.get_projections():
                proj.reset(attributes=-1, synapses=synapses)

        if monitors:
            for monitor in self.get_monitors():
                monitor.reset()

        if reseed_rng:
            # Python:   re-initialize the RNG with initially stored state
            self._default_rng.bit_generator.state = self._default_rng_state

            # CPP:      re-initialize the RNG with the stored configuration
            if self._get_config('disable_parallel_rng'):
                self.instance.set_seed(self._get_config('seed'), 1, self._get_config('use_seed_seq'))
            else:
                self.instance.set_seed(self._get_config('seed'), self._get_config('num_threads'), self._get_config('use_seed_seq'))

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

        
    def clear(self):
        """
        Empties the network to prevent a memory leak until the garbage collector wakes up.
        """
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

    ###################################
    # Parallel run
    ###################################

    def _instantiate(self, import_id=-1):
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
            seeds: int | str| list[int] = None,
            measure_time:bool=False, 
            **kwargs
        ):
        """
        Runs the provided method for multiple copies of the network.

        **Important:** The network must have defined as a subclass of `Network`, not a `Network` instance where `create()` and `connect()` where sequentially called. 
        
        See the manual for more explanations:

        ```python
        class PulseNetwork(ann.Network):
            def __init__(self):
                self.create(...)

        # Simulation method
        def simulation(net, duration=1000.):
            net.simulate(duration)
            t, n = net.m.raster_plot()
            return t, n

        # Create a network
        net = PulseNetwork()
        net.compile()

        # Parallel simulation
        results = net.parallel_run(method=simulation, number=4)
        ```

        :param method: method invoked for each copy of the network.
        :param number: number of simulations. 
        :param max_processes: maximum number of concurrent processes. Defaults to the number of cores.
        :param seeds: list of seeds for each network. If `None`, the seeds will be be randomly set. If `'same'`, it will be the same as the current network. If `'sequential'`, the seeds will be incremented for each network (42, 43, 44, etc).
        :param measure_time: if the total duration of the simulation should be reported at the end.
        """
        Messages._debug("Network was created with ", self._init_args, "and", self._init_kwargs)

        if type(self) is Network:
            Messages._error("Network.parallel_run(): the network must be an instance of a class deriving from Network, not Network itself.")

        if measure_time:
            tstart = time.time()

        # Seed
        if seeds is None: # Random seed for each network
            seeds = [None for _ in range(number)]
        elif seeds == 'same': # Same seed as the current one for each network
            seeds = [self._get_config('seed') for _ in range(number)]
        elif seeds == 'sequential': # Sequential seeds
            seeds = [self._get_config('seed') + i + 1 for i in range(number)]

        seeds = list(seeds)
        if len(seeds) != number:
            Messages._error("Network.parallel_run(): the list of seeds must have the same size as the number of simulations.")

        # Retrieve dictionary of parameters passed to the current constructor
        current_init_args = self._retrieve_initargs()

        # Arguments to the instances and method
        init_args = [current_init_args.copy() for _ in range(number)]
        method_args = [{} for _ in range(number)]

        # Iterate over 
        for key, val in kwargs.items():
            if key in current_init_args.keys(): # put the value in the constructor
                if isinstance(val, list):
                    if len(val) != number:
                        Messages._error("Network.parallel_run(): the list of values for", key, "must have the same size as the number of simulations.")
                    for i in range(number): init_args[i][key] = val[i]
                else:
                    for i in range(number): init_args[i][key] = val

            else: # put it in the simulation method
                if isinstance(val, list):
                    if len(val) != number:
                        Messages._error("Network.parallel_run(): the list of values for", key, "must have the same size as the number of simulations.")
                    for i in range(number): method_args[i][key] = val[i]
                else:
                    for i in range(number): method_args[i][key] = val

        # Config
        config = ConfigManager().get_config(self.id)
        config.pop('dt', 0.0)
        config.pop('seed', 0)

        # Import multiprocessing here to avoid warnings when setting the start method
        import multiprocessing as mp

        # Create arguments
        args = []
        for i in range(number):
            parameters = (
                    self.id, 
                    self.__class__, 
                    config,
                    method,
                    self.dt,
                    seeds[i],
                    init_args[i], 
                    method_args[i],
            )
            args.append(parameters)

        # Create and run the processes
        with mp.Pool(processes=min(number, max_processes if max_processes > 0 else mp.cpu_count())) as pool:
            results = pool.map(
                        self._worker, # method to call
                        args # arguments
                    )

        # Time measurement
        if measure_time:
            Messages._print('Simulating', number, 'networks in parallel took', time.time() - tstart, 'seconds.')

        return results

    @staticmethod
    def _worker(params):
        "Worker method called by parallel_run()."

        # Get the parameters
        id, classname, config, method, dt, seed, init_args, method_args = params

        # Create an instance with the same parameters as the original instance
        net = classname(dt=dt, seed=seed, **init_args)

        # Set the config
        ConfigManager().set_config(net.id, config)

        # Transfer compilation state of "parent" network
        net.compiled = NetworkManager().get_network(id).compiled

        # Instantiate the network but not compile
        net._instantiate(import_id=id)
        
        # Run the simulation
        result = method(net, **method_args)
        
        # Delete the network
        net.__del__()
        
        return result
    
    def _retrieve_initargs(self):

        # args and kwargs were saved here
        Messages._debug(self._init_args, self._init_kwargs,)

        # inspect the signature of the constructor
        signature = inspect.signature(self.__init__)
        params = list(signature.parameters.values())
        
        # Create a dictionary to hold the reconstructed arguments
        reconstructed_args = {
            param.name: param.default
            for param in params
            if param.default is not param.empty
        }
        
        # Map *args to the corresponding parameter names
        for i, arg in enumerate(self._init_args):
            if i < len(params):
                param_name = params[i].name
                reconstructed_args[param_name] = arg
        
        # Add the **kwargs to the reconstructed arguments
        reconstructed_args.update(self._init_kwargs)
        
        return reconstructed_args
    

    def copy(self, *args, **kwargs):
        """
        Returns a new instance of the Network class, using the provided arguments to the constructor. 
        
        Beware, `Network.compile()` is not called, only the instantiation of the data structures. Nothing in the constructor should induce a recompilation.
        """
        if not 'dt' in kwargs.keys():
            kwargs.update(dict(dt=self.dt))

        # Sanity check
        if not self.compiled:
            Messages._error("The copied network should be compiled before calling Network.copy().")

        # Create an instance of the child class
        net = self.__class__(*args, **kwargs)

        # Transfer the compiled state from "parent" network
        net.compiled = self.compiled

        # Instantiate the network with the current id.
        net._instantiate(self.id)

        return net


    ###################################
    # Config
    ###################################
    def _get_config(self, key:str):
        "Returns the config value with the given key."
        return  ConfigManager().get(key=key, net_id=self.id)
    def _set_config(self, key:str, value):
        "Sets the config value with the given key."
        return  ConfigManager().set(key=key, value=value, net_id=self.id)
    
    def config(self, *args, **kwargs):
        """
        Configuration of the network. 

        This method is equivalent to calling `setup()` at the global level, but only influences the current network. The initial configuration of the network copies the values set in `setup()` at the time of the creation of the network.
        
        It can be called multiple times until `compile()` is called, new values of keys erasing older ones.

        The only functional difference with `setup()` is the seed, which should be passed to the constructor of `Network`, otherwise any random number generation in the constructor might be unseeded. `dt` can also be passed to the constructor, but setting it in `config()` is also fine.
        
        It takes various optional arguments. The most useful ones are:

        * `dt`: simulation step size in milliseconds (default: 1.0).
        * `paradigm`: parallel framework for code generation. Accepted values: "openmp" or "cuda" (default: "openmp").
        * `method`: default method to numerize the ODEs. Default is the explicit forward Euler method ('explicit').
        * `precision`: default floating precision for variables in ANNarchy. Accepted values: "float" or "double" (default: "double")
        * `structural_plasticity`: allows synapses to be dynamically added/removed during the simulation (default: False).
        * `seed`: the seed (integer) to be used in the random number generators (default = None is equivalent to time(NULL)).
        * `num_threads`: number of treads used by openMP (overrides the environment variable ``OMP_NUM_THREADS`` when set, default = None).

        Flags related to the optimization of the simulation kernels are:

        * `sparse_matrix_format`: the default matrix format for projections in ANNarchy (by default: List-In-List for CPUs and Compressed Sparse Row). Note that this affects only the C++ data structures.
        * `sparse_matrix_storage_order`: encodes whether the row in a connectivity matrix encodes pre-synaptic neurons (post_to_pre, default) or post-synaptic neurons (pre_to_post). Note that affects only the C++ data structures.
        * `only_int_idx_type`: if set to True (default) only signed integers are used to store pre-/post-synaptic ranks which was default until 4.7. If set to False, the index type used in a single projection is selected based on the size of the corresponding populations.
        * `visible_cores`: allows a fine-grained control which cores are useable for the created threads (default = [] for no limitation). It can be used to limit created openMP threads to a physical socket.

        The following parameters are mainly for debugging and profiling, and should be ignored by most users:

        * `verbose`: shows details about compilation process on console (by default False). Additional some information of the network construction will be shown.
        * `suppress_warnings`: if True, warnings (e. g. from the mathematical parser) are suppressed.
        * `show_time`: if True, initialization times are shown. Attention: verbose should be set to True additionally.
        * `disable_shared_library_time_offset`: by default False. If set to True, the shared library generated by ANNarchy will not be extended by time offset.
        """

        # RNG-related arguments are treated differently
        rng_changed = False
        if 'use_seed_seq' in kwargs.keys():
            seed_seq = kwargs.pop('use_seed_seq', self._get_config('use_seed_seq'))
            self._set_config('use_seed_seq', seed_seq)
            rng_changed = True
        if 'disable_parallel_rng' in kwargs.keys():
            par_rng = kwargs.pop('disable_parallel_rng', self._get_config('disable_parallel_rng'))
            self._set_config('disable_parallel_rng', par_rng)
            rng_changed = True
        if 'seed' in kwargs.keys():
            # update the value for both NumPy and C++ core (when the latter is already compiled)
            self.seed = int(kwargs.pop('seed'))
            rng_changed = True

        if rng_changed:
            Messages._debug("RNG generators are now configured as following:")
            Messages._debug("  seed={} (accounts for C++ and NumPy)".format(self.seed))
            Messages._debug("  use_seed_seq={}".format("True" if self._get_config('use_seed_seq') else "False"))
            Messages._debug("  disable_parallel_rng={}".format("True" if self._get_config('disable_parallel_rng') else "False"))

        # Process all other provided key-value pairs
        for key, value in kwargs.items():
            # sanity check: filter out performance flags
            if key in ConfigManager()._performance_related_config_keys:
                Messages._error("The performance related flag '"+key+"' can not be configured by setup()")
            elif key in ConfigManager().keys():
                ConfigManager().set(key, value, net_id=self.id)
            else:
                Messages._warning('setup(): unknown key:', key)

            if key == 'sparse_matrix_format':
                # check if this is a supported format
                if kwargs[key] not in ["lil", "csr", "csr_vector", "csr_scalar", "dense", "ell", "ellr", "sell", "coo", "bsr", "hyb", "auto"]:
                    Messages._error("The value", kwargs[key], "provided to sparse_matrix_format is not valid.")


    @property
    def seed(self) -> int:
        "Seed for the random number generator being used for both Python and C++."
        return self._get_config('seed')

    @seed.setter
    def seed(self, seed: int) -> None:
        "Prevent the setting of a seed by hand."
        Messages._error("The seed argument should not be overwritten.")

    @property
    def default_rng(self) -> np.random.Generator:
        """
        Get a pre-seeded RNG instance. During construction of the network object, a numpy.random.Generator
        object has been instantiated. This can be used to initialize random distributions. e.g., instances of
        ANNarchy.core.RandomDistribution.
        """
        return self._default_rng

    @default_rng.setter
    def default_rng(self, new_rng: np.random.Generator) -> None:
        "Prevent the setting of RNG instance by hand."
        Messages._error("The default_rng argument should not be overwritten.")

    @property
    def dt(self) -> float:
        "Step size in milliseconds for the integration of the ODEs."
        return self._get_config('dt')

    @dt.setter
    def dt(self, dt:float) -> None:
        self._set_config('dt', dt)



    ###################################
    # IO
    ###################################
    def load(self, filename:str, populations:bool=True, projections:bool=True, pickle_encoding:str=None):
        """
        Loads parameters and variables from a file created with `Network.save()`.

        :param filename: filename, may contain relative or absolute path.
        :param populations: if `True`, population data will be saved.
        :param projections: if `True`, projection data will be saved.
        :param pickle_encoding: optional parameter provided to the pickle.load() method. If set to None the default is used.
        """
        IO.load(filename=filename, populations=populations, projections=projections, pickle_encoding=pickle_encoding, net_id=self.id)

    def save(self, filename:str, populations:bool=True, projections:bool=True):
        """
        Saves the parameters and variables of the networkin a file.

        * If the extension is '.npz', the data will be saved and compressed using `np.savez_compressed` (recommended).
        * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.
        * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.
        * Otherwise, the data will be pickled into a simple binary text file using cPickle.

        **Warning:** The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

        Example:

        ```python
        net.save('results/init.npz')

        net.save('results/init.data')

        net.save('results/init.txt.gz')

        net.save('1000_trials.mat')
        ```

        :param filename: filename, may contain relative or absolute path.
        :param populations: if `True`, population data will be saved.
        :param projections: if `True`, projection data will be saved.
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
            print(pop.name, pop._size_in_bytes())

        for proj in self._data.projections:
            print(proj.name, proj._size_in_bytes())

        for mon in self._data.monitors:
            print(type(mon), mon._size_in_bytes())

    ###################################
    # Access methods using names
    ###################################

    def get_population(self, name:str) -> "Population":
        """
        Returns the population with the given name.

        :param name: name of the population
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
        """
        for proj in self._data.projections:
            if proj.name == name:
                return proj
        Messages._print('get_projection(): the projection', name, 'does not exist in this network.')
        return None

    def get_monitor(self, name:str) -> "Monitor":
        """
        Returns the monitor with the given name.

        :param name: name of the monitor
        """
        for mon in self._data.monitors:
            if mon.name == name:
                return mon
        Messages._print('get_monitor(): the monitor', name, 'does not exist in this network.')
        return None

    def get_extension(self, name:str) :
        """
        Returns the extension with the given name.

        :param name: name of the extension
        """
        for ext in self._data.extensions:
            if ext.name == name:
                return ext
        Messages._print('get_extension(): the extension', name, 'does not exist in this network.')
        return None

    def get_constant(self, name:str) -> "Constant":
        """
        Returns the constant with the given name.

        :param name: name of the constant
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
        """
        return self._data.populations

    def get_projections(self) -> list["Projection"]:
        """
        Returns a list of all declared projections for the current network. 
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
        return self._data.instance
    
    @instance.setter
    def instance(self, instance) -> None:
        self._data.instance = instance

    @property
    def compiled(self) -> bool:
        """
        Boolean indicating whether the network has been compiled.
        """
        return self._data.compiled
    
    @compiled.setter
    def compiled(self, compilation_state: bool) -> None:
        self._data.compiled = compilation_state

    @property
    def directory(self) -> str:
        """
        Directory in which the network has been compiled.
        """
        return self._data.directory
    
    @directory.setter
    def directory(self, directory: str) -> None:
        self._data.directory = directory

    @property
    def time(self) -> float:
        "Current time t in milliseconds."
        return Global.get_time(self.id)

    @time.setter
    def time(self, t:float) -> None:
        Global.set_time(t, self.id)

    @property
    def current_step(self) -> int:
        "Current simulation step."
        return Global.get_current_step(self.id)

    @current_step.setter
    def current_step(self, t:int) -> None:
        Global.set_current_step(t, self.id)

    ################################
    ## @every decorator
    ################################

    def callbacks_enabled(self):
        """
        Returns True if callbacks are enabled for the network.
        """
        return self._callbacks_enabled

    def disable_callbacks(self):
        """
        Disables all callbacks for the network.
        """
        self._callbacks_enabled = False

    def enable_callbacks(self):
        """
        Enables all declared callbacks for the network.
        """
        self._callbacks_enabled = True

    def clear_all_callbacks(self):
        """
        Clears the list of declared callbacks for the network.

        Cannot be undone!
        """
        self._callbacks.clear()

