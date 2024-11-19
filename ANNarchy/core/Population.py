"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Constant import Constant
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern import Messages

from .PopulationView import PopulationView
from .Random import RandomDistribution
from .Neuron import Neuron, IndividualNeuron

from typing import Iterator
import numpy as np
import copy, inspect


class Population :
    """
    Structure for a population of homogeneous neurons.

    Example:

    ```python
    pop = ann.Population(100, neuron=ann.Izhikevich, name="Excitatory population")
    ```
    
    :param geometry: population geometry as tuple. If an integer is given, it is the size of the population.
    :param neuron: `Neuron`instance. It can be user-defined or a built-in model.
    :param name: unique name of the population (optional, it defaults to `pop0`, `pop1`, etc).
    :param stop_condition: a single condition on a neural variable which can stop the simulation whenever it is true.
    """

    def __init__(self, 
                 geometry: tuple | int, 
                 neuron: "Neuron", 
                 name:str = None, 
                 stop_condition:str = None, 
                 storage_order:str = 'post_to_pre', 
                 copied = False):

        # Check if the network has already been compiled
        if NetworkManager().is_compiled(net_id=0) and not copied:
            Messages._error('You cannot add a population after the network has been compiled.')

        # Store the provided geometry. Automatically defines w, h, d, size
        self.geometry = geometry
        "Geometry of the population."
        self.width = 0
        "Width of the population."
        self.height = 0
        "Height of the population."
        self.depth = 0
        "Depth of the population."
        self.dimension = 0
        "Number of dimensions of the population."

        if isinstance(geometry, (int, float)):
            # 1D
            self.geometry = (int(geometry), )
            self.width = int(geometry)
            self.height = int(1)
            self.depth = int(1)
            self.dimension = int(1)

        elif isinstance(geometry, tuple):
            # a tuple is given, can be 1 .. N dimensional
            self.geometry = ()
            for d in geometry:
                self.geometry += (int(d),)
            self.width = int(geometry[0])
            if len(geometry)>=2:
                self.height = int(geometry[1])
            else:
                self.height = int(1)
            if len(geometry)>=3:
                self.depth = int(geometry[2])
            else:
                self.depth = int(1)

            self.dimension = len(geometry)
        else:
            Messages._error('Population(): the geometry must be either an integer or a tuple.')

        # Compute the size
        size = int(1)
        for i in range(len(self.geometry)):
            size *= int(self.geometry[i])
        self.size = int(size)
        "Size of the population."
        self.ranks = np.arange(self.size, dtype="int32")
        "Array of ranks in the population (between 0 and `size - 1`)."

        # Store the neuron type
        if inspect.isclass(neuron):
            self.neuron_type = neuron()
        else:
            self.neuron_type = copy.deepcopy(neuron)
        self.neuron_type._analyse()

        # Store the stop condition
        self.stop_condition = stop_condition

        # Attribute a name if not provided
        self.id = NetworkManager().number_populations(net_id=0)
        self.class_name = 'pop'+str(self.id)

        if name:
            self.name = name
            "Name of the population"
        else:
            self.name = self.class_name

        # Sanity check: population names should be unique
        if self.name in NetworkManager().get_population_names(net_id=0) and not copied:
            Messages._error("Population name='"+self.name+"' is already used.")

        # Add the population to the global variable
        NetworkManager().add_population(0, self)

        # Get a list of parameters and variables
        self.parameters = []
        "List of parameter names."
        self.variables = []
        "List of variable names."
        for param in self.neuron_type.description['parameters']:
            self.parameters.append(param['name'])
        for var in self.neuron_type.description['variables']:
            self.variables.append(var['name'])
        self.attributes = self.parameters + self.variables
        "List of attribute names."

        # Get a list of user-defined functions
        self.functions = [func['name'] for func in self.neuron_type.description['functions']]
        "List of declared functions."

        # Store initial values
        self.init = {}
        for param in self.neuron_type.description['parameters']:
            self.init[param['name']] = param['init']
        for var in self.neuron_type.description['variables']:
            self.init[var['name']] = var['init']

        # List of targets actually connected
        self.targets = []
        "List of connected targets."

        # List of global operations needed by connected projections
        self.global_operations = []

        # Maximum delay of connected projections
        self.max_delay = 0

        # Spiking neurons: do they have to compute an average?
        self._compute_mean_fr = -1.

        # Finalize initialization
        self.initialized = False
        self.cyInstance = None
        self.enabled = True

        # Rank <-> Coordinates methods
        # for the one till three dimensional case we use cython optimized functions.
        from ANNarchy.cython_ext import Coordinates
        if self.dimension==1:
            self._rank_from_coord = Coordinates.get_rank_from_1d_coord
            self._coord_from_rank = Coordinates.get_1d_coord
        elif self.dimension==2:
            self._rank_from_coord = Coordinates.get_rank_from_2d_coord
            self._coord_from_rank = Coordinates.get_2d_coord
        elif self.dimension==3:
            self._rank_from_coord = Coordinates.get_rank_from_3d_coord
            self._coord_from_rank = Coordinates.get_3d_coord
        else:
            self._rank_from_coord = np.ravel_multi_index
            self._coord_from_rank = np.unravel_index

        self._norm_coord_dict = {
            1: Coordinates.get_normalized_1d_coord,
            2: Coordinates.get_normalized_2d_coord,
            3: Coordinates.get_normalized_3d_coord
        }

        # Recorded variables
        self._monitor = None

        # Is overwritten by SpecificPopulations
        self._specific_template = {}

        # Storage order. TODO: why?
        self._storage_order = storage_order

    def _copy(self):
        "Returns a copy of the population when creating networks. Internal use only."
        return Population(geometry=self.geometry, neuron=self.neuron_type, name=self.name, stop_condition=self.stop_condition, storage_order=self._storage_order, copied=True)

    def _generate(self):
        "Overriden by specific populations to generate the code."
        pass

    def _instantiate(self, module):
        """
        Instantiates the population after compilation of the C++ simulation core.
        The function should solely called by Compiler._instantiate().

        :param module: cython module (ANNarchyCore instance)
        """
        if Profiler().enabled:
            import time
            t1 = time.time()

        try:
            self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)
        except:
            Messages._error('unable to instantiate the population', self.name)

        if Profiler().enabled:
            t2 = time.time()
            Profiler().add_entry(t1, t2, "pop"+str(self.id), "instantiate")

    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        # Initialize the population
        self.initialized = True

        # Transfer the initial values of all attributes
        for name, value in self.init.items():
            if isinstance(value, Constant):
                self.__setattr__(name, value.value)
            else:
                self.__setattr__(name, value)


        # Activate the population
        self.cyInstance.activate(self.enabled)

        # Reset to generate the right structures
        self.cyInstance.reset()

        # If the spike population has a refractory period:
        if self.neuron_type.type == 'spike' and self.neuron_type.description['refractory']:
            if not isinstance(self.neuron_type.description['refractory'], str): # the variable will be used directly
                self.refractory = self.neuron_type.description['refractory']

        # Spiking neurons can compute a mean FR
        if self.neuron_type.type == 'spike':
            getattr(self.cyInstance, 'compute_firing_rate')(self._compute_mean_fr)

    def size_in_bytes(self) -> int:
        """
        Returns the size of allocated memory on the C++ side. Please note that this does not contain monitored data and works only if compile() has been invoked.

        :returns: Size.
        """
        if self.initialized:
            return self.cyInstance.size_in_bytes()
        else:
            return 0

    def _clear(self):
        """
        Deallocates container within the C++ instance. The population object is not usable anymore after calling this function.

        Warning: should be only called by the net deconstruction ( in context of parallel_run() ).
        """
        if self.initialized:
            self.cyInstance.clear()
            self.initialized = False

    def reset(self, attributes:list = None)  -> None:
        """
        Resets all parameters and variables of the population to the value they had before the call to compile().

        :param attributes: list of attributes (parameter or variable) which should be reinitialized. Default: all attributes.
        """
        if attributes is None:
            try:
                self.set(self.init)
            except Exception as e:
                Messages._print(e)
                Messages._error("Population.reset(): something went wrong while resetting.")
        else: # only some of them
            for var in attributes:
                # check it exists
                if not var in self.attributes:
                    Messages._warning("Population.reset():", var, "is not an attribute of the population, skipping.")
                    continue

                try:
                    self.__setattr__(var, self.init[var])
                except Exception as e:
                    Messages._print(e)
                    Messages._warning("Population.reset(): something went wrong while resetting", var)

        self.cyInstance.activate(self.enabled)
        self.cyInstance.reset()

    def clear(self) -> None:
        """
        Clears all spiking events previously emitted (history of spikes, delayed spikes).

        Can be useful if you do not want to totally reset a population (i.e. all variables), only to clear the spiking history between two trials.

        Note: does nothing for rate-coded networks.
        """
        self.cyInstance.reset()

    def enable(self) -> None:
        """
        (Re)-enables computations in this population, after they were disabled by the `disable()` method.

        The status of the population is accessible through the `enabled` flag.
        """
        if self.initialized:
            self.cyInstance.activate(True)
        self.enabled = True

    def disable(self) -> None:
        """
        Temporarily disables computations in this population (including the projections leading to it).

        You can re-enable it with the `enable()` method.
        """
        if self.initialized:
            self.cyInstance.activate(False)
        self.enabled = False

    def __getattr__(self, name):
        # Method called when accessing an attribute.
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif name == 'spike':
            if not self.initialized:
                Messages._error("Accessing spike events is only valid after compile()")
            else:
                return self._get_cython_attribute(name)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if self.initialized: # access after compile()
                    return self._get_cython_attribute(name)
                else: # access before compile()
                    if name in self.neuron_type.description['local']:
                        if isinstance(self.init[name], np.ndarray):
                            return self.init[name]
                        else:
                            return np.array([self.init[name]] * self.size).reshape(self.geometry)
                    else:
                        return self.init[name]
            elif name in self.functions:
                return self._function(name)
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        # Method called when setting an attribute.
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            object.__setattr__(self, name, value)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
                    if isinstance(value, RandomDistribution): # Make sure it is generated only once
                        self.init[name] = np.array(value.get_values(self.size)).reshape(self.geometry)
                    else:
                        self.init[name] = value
                else:
                    self._set_cython_attribute(name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def _get_cython_attribute(self, attribute):
        """
        Returns the value of the given attribute for all neurons in the population,
        as a Numpy array having the same geometry as the population if it is local.

        :param attribute: should be a string representing the variables's name.
        """
        try:
            if attribute == 'spike':
                return self.cyInstance.get_local_attribute_all('spiked', 'int')
            else:
                ctype = self._get_attribute_cpp_type(attribute)
                if attribute in self.neuron_type.description['local']:
                    data = self.cyInstance.get_local_attribute_all(attribute, ctype)
                    return data.reshape(self.geometry)
                else:
                    return self.cyInstance.get_global_attribute(attribute, ctype)
        except Exception as e:
            Messages._print(e)
            Messages._error(' the variable ' +  attribute +  ' does not exist in this population (' + self.name + ')')

    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all neurons in the population,
        as a Numpy array having the same geometry as the population if it is local.

        :param attribute: should be a string representing the variables's name.
        :param value: a value or Numpy array of the right size.
        """
        try:
            ctype = self._get_attribute_cpp_type(attribute)
            if attribute in self.neuron_type.description['local']:
                if isinstance(value, np.ndarray):
                    self.cyInstance.set_local_attribute_all(attribute, value.reshape(self.size), ctype)
                elif isinstance(value, list):
                    self.cyInstance.set_local_attribute_all(attribute, np.array(value).reshape(self.size), ctype)
                else:
                    self.cyInstance.set_local_attribute_all(attribute, value * np.ones( self.size ), ctype)
            else:
                self.cyInstance.set_global_attribute(attribute, value, ctype)
        except Exception as e:
            Messages._debug(e)
            err_msg = """Population.set(): either the variable '%(attr)s' does not exist in the population '%(pop)s', or the provided array does not have the right size."""
            Messages._error(err_msg  % { 'attr': attribute, 'pop': self.name } )

    def _get_attribute_cpp_type(self, attribute):
        """
        Determines C++ data type for a given attribute
        """
        ctype = None
        for var in self.neuron_type.description['variables']+self.neuron_type.description['parameters']:
            if var['name'] == attribute:
                ctype = var['ctype']
                break

        return ctype

    def __len__(self):
        # Number of neurons in the population.
        return self.size


    def set(self, values:dict) -> None:
        """
        Sets the value of neural variables and parameters.

        Example:

        ```python
        pop.set({'tau': 20.0, 'r': np.random.rand((8,8)) } )
        ```

        :param values: dictionary of attributes to be updated.
        """
        for name, value in values.items():
            self.__setattr__(name, value)

    def get(self, name:str) -> np.ndarray:
        """
        Returns the value of neural variables and parameters.

        :param name: attribute name as a string.
        """
        return self.__getattr__(name)



    ################################
    ## Access to functions
    ################################
    def _function(self, func):
        "Access a user defined function"
        if not self.initialized:
            Messages._warning('the network is not compiled yet, cannot access the function ' + func)
            return

        return getattr(self.cyInstance, func)

    ################################
    ## Access to weighted sums
    ################################
    def sum(self, target:str):
        """
        Returns the array of weighted sums corresponding to the target:

        ```python
        excitatory = pop.sum('exc')
        ```

        For spiking networks, this is equivalent to accessing the conductances directly:

        ```python
        excitatory = pop.g_exc
        ```

        If no incoming projection has the given target, the method returns zeros.

        **Note:** it is not possible to distinguish the original population when the same target is used.

        :param target: the desired projection target.
        """
        # Check if the network is initialized
        if not self.initialized:
            Messages._warning('sum(): the population', self.name, 'is not initialized yet.')
            return np.zeros(self.geometry)
        # Check if a projection has this type
        if not target in self.targets:
            Messages._warning('sum(): the population', self.name, 'receives no projection with the target', target)
            return np.zeros(self.geometry)
        # Spiking neurons already have conductances available
        if self.neuron_type.type == 'spike':
            return getattr(self, 'g_'+target)
        # Otherwise, call the Cython method
        return self.cyInstance.get_local_attribute_all("_sum_"+target, get_global_config('precision'))

    ################################
    ## Refractory period
    ################################
    @property
    def refractory(self) -> float | str:
        "Refractory period (in ms)."
        if self.neuron_type.description['type'] == 'spike':
            if self.initialized:
                if not isinstance(self.neuron_type.description['refractory'], str):
                    return get_global_config('dt')*self.cyInstance.get_refractory()
                else:
                    return getattr(self, self.neuron_type.description['refractory'])
            else :
                return self.neuron_type.description['refractory']
        else:
            Messages._warning('Rate-coded neurons do not have refractory periods...')
            return None

    @refractory.setter
    def refractory(self, value:float | str):
        if self.neuron_type.description['type'] == 'spike':

            if isinstance(self.neuron_type.description['refractory'], str):
                Messages._warning("The refractory period is linked to the neural variable", self.neuron_type.description['refractory'], ", doing nothing... Change its value instead.")
                return

            if self.initialized:
                if isinstance(value, RandomDistribution):
                    refs = (value.get_values(self.size)/get_global_config('dt')).astype(int)
                elif isinstance(value, np.ndarray):
                    refs = (value / get_global_config('dt')).astype(int).reshape(self.size)
                else:
                    refs = (value/ get_global_config('dt')*np.ones(self.size)).astype(int)
                # TODO cast into int
                self.cyInstance.set_refractory(refs)
            else: # not initialized yet, saving for later
                self.neuron_type.description['refractory'] = value
        else:
            Messages._warning('Rate-coded neurons do not have refractory periods...')

    ################################
    ## Spiking neurons can compute a mean FR
    ################################
    def compute_firing_rate(self, window:float) -> None:
        """
        Tells spiking neurons in the population to compute their mean firing rate over the given window and store the values in the variable `r`.

        This method has an effect on spiking neurons only.

        If this method is not called, `r` will always be 0.0. `r` can of course be accessed and recorded as any other variable.

        :param window: window in ms over which the spikes will be counted.
        """
        if _check_paradigm('cuda'):
            Messages._warning('compute_firing_rate() is currently being evaluated on the host-side, so may be slow ... ')

        if self.neuron_type.type == 'rate':
            Messages._error('compute_firing_rate(): the neuron is already rate-coded...')

        self._compute_mean_fr = float(window)

        if self.initialized:
            getattr(self.cyInstance, 'compute_firing_rate')(self._compute_mean_fr)

    ################################
    ## Access to individual neurons
    ################################
    def neuron(self, *coord) -> IndividualNeuron:
        """
        Returns an ``IndividualNeuron`` object wrapping the neuron with the provided rank or coordinates.

        :returns: IndividualNeuron instance.
        """
        # Transform arguments
        if len(coord) == 1:
            if isinstance(coord[0], int):
                rank = coord[0]
                if not rank < self.size:
                    Messages._error(' when accessing neuron', str(rank), ': the population', self.name, 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
            else:
                rank = self.rank_from_coordinates( coord[0] )
                if rank is None:
                    return None
        else: # a tuple
            rank = self.rank_from_coordinates( coord )
            if rank is None:
                return None

        # Return corresponding neuron
        return IndividualNeuron(self, rank)

    @property
    def neurons(self) -> Iterator[IndividualNeuron]:
        """
        Returns iteratively each neuron in the population.

        For instance, if you want to iterate over all neurons of a population:

        ```python
        for neuron in pop.neurons:
            neuron.r = 0.0
        ```

        Alternatively, one could also benefit from the ``__iter__`` special command. 
        The following code is equivalent:

        ```python
        for neuron in pop:
            neuron.r = 0.0
        ```
        """
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank)

    # Iterators
    def __getitem__(self, *args, **kwds):
        """ 
        Returns neurons froms coordinates in the population.

        If only one argument is given, it is interpeted as a rank and returns a single neuron.

        If slices are given, it returns a PopulationView object.
        """
        indices =  args[0]
        try:
            if np.issubdtype(indices, int):
                indices = int(indices)
        except:
            pass

        if isinstance(indices, int): # a single neuron
            return PopulationView(self, ranks=np.array([int(indices)]), geometry=(1,))

        elif isinstance(indices, (list)):
            indices = np.array(indices)
            return PopulationView(self, indices, geometry=(len(indices),))

        elif isinstance(indices, (np.ndarray)):
            # Sanity check
            if isinstance(indices, (np.ndarray)):
                if indices.ndim != 1:
                    Messages._error('only one-dimensional lists/arrays are allowed to address a population.')

            return PopulationView(self, indices, geometry=(len(indices),))

        elif isinstance(indices, slice): # a single slice of ranks
            start, stop, step = indices.start, indices.stop, indices.step

            # no value defined for a position
            if indices.start is None:
                start = 0
            if indices.stop is None:
                stop = self.size
            if indices.step is None:
                step = 1

            # start or end arguments are negative
            if start < 0:
                start = self.size + start
            if stop < 0:
                stop = self.size + stop

            # generate a new set of indices
            rk_range = np.arange(start, stop, step, dtype="int32")
            return PopulationView(self, rk_range, geometry=(len(rk_range),))

        elif isinstance(indices, tuple): # a tuple
            slices = False
            for idx in indices: # check if there are slices in the coordinates
                if isinstance(idx, slice): # there is at least one
                    slices = True
            if not slices: # return one neuron
                return self.neuron(indices)
            else: # Compute a list of ranks from the slices
                coords = []
                # Expand the slices
                for rank in range(len(indices)):
                    idx = indices[rank]
                    if isinstance(idx, int): # no slice
                        coords.append([idx])
                    elif isinstance(idx, slice): # slice
                        start, stop, step = idx.start, idx.stop, idx.step
                        if idx.start is None:
                            start = 0
                        if idx.stop is None:
                            stop = self.geometry[rank]
                        if idx.step is None:
                            step = 1
                        rk_range = list(range(start, stop, step))
                        coords.append(rk_range)
                # Generate all ranks from the indices
                if self.dimension == 2:
                    ranks = [self.rank_from_coordinates((x, y)) for x in coords[0] for y in coords[1]]
                    geometry = (len(coords[0]), len(coords[1]))
                elif self.dimension == 3:
                    ranks = [self.rank_from_coordinates((x, y, z)) for x in coords[0] for y in coords[1] for z in coords[2]]
                    geometry = (len(coords[0]), len(coords[1]), len(coords[2]))
                elif self.dimension == 4:
                    ranks = [self.rank_from_coordinates((x, y, z, k)) for x in coords[0] for y in coords[1] for z in coords[2] for k in coords[3]]
                    geometry = (len(coords[0]), len(coords[1]), len(coords[2]), len(coords[3]))
                else:
                    Messages._error("Slicing is implemented only for population with 4 dimensions at maximum", self.geometry)
                if not max(ranks) < self.size:
                    Messages._error("Indices do not match the geometry of the population", self.geometry)
                
                return PopulationView(self, ranks, geometry=geometry)

        Messages._warning('Population' + self.name + ': can not address the population with', indices)
        return None

    def __iter__(self) -> Iterator[IndividualNeuron]:
        # Returns iteratively each neuron in the population in ascending rank order.
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank)

    ################################
    ## Coordinate transformations
    ################################
    def rank_from_coordinates(self, coord:tuple) -> int:
        """
        Returns the rank of a neuron based on coordinates.

        :param coord: Coordinate tuple, can be multidimensional.
        :returns: Rank.
        """
        try:
            rank = self._rank_from_coord( coord, self.geometry )
        except:
            Messages._error('rank_from_coordinates(): There is no neuron of coordinates', coord, 'in the population', self.name, self.geometry)

        if rank > self.size:
            Messages._error('rank_from_coordinates(), neuron', str(coord), ': the population' , self.name , 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
        else:
            return rank

    def coordinates_from_rank(self, rank:int) -> tuple:
        """
        Returns the coordinates of a neuron based on its rank.

        :param rank: Rank of the neuron.
        :returns: Coordinates.
        """
        # Check the rank
        if not rank < self.size:
            Messages._error('The given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')

        try:
            coord = self._coord_from_rank( rank, self.geometry )
        except:
            Messages._error('The given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')
        else:
            return coord

    def normalized_coordinates_from_rank(self, rank:int, norm:float=1.) -> tuple:
        """
        Returns normalized coordinates of a neuron based on its rank. 
        The geometry of the population is mapped to the hypercube $[0, 1]^d$

        :param rank: Rank of the neuron
        :param norm: Norm of the cube (default = 1.0)
        :returns: Coordinates.
        """
        try:
            normal = self._norm_coord_dict[self.dimension](rank, self.geometry)
        except KeyError:
            coord = self.coordinates_from_rank(rank)

            normal = tuple()
            for dim in range(self.dimension):
                if self._geometry[dim] > 1:
                    normal += ( norm * float(coord[dim])/float(self.geometry[dim]-1), )
                else:
                    normal += (float(rank)/(float(self.size)-1.0),) # default?

        return normal


    ################################
    ## Save/load methods
    ################################
    def _data(self):
        "Returns a dictionary containing all information about the population. Used for saving."
        desc = {}
        desc['name'] = self.name
        desc['geometry'] = self.geometry
        desc['size'] = self.size
        # Attributes
        desc['attributes'] = self.attributes
        desc['parameters'] = self.parameters
        desc['variables'] = self.variables
        # Save all attributes
        for var in self.attributes:
            try:
                ctype = self._get_attribute_cpp_type(var)
                if var in self.neuron_type.description['local']:
                    data = self.cyInstance.get_local_attribute_all(var, ctype)
                    desc[var] = data.reshape(self.geometry)
                else:
                    desc[var] = self.cyInstance.get_global_attribute(var, ctype)

            except:
                Messages._warning('Can not save the attribute ' + var + ' in the population ' + self.name + '.')

        return desc

    def save(self, filename:str) -> None:
        """
        Saves all information about the population (structure, current value of parameters and variables) into a file.

        * If the file name is '.npz', the data will be saved and compressed using `np.savez_compressed` (recommended).

        * If the file name ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * If the file name is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * Otherwise, the data will be pickled into a simple binary text file using pickle.

        **Warning:** The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

        Example:

        ```python
        pop.save('pop1.npz')
        pop.save('pop1.txt')
        pop.save('pop1.txt.gz')
        pop.save('pop1.mat')
        ```

        :param filename: Filename, may contain relative or absolute path.
        """
        from ANNarchy.core.IO import _save_data
        _save_data(filename, self._data())


    def load(self, filename, pickle_encoding=None) -> None:
        """
        Load the saved state of the population by `Population.save()`.

        Warning: Matlab data can not be loaded.

        Example:

        ```python
        pop.load('pop1.npz')
        pop.load('pop1.txt')
        pop.load('pop1.txt.gz')
        ```

        :param filename: the filename with relative or absolute path.
        """
        from ANNarchy.core.IO import _load_data
        self._load_pop_data(_load_data(filename, pickle_encoding))

    def _load_pop_data(self, desc):
        """
        Updates the population with the stored data set.
        """
        if not 'attributes' in desc.keys():
            Messages._error('Saved with a too old version of ANNarchy (< 4.2).', exit=True)

        for var in desc['attributes']:
            try:
                self._set_cython_attribute(var, desc[var])

            except Exception as e:
                Messages._print(e)
                Messages._warning('Can not load the variable ' + var + ' in the population ' + self.name)
                Messages._print('Skipping this variable.')
                continue
