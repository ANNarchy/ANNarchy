"""

    Population.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import ANNarchy.core.Global as Global

from .PopulationView import PopulationView
from .Random import RandomDistribution
from .Neuron import IndividualNeuron

import numpy as np
import copy, inspect

class Population(object):
    """
    Represents a population of homogeneous neurons.
    """

    def __init__(self, geometry, neuron, name=None, stop_condition=None):
        """
        *Parameters*:

            * **geometry**: population geometry as tuple. If an integer is given, it is the size of the population.

            * **neuron**: instance of ``ANNarchy.Neuron``

            * **name**: unique name of the population (optional).

            * **stop_condition**: a single condition on a neural variable which can stop the simulation whenever it is true.

        """
        # Store the provided geometry
        # automatically defines w, h, d, size
        if isinstance(geometry, (int, float)):
            # 1D
            self.geometry = (int(geometry), )
            self.width = int(geometry)
            self.height = int(1)
            self.depth = int(1)
            self.dimension = int(1)
        else:
            # a tuple is given, can be 1 .. N dimensional
            self.geometry = geometry
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

        # Compute the size
        size = int(1)
        for i in range(len(self.geometry)):
            size *= int(self.geometry[i])
        self.size = int(size)
        self.ranks = list(range(self.size))

        # Store the neuron type
        if inspect.isclass(neuron):
            self.neuron_type = neuron()
        else:
            self.neuron_type = copy.deepcopy(neuron)
        self.neuron_type._analyse()

        # Store the stop condition
        self.stop_condition = stop_condition

        # Attribute a name if not provided
        self.id = len(Global._network[0]['populations'])
        self.class_name = 'pop'+str(self.id)

        if name:
            self.name = name
        else:
            self.name = self.class_name

        # Add the population to the global variable
        Global._network[0]['populations'].append(self)

        # Get a list of parameters and variables
        self.parameters = []
        self.variables = []
        for param in self.neuron_type.description['parameters']:
            self.parameters.append(param['name'])
        for var in self.neuron_type.description['variables']:
            self.variables.append(var['name'])
        self.attributes = self.parameters + self.variables

        # Store initial values
        self.init = {}
        for param in self.neuron_type.description['parameters']:
            self.init[param['name']] = param['init']
        for var in self.neuron_type.description['variables']:
            self.init[var['name']] = var['init']

        # List of targets actually connected
        self.targets = []

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
        import ANNarchy.core.cython_ext.Coordinates as Coordinates
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

    def _generate(self):
        "Overriden by specific populations to generate the code."
        pass

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size)


    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        self.initialized = True
        self.set(self.init)
        self.cyInstance.activate(self.enabled)
        self.cyInstance.reset()

        # If the spike population has a refractory period:
        if self.neuron_type.type == 'spike' and self.neuron_type.description['refractory']:
            if isinstance(self.neuron_type.description['refractory'], str): # a global variable
                try:
                    self.refractory = eval('self.'+self.neuron_type.description['refractory'])
                except Exception as e:
                    Global._print(e, self.neuron_type.description['refractory'])
                    Global._error('The initialization for the refractory period is not valid.')

            else: # a value
                self.refractory = self.neuron_type.description['refractory']

    def reset(self, attributes=-1):
        """
        Resets all parameters and variables of the population to the value they had before the call to compile().

        *Parameters:*

        * **attributes**: list of attributes (parameter or variable) which should be reinitialized. Default: all attributes.
        """
        if attributes == -1:
            self.set(self.init)
        else: # only some of them
            for var in attributes:
                # check it exists
                if not var in self.attributes:
                    _warning("Population.reset():", var, "is not an attribute of the population, won't reset.")
                    continue
                try:
                    self.__setattr__(var, self.init[var])
                except Exception as e:
                    _print(e)
                    _warning("Population.reset(): something went wrong while resetting", var)

        self.cyInstance.activate(self.enabled)
        self.cyInstance.reset()

    def clear(self):
        """
        Clears all spiking events previously emitted (history of spikes, delayed spikes).

        Can be useful if you do not want to totally reset a population (i.e. all variables), only to clear the spiking history between two trials.

        Note: does nothing for rate-coded networks.
        """
        self.cyInstance.reset()

    def enable(self):
        """
        (Re)-enables computations in this population, after they were disabled by the ``disable()`` method.

        The status of the population is accessible through the ``enabled`` flag.
        """
        if self.initialized:
            self.cyInstance.activate(True)
        self.enabled = True

    def disable(self):
        """
        Temporarily disables computations in this population (including the projections leading to it).

        You can re-enable it with the ``enable()`` method.
        """
        if self.initialized:
            self.cyInstance.activate(False)
        self.enabled = False

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
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
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        " Method called when setting an attribute."
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
        as a NumPy array having the same geometry as the population if it is local.

        Parameter:

        * *attribute*: should be a string representing the variables's name.

        """
        try:
            if attribute in self.neuron_type.description['local']:
                return getattr(self.cyInstance, 'get_'+attribute)().reshape(self.geometry)
            else:
                return getattr(self.cyInstance, 'get_'+attribute)()
        except Exception as e:
            Global._print(e)
            Global._error(' the variable ' +  attribute +  ' does not exist in this population (' + self.name + ')')

    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all neurons in the population,
        as a NumPy array having the same geometry as the population if it is local.

        Parameter:

        * *attribute*: should be a string representing the variables's name.
        * *value*: a value or Numpy array of the right size.

        """
        try:
            if attribute in self.neuron_type.description['local']:
                if isinstance(value, np.ndarray):
                    getattr(self.cyInstance, 'set_'+attribute)(value.reshape(self.size))
                elif isinstance(value, list):
                    getattr(self.cyInstance, 'set_'+attribute)(np.array(value).reshape(self.size))
                else:
                    getattr(self.cyInstance, 'set_'+attribute)(value * np.ones( self.size ))
            else:
                getattr(self.cyInstance, 'set_'+attribute)(value)
        except Exception as e:
            Global._debug(e)
            err_msg = """either the variable '%(attr)s' does not exist in the population '%(pop)s', or the provided array does not have the right size."""
            Global._error(err_msg  % { 'attr': attribute, 'pop': self.name } )

    def __len__(self):
        """
        Number of neurons in the population.
        """
        return self.size


    def set(self, values):
        """
        Sets the value of neural variables and parameters.

        *Parameter*:

        * **values**: dictionary of attributes to be updated.

        .. code-block:: python

            set({ 'tau' : 20.0, 'r'= np.random.rand((8,8)) } )
        """
        for name, value in values.items():
            self.__setattr__(name, value)

    def get(self, name):
        """
        Returns the value of neural variables and parameters.

        *Parameter*:

        * **name**: attribute name as a string.
        """
        return self.__getattr__(name)



    ################################
    ## Access to weighted sums
    ################################
    def sum(self, target):
        """
        Returns the array of weighted sums corresponding to the target::

            excitatory = pop.sum('exc')

        For spiking networks, this is equivalent to accessing the conductances directly::

            excitatory = pop.g_exc

        If no incoming projection has the given target, the method returns zeros.

        *Parameter:*

        * **target**: the desired projection target.

        **Note:** it is not possible to distinguish the original population when the same target is used.
        """
        # Check if the network is initialized
        if not self.initialized:
            Global._warning('sum(): the population', self.name, 'is not initialized yet.')
            return np.zeros(self.geometry)
        # Check if a projection has this type
        if not target in self.targets:
            Global._warning('sum(): the population', self.name, 'receives no projection with the target', target)
            return np.zeros(self.geometry)
        # Spiking neurons already have conductances available
        if self.neuron_type.type == 'spike':
            return getattr(self, 'g_'+target)
        # Otherwise, call the Cython method
        return getattr(self.cyInstance, 'get_sum_'+target)()

    ################################
    ## Refractory period
    ################################
    @property
    def refractory(self):
        if self.neuron_type.description['type'] == 'spike':
            if self.initialized:
                return Global.config['dt']*self.cyInstance.get_refractory()
            else :
                return self.neuron_type.description['refractory']
        else:
            Global._warning('rate-coded neurons do not have refractory periods...')
            return None

    @refractory.setter
    def refractory(self, value):
        if self.neuron_type.description['type'] == 'spike':
            if self.initialized:
                if isinstance(value, RandomDistribution):
                    refs = (value.get_values(self.size)/Global.config['dt']).astype(int)
                elif isinstance(value, np.ndarray):
                    refs = (value / Global.config['dt']).astype(int).reshape(self.size)
                else:
                    refs = (value/ Global.config['dt']*np.ones(self.size)).astype(int)
                # TODO cast into int
                self.cyInstance.set_refractory(refs)
            else: # not initialized yet, saving for later
                self.neuron_type.description['refractory'] = value
        else:
            Global._warning('rate-coded neurons do not have refractory periods...')

    ################################
    ## Spiking neurons can compute a mean FR
    ################################
    def compute_firing_rate(self, window):
        """
        Tells spiking neurons in the population to compute their mean firing rate over the given window and store the values in the variable `r`.

        This method has an impact on spiking neurons only.

        **Warning:** this method must be called before compile() and the window can not be changed afterwards.

        If this method is not called, `r` will always be 0.0. `r` can of course be accessed and recorded as any other variable.

        *Parameter*:

        * **window**: window in ms over which the spikes will be counted.
        """
        if self.neuron_type.type == 'rate':
            _warning('compute_firing_rate(): the neuron is already rate-coded...')
        else:
            self._compute_mean_fr = float(window)

    ################################
    ## Access to individual neurons
    ################################
    def neuron(self, *coord):
        """
        Returns an ``IndividualNeuron`` object wrapping the neuron with the provided rank or coordinates.
        """
        # Transform arguments
        if len(coord) == 1:
            if isinstance(coord[0], int):
                rank = coord[0]
                if not rank < self.size:
                    Global._error(' when accessing neuron', str(rank), ': the population', self.name, 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
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
    def neurons(self):
        """ Returns iteratively each neuron in the population.

        For instance, if you want to iterate over all neurons of a population:

        >>> for neur in pop.neurons:
        ...     neur.r = 0.0

        Alternatively, one could also benefit from the ``__iter__`` special command. The following code is equivalent:

        >>> for neur in pop:
        ...     neur.r = 0.0
        """
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank)

    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns neuron of coordinates (width, height, depth) in the population.

        If only one argument is given, it is a rank.

        If slices are given, it returns a PopulationView object.
        """
        indices =  args[0]
        try:
            if np.issubdtype(indices, int):
                indices = int(indices)
        except:
            pass
        if isinstance(indices, int): # a single neuron
            return PopulationView(self, [int(indices)])
        elif isinstance(indices, (list, np.ndarray)):
            if isinstance(indices, (np.ndarray)):
                if indices.ndim != 1:
                    Global._error('only one-dimensional lists/arrays are allowed to address a population.')
                indices = list(indices.astype(int))
            return PopulationView(self, list(indices))
        elif isinstance(indices, slice): # a slice of ranks
            start, stop, step = indices.start, indices.stop, indices.step
            if indices.start is None:
                start = 0
            if indices.stop is None:
                stop = self.size
            if indices.step is None:
                step = 1
            rk_range = list(range(start, stop, step))
            return PopulationView(self, rk_range)
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
                elif self.dimension == 3:
                    ranks = [self.rank_from_coordinates((x, y, z)) for x in coords[0] for y in coords[1] for z in coords[2]]
                elif self.dimension == 4:
                    ranks = [self.rank_from_coordinates((x, y, z, k)) for x in coords[0] for y in coords[1] for z in coords[2] for k in coords[3]]
                else:
                    Global._error("Slicing is implemented only for population with 4 dimensions at maximum", self.geometry)
                if not max(ranks) < self.size:
                    Global._error("Indices do not match the geometry of the population", self.geometry)
                return PopulationView(self, ranks)

        Global._warning('Population' + self.name + ': can not address the population with', indices)
        return None

    def __iter__(self):
        " Returns iteratively each neuron in the population in ascending rank order."
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank)

    ################################
    ## Coordinate transformations
    ################################
    def rank_from_coordinates(self, coord):
        """
        Returns the rank of a neuron based on coordinates.

        *Parameter*:

            * **coord**: coordinate tuple, can be multidimensional.
        """
        try:
            rank = self._rank_from_coord( coord, self.geometry )
        except:
            Global._error('rank_from_coordinates(): There is no neuron of coordinates', coord, 'in the population', self.name, self.geometry)

        if rank > self.size:
            Global._error('rank_from_coordinates(), neuron', str(coord), ': the population' , self.name , 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
        else:
            return rank

    def coordinates_from_rank(self, rank):
        """
        Returns the coordinates of a neuron based on its rank.

        *Parameter*:

            * **rank**: rank of the neuron.
        """
        # Check the rank
        if not rank < self.size:
            Global._error('The given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')

        try:
            coord = self._coord_from_rank( rank, self.geometry )
        except:
            Global._error('The given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')
        else:
            return coord

    def normalized_coordinates_from_rank(self, rank, norm=1.):
        """
        Returns normalized coordinates of a neuron based on its rank. The geometry of the population is mapped to the hypercube [0, 1]^d.

        *Parameters*:

        * **rank**: rank of the neuron
        * **norm**: norm of the cube (default = 1.0)

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
    ## Recording
    ################################
    def start_record(self, variable, period = None, ranks='all'):
        """
        **Deprecated!!**

        Start recording neural variables.

        Parameter:

            * **variable**: single variable name or list of variable names.

            * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable.

            * **ranks**: list of ranks of the neurons to record (default: 'all').

        Example::

            pop1.start_record('r')
            pop2.start_record(['mp', 'r'], period=10.0)
            pop3.start_record(['spike'])
            pop4.start_record(['r'], ranks=range(10, 100))
        """
        Global._warning("recording from a Population is deprecated, use a Monitor instead.")
        from .Record import Monitor
        if ranks == 'all':
            self._monitor = Monitor(self, variable, period=period)
        else:
            self._monitor = Monitor(PopulationView(self, ranks), variable, period=period)

    def stop_record(self):
        """
        **Deprecated!!**

        Stops recording all the previously defined variables.

        Example::

            pop1.stop_record()
        """
        Global._warning("recording from a Population is deprecated, use a Monitor instead.")
        if self._monitor:
            self._monitor.stop()
            self._monitor = None

    def pause_record(self, variable=None):
        """
        **Deprecated!!**

        Pauses the recording of variables (can be resumed later with resume_record()).

        *Parameter*:

        * **variable**: single variable name or list of variable names. If no argument is provided all recordings will pause.

        Example::

            pop1.pause_record('r')
            pop2.pause_record(['mp', 'r'])
        """
        Global._warning("recording from a Population is deprecated, use a Monitor instead.")
        if self._monitor:
            self._monitor.pause(variable)

    def resume_record(self, variable=None):
        """
        **Deprecated!!**

        Resume recording the previous defined variables.

        *Parameter*:

            * **variable**: single variable name or list of variable names.

        Example::

            pop1.resume_record('r')
            pop2.resume_record(['mp', 'r'])
        """
        Global._warning("recording from a Population is deprecated, use a Monitor instead.")
        if self._monitor:
            self._monitor.resume(variable)

    def get_record(self, variable=None, reshape=False):
        """
        **Deprecated!!**

        Returns the recorded data as a nested dictionary. The first key corresponds to the variable name if several were recorded.

        The second keys correspond to:

        * ``start`` simulations step(s) were recording started.
        * ``stop`` simulations step(s) were recording stopped or paused.
        * ``data`` the recorded data as a numpy array. The last index represents the time, the remaining dimensions are the population size or geometry, depending on the value of ``reshape``.

        .. warning::

            Once get_record is called, the recorded data is internally erased.

        *Parameters*:

        * **variable**: single variable name or list of variable names. If no argument provided, all currently recorded data are returned.
        * **reshape**: by default this functions returns the data as a 2D matrix (number of neurons * time). If *reshape* is set to True, the population data will be reshaped into its geometry (geometry[0], ... , geometry[n], time)
        """
        Global._warning("recording from a Population is deprecated, use a Monitor instead.")
        if not self._monitor:
            Global._error('get_record(): there is currently no recording.')

        if variable:
            if not isinstance(variable, list):
                variables = [variable]
            else:
                variables = variable
        else:
            variables = self._monitor.variables

        if self._monitor:
            var_data = self._monitor.get(variables, force_dict=True)
            var_times = self._monitor.times(variables)
        else:
            return {}

        data = {}

        for var in variables:
            if not reshape:
                data[var] = {
                    'start': var_times[var]['start'] if len(var_times[var]['start']) >1 else var_times[var]['start'][0],
                    'stop' : var_times[var]['stop'] if len(var_times[var]['stop']) >1 else var_times[var]['stop'][0] ,
                    'data' : np.array(var_data[var]).T if not var == 'spike' else var_data[var].values(),
                    'period' : self._monitor.period,
                    'ranks' : self._monitor.ranks
                }
            else:
                if not var == 'spike':
                    mat1 = np.array(var_data).reshape((np.array(var_data).shape[0],)+self.geometry)
                    data[var] = {
                        'start': var_times[var]['start'] if len(var_times[var]['start']) >1 else var_times[var]['start'][0],
                        'stop' : var_times[var]['stop'] if len(var_times[var]['stop']) >1 else var_times[var]['stop'][0] ,
                        'data' : np.transpose(mat1, tuple( range(1, self.dimension+1)+[0])),
                        'period' : self._monitor.period,
                        'ranks' : self._monitor.ranks
                        }
                else:
                    Global._error("reshape=true is invalid for get_record('spike')")

        return data

    ################################
    ## Modification of the variables
    ################################
    def set_variable_flags(self, name, value):
        """ Sets the flags of a variable for the population.

        If the variable ``r`` is defined in the Neuron description through::

            r = sum(exc) : max=1.0

        one can change its maximum value with::

            pop.set_variable_flags('r', {'max': 2.0})

        For valued flags (init, min, max), ``value`` must be a dictionary containing the flag as key ('init', 'min', 'max') and its value.

        For positional flags (population, implicit), the value in the dictionary must be set to the empty string ''::

            pop.set_variable_flags('r', {'implicit': ''})

        A None value in the dictionary deletes the corresponding flag::

            pop.set_variable_flags('r', {'max': None})

        """
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The population '+self.name+' has no variable called ' + name)

        for key, val in value.items():
            if val == '': # a flag
                try:
                    self.neuron_type.description['variables'][rk_var]['flags'].index(key)
                except: # the flag does not exist yet, we can add it
                    self.neuron_type.description['variables'][rk_var]['flags'].append(key)
            elif val is None: # delete the flag
                try:
                    self.neuron_type.description['variables'][rk_var]['flags'].remove(key)
                except: # the flag did not exist, check if it is a bound
                    if has_key(self.neuron_type.description['variables'][rk_var]['bounds'], key):
                        self.neuron_type.description['variables'][rk_var]['bounds'].pop(key)
            else: # new value for init, min, max...
                if key == 'init':
                    self.neuron_type.description['variables'][rk_var]['init'] = val
                    self.init[name] = val
                else:
                    self.neuron_type.description['variables'][rk_var]['bounds'][key] = val



    def set_variable_equation(self, name, equation):
        """ Changes the equation of a variable for the population.

        If the variable ``r`` is defined in the Neuron description through::

            tau * dr/dt + r  = sum(exc) : max=1.0

        one can change the equation with::

            pop.set_variable_equation('r', 'r = sum(exc)')

        Only the equation should be provided, the flags have to be changed with ``set_variable_flags()``.

        .. warning::

            This method should be used with great care, it is advised to define another Neuron object instead.

        """
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The population '+self.name+' has no variable called ' + name)

        self.neuron_type.description['variables'][rk_var]['eq'] = equation


    def _find_variable_index(self, name):
        " Returns the index of the variable name in self.description['variables']"
        for idx in range(len(self.neuron_type.description['variables'])):
            if self.neuron_type.description['variables'][idx]['name'] == name:
                return idx
        return -1

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
                desc[var] = getattr(self.cyInstance, 'get_'+var)()
            except:
                Global._warning('Can not save the attribute ' + var + 'in the population ' + self.name + '.')

        return desc

    def save(self, filename):
        """
        Saves all information about the population (structure, current value of parameters and variables) into a file.

        * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * Otherwise, the data will be pickled into a simple binary text file using pickle.

        *Parameter*:

        * **filename**: filename, may contain relative or absolute path.

            .. warning::

                The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

        Example::

            pop.save('pop1.txt')

        """
        from ANNarchy.core.IO import _save_data
        _save_data(filename, self._data())


    def load(self, filename):
        """
        Load the saved state of the population.

        Warning: Matlab data can not be loaded.

        *Parameters*:

        * **filename**: the filename with relative or absolute path.

        Example::

            pop.load('pop1.txt')

        """
        from ANNarchy.core.IO import _load_data, _load_pop_data
        _load_pop_data(self, _load_data(filename))
