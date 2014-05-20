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
from ANNarchy.parser.Analyser import analyse_population

from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Neuron import IndividualNeuron

from ANNarchy.core.Record import Record
import ANNarchy.core.Global as Global

import traceback
import numpy as np


class Population(object):
    """
    Represents a population of neurons.
    """

    def __init__(self, geometry, neuron, name=None):
        """
        Constructor of the population.
        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. If an integer is given, it is the size of the population.

            * *neuron*: instance of ``ANNarchy.RateNeuron`` or ``ANNarchy.SpikeNeuron``

            * *name*: unique name of the population (optional).
        
        """
        # Store the provided geometry
        # automatically defines w, h, d, size
        if isinstance(geometry, int): 
            # 1D
            self._geometry = (geometry, )
            self._width = geometry
            self._height = 1
            self._depth = 1
            self._dimension = 1
        else: 
            # a tuple is given, can be 1 .. N dimensional
            self._geometry = geometry
            self._width = geometry[0]
            if len(geometry)>=2:
                self._height = geometry[1]
            else:
                self._height = 1
            if len(geometry)>=3:                
                self._depth = geometry[2]
            else:
                self._depth = 1
                
            self._dimension = len(geometry)

        size = 1        
        for i in range(len(self._geometry)):
            size *= self._geometry[i]
            
        self._size = size
        
        # Store the neuron type
        self.neuron_type = neuron
        
        # Attribute a name if not provided
        self._id = len(Global._populations)
        self.class_name = 'Population'+str(self._id)
        
        if name:
            self.name = name
        else:
            self.name = self.class_name
                
        # Add the population to the global variable
        Global._populations.append(self)
        
        # Get a list of parameters and variables
        self.description = analyse_population(self)
        self.parameters = []
        self.variables = []
        for param in self.description['parameters']:
            self.parameters.append(param['name'])
        for var in self.description['variables']:
            self.variables.append(var['name'])
        self.attributes = self.parameters + self.variables
        
        # Store initial values
        self.init = {}
        for param in self.description['parameters']:
            self.init[param['name']] = param['init']
        for var in self.description['variables']:
            self.init[var['name']] = var['init']
        
        # List of targets actually connected
        self.targets = []
        self.sources = []
                
        # Allow recording of variables
        self._recorded_variables = {}        
        for var in self.variables:
            self._recorded_variables[var] = Record(var)
        if self.description['type'] == 'spike':
            self._recorded_variables['spike'] = Record('spike')

        # Finalize initialization
        self.initialized = False

        # Rank <-> Coordinates methods
        # for the one till three dimensional case we use cython optimized functions. 
        import ANNarchy.core.cython_ext.Coordinates as Coordinates
        if self._dimension==1:
            self._rank_from_coord = Coordinates.get_rank_from_1d_coord
            self._coord_from_rank = Coordinates.get_1d_coord
        elif self._dimension==2:
            self._rank_from_coord = Coordinates.get_rank_from_2d_coord
            self._coord_from_rank = Coordinates.get_2d_coord
        elif self._dimension==3:
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
        
    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        self.initialized = True  
        for attr in self.attributes:
            if attr in self.description['local']: # Only local variables are not directly initialized in the C++ code
                if isinstance(self.init[attr], list) or isinstance(self.init[attr], np.ndarray):
                    self._set_cython_attribute(attr, self.init[attr])

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if self.initialized: # access after compile()
                    return self._get_cython_attribute(name)
                else: # access before compile()
                    if name in self.description['local']:
                        if isinstance(self.init[name], np.ndarray):
                            return self.init[name]
                        else:
                            return np.array([self.init[name]] * self.size).reshape(self._geometry)
                    else:
                        return self.init[name]
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            object.__setattr__(self, name, value)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
                    if isinstance(value, RandomDistribution): # Make sure it is generated only once
                        self.init[name] = np.array(value.get_values(self.size)).reshape(self._geometry)
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
            if attribute in self.description['local']:
                return getattr(self.cyInstance, '_get_'+attribute)().reshape(self._geometry)
            else:
                return getattr(self.cyInstance, '_get_'+attribute)()
        except:
            print('Error: attribute', attribute, 'does not exist in this population.')
            print(traceback.print_stack())
        
    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all neurons in the population, 
        as a NumPy array having the same geometry as the population if it is local.
        
        Parameter:
        
        * *attribute*: should be a string representing the variables's name.
        
        """
        try:
            if attribute in self.description['local']:
                if isinstance(value, np.ndarray):
                    getattr(self.cyInstance, '_set_'+attribute)(value.reshape(self.size))
                elif isinstance(value, list):
                    getattr(self.cyInstance, '_set_'+attribute)(np.array(value).reshape(self.size))
                else:
                    getattr(self.cyInstance, '_set_'+attribute)(np.array( [value]*self.size ))
            else:
                getattr(self.cyInstance, '_set_'+attribute)(value)
        except:
            print('Error: variable', attribute, 'does not exist in this population.')
            print(traceback.print_stack())

    @property
    def geometry(self):
        """
        Width of population.
        """
        return self._geometry

    @property
    def width(self):
        """
        Width of population.
        """
        return self._width

    @property
    def height(self):
        """
        Height of population.
        """
        return self._height

    @property
    def depth(self):
        """
        Depth of population.
        """
        return self._depth
        
    @property
    def size(self):
        """
        Number of neurons in the population.
        """
        return self._size
        
    def __len__(self):
        """
        Number of neurons in the population.
        """
        return self.size

    @property
    def dimension(self):
        """
        Dimension of the population (1, 2 or 3)
        """
        return self._dimension
        
    @property 
    def rank(self):
        """
        Unique identifier of the population, e.g. usable on connection patterns.
        """
        return self._id
        
    def reset(self):
        """
        Reset the population variables to their initial values.
        """
        try:
            self.cyInstance.reset()
        except:
            print('reset population', self.name, 'failed.')
        
    def start_record(self, variable):
        """
        Start recording the previous defined variables.
        
        Parameter:
            
            * *variable*: single variable name or list of variable names.        
        """
        _variable = []
        
        if isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        for var in _variable:
            
            if not var in self._recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue

            if not self._recorded_variables[var].is_inited:
                continue
            
            try:
                getattr(self.cyInstance, '_start_record_'+var)()

                if Global.config['verbose']:
                    print('start record of', var, '(', self.name, ')')
                    
                self._recorded_variables[var].start()
            except:
                Global._error('(start_record): the variable ' + var + ' is not recordable.')

    def pause_record(self, variable=None):
        """
        pause recording the previous defined variables.

        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument is provided all records will stop.
        """
        _variable = []
        if variable == None:
            _variable = self._running_recorded_variables
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')       
        
        for var in _variable:
            
            if not var in self._recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue

            if not self._recorded_variables[var].is_running:
                print('record of', var, 'was not running on population', self.name)
                continue
            
            try:
                getattr(self.cyInstance, '_stop_record_'+var)()

                if Global.config['verbose']:
                    print('pause record of', var, '(', self.name, ')')
                self._recorded_variables[var].pause()
            except:
                print("Error (pause_record): only possible after compilation.")

    def resume_record(self, variable):
        """
        Resume recording the previous defined variables.
        
        Parameter:
            
            * *variable*: single variable name or list of variable names.        
        """
        _variable = []
        
        if isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        for var in _variable:
            
            if not var in var in self._recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue
            
            if not self._recorded_variables[var].is_running:
                print('record of', var, 'is already running on population', self.name)
                continue
            
            try:
                getattr(self.cyInstance, '_start_record_'+var)()
                
                if Global.config['verbose']:
                    print('resume record of', var, '(' , self.name, ')')

                self._recorded_variables[var].start()
            except:
                print("Error: only possible after compilation.")
                
    def get_record(self, variable=None, reshape=False):
        """
        Returns the recorded data as one matrix or a dictionary if more then one variable is requested. 
        The last dimension represents the time, the remaining dimensions are the population geometry.
        
        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument provided, the remaining recorded data is returned.  
        * *reshape*: by default this functions returns the data as a 2D matrix (number of neurons, time). If **reshape* is set to True, the population data will be reshaped into its geometry (geometry[0], ... , geometry[n], time)
        """
        
        _variable = []
        if variable == None:
            _variable = self._recorded_variables
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        data_dict = {}
        
        for var in _variable:

            if not var in var in self._recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue
            
            if self._recorded_variables[var].is_running:
                self.pause_record(var)
            
            try:
                if Global.config['verbose']:
                    print('get record of', var, '(', self.name, ')')
                    
                data = getattr(self.cyInstance, '_get_recorded_'+var)()
                
                if var == 'spike':
                    data_dict[var] = { 
                        'data': data,
                        'start': self._recorded_variables[var].start_time,
                        'stop': self._recorded_variables[var].stop_time
                    }

                elif not reshape:
                    #
                    # [ time, data(1D) ] => [ time, data(1D) ] 
                    data_dict[var] = { 
                        'data': data.T,
                        'start': self._recorded_variables[var].start_time,
                        'stop': self._recorded_variables[var].stop_time
                    }
                else:
                    #
                    # [ time, data(1D) ] => [ time, data(geometry) ] 
                    mat1 = data.reshape((data.shape[0],)+self._geometry)

                    data_dict[var] = { 
                                #
                                # [ time, data(geometry) ] => [  data(geometry), time ]                 
                        'data': np.transpose(mat1, tuple( range(1,self.dimension+1)+[0]) ),
                        'start': self._recorded_variables[var].start_time,
                        'stop': self._recorded_variables[var].stop_time
                    }
                                
                self._recorded_variables[var].reset()
            except:
                print("Error: only possible after compilation.")

        if( len(_variable)==1 and variable!=None):
            return data_dict[_variable[0]]
        else:
            return data_dict          

    def rank_from_coordinates(self, coord):
        """
        Returns rank corresponding to the given coordinates.
        
        Parameter:
        
            * *coord*: coordinate tuple, can be multidimensional.
        """
        rank = self._rank_from_coord( coord, self._geometry )
        
        if rank > self.size:
            Global._warning('Error when accessing neuron', str(coord), ': the population' , self.name , 'has only', self.size, 'neurons (geometry '+ str(self._geometry) +').')
            return None
        else:
            return rank

    def coordinates_from_rank(self, rank):
        """
        Returns a tuple representing the spatial coordinates corresponding to the geometry of the population.
        """
        # Check the rank
        if not rank < self.size:
            Global._warning('Error: the given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')
            return None
        
        return self._coord_from_rank( rank, self._geometry )

    def normalized_coordinates_from_rank(self, rank, norm=1.):
        """
        Returns a tuple of coordinates corresponding to the rank or coordinates, normalized between 0.0 and norm in each dimension.
        
        Parameters:
        
        * *rank*: position to normalize
        * *norm*: upper limit (default = 1.0)
        
        """
        try:
            normal = self._norm_coord_dict[self.dimension](rank, self._geometry)
        except KeyError:
            coord = self.coordinates_from_rank(rank)
                
            normal = tuple()
            for dim in range(self.dimension):
                if self._geometry[dim] > 1:
                    normal += ( norm * float(coord[dim])/float(self._geometry[dim]-1), )
                else:
                    normal += (0.0,) # default?
     
        return normal

    def set(self, values):
        """
        Sets neuron variable/parameter values.
        
        Parameter:
        
        * *values*: dictionary of attributes to be updated
            
            .. code-block:: python
                
                set({ 'tau' : 20.0, 'rate'= np.random.rand((8,8)) } )
        """
        for name, value in values.iteritems():
            self.__setattr__(name, value)
        
    def get(self, name):
        """
        Gets current variable/parameter values.
        
        Parameter:
        
        * *name*: attribute name as string
        """
        return self.__getattr__(name)
            
    def neuron(self, *coord):  
        " Returns neuron of coordinates coord in the population. If only one argument is given, it is the rank."  
        # Transform arguments
        if len(coord) == 1:
            if isinstance(coord[0], int):
                rank = coord[0]
                if not rank < self.size:
                    Global._error(' when accessing neuron', str(rank), ': the population', self.name, 'has only', self.size, 'neurons (geometry '+ str(self._geometry) +').')
                    return None
            else:
                rank = self.rank_from_coordinates( coord[0] )
                if rank == None:
                    return None
        else: # a tuple
            rank = self.rank_from_coordinates( coord )
            if rank == None:
                return None
        # Return corresponding neuron
        return IndividualNeuron(self, rank)
        
    @property   
    def neurons(self):
        """ Returns iteratively each neuron in the population.
        
        For instance, if you want to iterate over all neurons of a population:
        
            >>> for neur in pop.neurons:
            ...     print neur.rate
            
        Alternatively, one could also benefit from the ``__iter__`` special command. The following code is equivalent:
        
            >>> for neur in pop:
            ...     print neur.rate               
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
        if isinstance(indices, int): # a single neuron
            return IndividualNeuron(self, indices)
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
                            stop = self._geometry[rank]
                        if idx.step is None:
                            step = 1
                        rk_range = list(range(start, stop, step))
                        coords.append(rk_range)
                # Generate all ranks from the indices
                if self.dimension == 2:
                    ranks = [self.rank_from_coordinates((x, y)) for x in coords[0] for y in coords[1]]
                elif self.dimension == 3:
                    ranks = [self.rank_from_coordinates((x, y, z)) for x in coords[0] for y in coords[1] for z in coords[2]]
                if not max(ranks) < self.size:
                    Global._error("Indices do not match the geometry of the population", self._geometry)
                    return 
                return PopulationView(self, ranks)
                
    def __iter__(self):
        " Returns iteratively each neuron in the population in ascending rank order."
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank) 
            
    def set_variable_flags(self, name, value):
        """ Sets the flags of a variable for the population.
        
        If the variable ``rate`` is defined in the Neuron description through:
        
            rate = sum(exc) : max=1.0  
            
        one can change its maximum value with:
        
            pop.set_variable_flags('rate', {'max': 2.0})
            
        For valued flags (init, min, max), ``value`` must be a dictionary containing the flag as key ('init', 'min', 'max') and its value. 
        
        For positional flags (population, implicit), the value in the dictionary must be set to the empty string '':
        
            pop.set_variable_flags('rate', {'implicit': ''})
        
        A None value in the dictionary deletes the corresponding flag:
        
            pop.set_variable_flags('rate', {'max': None})
            
        """
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The population '+self.name+' has no variable called ' + name)
            return
            
        for key, val in value.iteritems():
            if val == '': # a flag
                try:
                    self.description['variables'][rk_var]['flags'].index(key)
                except: # the flag does not exist yet, we can add it
                    self.description['variables'][rk_var]['flags'].append(key)
            elif val == None: # delete the flag
                try:
                    self.description['variables'][rk_var]['flags'].remove(key)
                except: # the flag did not exist, check if it is a bound
                    if has_key(self.description['variables'][rk_var]['bounds'], key):
                        self.description['variables'][rk_var]['bounds'].pop(key)
            else: # new value for init, min, max...
                if key == 'init':
                    self.description['variables'][rk_var]['init'] = val 
                    self.init[name] = val              
                else:
                    self.description['variables'][rk_var]['bounds'][key] = val
                
       
            
    def set_variable_equation(self, name, equation):
        """ Changes the equation of a variable for the population.
        
        If the variable ``rate`` is defined in the Neuron description through:
        
            tau * drate/dt + rate  = sum(exc) : max=1.0  
            
        one can change the equation with:
        
            pop.set_variable_equation('rate', 'rate = sum(exc)')
            
        Only the equation should be provided, the flags have to be changed with ``set_variable_flags()``.
        
        .. warning::
            
            This method should be used with great care, it is advised to define another Neuron object instead. 
            
        """         
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The population '+self.name+' has no variable called ' + name)
            return               
        self.description['variables'][rk_var]['eq'] = equation    
            
            
    def _find_variable_index(self, name):
        " Returns the index of the variable name in self.description['variables']"
        for idx in range(len(self.description['variables'])):
            if self.description['variables'][idx]['name'] == name:
                return idx
        return -1

    def raster_plot(self, compact=False):
        """ Returns data allowing to display a raster plot for a spiking population.

        For all neurons in the population, the absolute time (in simulation steps since the beginning) where a spike was emitted is given.

        By default, it returns a (N, 2) Numpy array where each spike (first index) is represented by the corresponding time (first column) and the neuron index (second column).  It can be very easily plotted, for example with matplotlib::

            >>> spikes = pop.raster_plot()
            >>> from pylab import *
            >>> plot(spikes[:, 0], spikes[:, 1], 'o')
            >>> show()

        If ``compact`` is set to ``True``, it will return a list of lists, where the first index corresponds to the neurons' ranks, and the second is a list of time steps where a spike was emitted.

        Parameters:

        * ``compact``: defines the format of the returned array.
        """
        if self.description['type'] != 'spike':
            Global._warning('raster_plot() is only available for spiking populations.')
            return []
        data = self.cyInstance._get_recorded_spike()
        if compact:
            return data
        else:
            return np.array([ [t, neuron] for neuron in range(len(data)) for t in data[neuron] ] )

