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
import Global 
import Neuron

from Neuron import RateNeuron, SpikeNeuron

import ANNarchy4.generator.Population as PopulationGenerator

from ANNarchy4.core.Descriptor import Descriptor, Attribute
from ANNarchy4.core.PopulationView import PopulationView
from ANNarchy4.core.Random import RandomDistribution

from ANNarchy4.core.Record import Record

import traceback
import numpy as np

class Population(Descriptor):
    """
    Represents a population of neurons.
    """

    def __init__(self, geometry, neuron, name=None):
        """
        Constructor of the population.
        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. If an integer is given, it is the size of the population.

            * *neuron*: instance of ``ANNarchy4.Neuron``

            * *name*: unique name of the population (optional).
        
        """
        if isinstance(geometry, int): # 1D
            self.geometry = (geometry, )
        else: # a tuple is given
            self.geometry = geometry
            
        self.neuron_type = neuron
        self._id = len(Global._populations)
        if name:
            self.name = name
        else:
            self.name = 'Population'+str(self._id)
        
        self.generator = PopulationGenerator.Population(self)
                
        Global._populations.append(self)
        
        self._recorded_variables = {}
        
        for var in self.variables:
            self._recorded_variables[var] = Record(var)

        self.initialized = True        
        
    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        for var in self.variables + self.parameters:
            setattr(self, var, Attribute(var))

    @property
    def width(self):
        """
        Width of population.
        """
        if self.dimension >= 1:
            return self.geometry[0]
        else:
            return 1

    @property
    def height(self):
        """
        Height of population.
        """
        if self.dimension >= 2:
            return self.geometry[1]
        else: 
            return 1

    @property
    def depth(self):
        """
        Depth of population.
        """
        if self.dimension == 3:
            return self.geometry[2]
        else: 
            return 1
        
    @property
    def cpp_class(self):
        """
        Returns name of cpp class.
        """
        return self.generator.class_name
    
    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        ret_var = [] #default        
        
        #check additional variables
        neur_var = self.generator.neuron_variables
        for name, var in neur_var.iteritems():
            if var['type'] == 'local':
                ret_var.append(name)      
                
        return ret_var

    @property
    def parameters(self):
        """
        Returns a list of all parameter names.
        """
        neur_var = self.generator.neuron_variables
        ret_par = []        
        for name, var in neur_var.iteritems():
            if var['type'] == 'global':
                ret_par.append(name)      

        return ret_par
        
    @property
    def size(self):
        """
        Number of neurons in the population.
        """
        size = 1
        for i in range(len(self.geometry)):
            size *= self.geometry[i]
        return size
        
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
        return len(self.geometry)
        
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
                #TODO:
                #print "Error (start_record): only possible after compilation."
                pass

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
                
    def get_record(self, variable=None, as_1D=False):
        """
        Returns the recorded data as one matrix or a dictionary if more then one variable is requested. 
        The last dimension represents the time, the remaining dimensions are the population geometry.
        
        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument provided, the remaining recorded data is returned.  
        * *as_1D*: by default this functions returns the data as matrix (geometry shape, time). If as_1D set to True, the data will be returned as two-dimensional plot (neuron x time)
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
                
                if as_1D:
                    #
                    # [ time, data(1D) ] => [ time, data(1D) ] 
                    mat1 = data.T

                    data_dict[var] = { 
                        'data': mat1,
                        'start': self._recorded_variables[var].start_time,
                        'stop': self._recorded_variables[var].stop_time
                    }
                else:
                    #
                    # [ time, data(1D) ] => [ time, data(geometry) ] 
                    mat1 = data.reshape((data.shape[0],)+self.geometry)

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
        
    def get_variable(self, variable):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameter:
        
        * *variable*: should be a string representing the variables's name.
        
        .. warning::
        
        In contrast to direct accessing variables, this function returns the synaptic as one dimensional array. 
        """
        if hasattr(self, 'cyInstance'):
            if hasattr(self.cyInstance, variable):
                return np.array(getattr(self.cyInstance, variable)).reshape(self.geometry)
            else:
                print('Error: variable', variable, 'does not exist in this population.')
                print(traceback.print_stack())
        else:
            print('Error: the network is not compiled yet.')
            print(traceback.print_stack())
            
    def get_parameter(self, parameter):
        """
        Returns the value of the given variable for all neurons in the population, as a NumPy array having the same geometry as the population.
        
        Parameter:
        
        * *parameter*: should be a string representing the variables's name.
        """
        
        if hasattr(self, 'cyInstance'):
            if hasattr(self.cyInstance, parameter):
                return getattr(self.cyInstance, parameter)
            else:
                print('Error: parameter', parameter, 'does not exist in this population.')
                print(traceback.print_stack())
        else:
            print('Error: the network is not compiled yet.')
            print(traceback.print_stack())
    
    def rank_from_coordinates(self, coord):
        """
        Returns rank corresponding to the given coordinates.
        
        Parameter:
        
            * *coord*: coordinate tuple, can be multidimensional.
        """
        # Check the coordinates
        if not len(coord) == self.dimension:
            Global.ANNarchyPrint('Error when accessing neuron', str(coord), ': the population', self.name , 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
            return None
        for dim in range(len(coord)):
            if not coord[dim] < self.geometry[dim]:
                Global.ANNarchyPrint('Error when accessing neuron', str(coord), ': the population' , self.name , 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
                return None
        # Return the rank
        return np.ravel_multi_index( coord, self.geometry)

    def coordinates_from_rank(self, rank):
        """
        Returns a tuple representing the spatial coordinates corresponding to the geometry of the population.
        """
        # Check the rank
        if not rank < self.size:
            Global.ANNarchyPrint('Error: the given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')
            return None
        
        coord = np.unravel_index(rank, self.geometry)
        return coord

    def normalized_coordinates_from_rank(self, pos, norm=1.):
        """
        Returns a tuple of coordinates corresponding to the rank or coordinates, normalized between 0.0 and norm in each dimension.
        
        Parameters:
        
        * *pos*: position to normalize
        * *norm*: upper limit (default = 1.0)
        
        """
        if isinstance(pos, int):
            coord = self.coordinates_from_rank(pos)
        else:
            coord = pos
            
        normal = tuple()
        for dim in range(self.dimension):
            if self.geometry[dim] > 1:
                normal += ( norm * float(coord[dim])/float(self.geometry[dim]-1), )
            else:
                normal += (0.0,) # default?

        #print self.geometry,'=> (1.0,1.0):',coord,'=>', normal               
        return normal

    def set(self, value):
        """
        Sets neuron variable/parameter values.
        
        Parameter:
        
            * *value*: dictionary of attributes to be updated
            
                .. code-block:: python
                
                    set( 'tau' : 20, 'rate'= np.random.rand((8,8)) } )
        """
        if hasattr(self, 'cyInstance'):
            for val_key in value.keys():
                if hasattr(self, val_key):
                    # Check the type of the data
                    if isinstance(value[val_key], RandomDistribution):
                        val = value[val_key].getValues(self.size) 
                    else: 
                        val = value[val_key] 
                    setattr(self.cyInstance, val_key, val)
                else:
                    Global._error("population does not have the attribute: ",val_key,".")
        else:
            Global._error("the network is not compiled yet.")
            Global._print(traceback.print_stack())
        
    def get(self, value):
        """
        Gets current variable/parameter values.
        
        Parameter:
        
        * *value*: value name as string
        """
        if hasattr(self, 'cyInstance'):
            if value in self.variables:
                return self.get_variable(value)
            elif value in self.parameters:
                return self.get_parameter(value)
            else:
                Global._error("population does not contain attribute: '"+value+"'")   
        else:
            Global._error("the network is not compiled yet.'")
            Global._print(traceback.print_stack())
            
    def neuron(self, coord):  
        " Returns neuron of coordinates coord in the population. If only one argument is given, it is the rank."  
    
        # Transform arguments
        if isinstance(coord, int):
            rank = coord
            if not rank < self.size:
                Global._error(' when accessing neuron', str(rank), ': the population', self.name, 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
                return None

        else: # a tuple
            rank = self.rank_from_coordinates( coord )
            if rank == None:
                return None
        # Return corresponding neuron
        return Neuron.IndividualNeuron(self, rank)
        
          
    def neurons(self):
        """ Returns iteratively each neuron in the population.
        
        For instance, if you want to iterate over all neurons of a population:
        
            >>> for neur in pop.neurons():
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
            return self.neuron(indices)
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
                if not max(ranks) < self.size:
                    Global._ANNarchyError("indices do not match the geometry of the population", self.geometry)
                    return 
                return PopulationView(self, ranks)
                
    def __iter__(self):
        " Returns iteratively each neuron in the population in ascending rank order."
        for neur_rank in range(self.size):
            yield self.neuron(neur_rank)  


