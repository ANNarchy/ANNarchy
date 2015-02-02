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

from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Neuron import IndividualNeuron

import numpy as np
import copy, inspect

# Template for specific population bypassing code generation.
# The id of the population should be let free with %(id)s
pop_generator_template = {
    'omp': {
        # C++ struct to encapsulate all data
        # Example:
        # struct PopStruct%(id)s{
        #     // Number of neurons
        #     int size;
        #     // Local parameter baseline
        #     std::vector< double > baseline ;
        #     // Local variable r
        #     std::vector< double > r ;
        #     std::vector< std::vector< double > > recorded_r ;
        #     bool record_r ;
        #     // Targets
        #     std::vector<double> sum_exc;
        #     // Random numbers
        #     std::vector<double> rand_0;
        #     std::uniform_real_distribution<double> dist_rand_0;
        # }; 
        'header_pop_struct' : None,

        # Initialize the random distribution objects
        # Example:
        # pop%(id)s.rand_0 = std::vector<double>(pop%(id)s.size, 0.0);
        # pop%(id)s.dist_rand_0 = std::uniform_real_distribution<double>(-0.5, 0.5);
        'body_random_dist_init': None,

        # Initialize the delayed arrays. Needs the extra %(delay)s argument, which represents the maximum delay in the population.
        # Example:
        # pop%(id)s._delayed_r = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(pop%(id)s.size, 0.0));
        'body_delay_init': None,

        # Initialize spike-specific arrays. Does not need to overriden in most cases.
        # Example:
        # pop%(id)s.spike = std::vector<bool>(pop%(id)s.size, false);
        # pop%(id)s.spiked = std::vector<int>(0, 0);
        # pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
        # pop%(id)s.refractory_remaining = std::vector<int>(pop%(id)s.size, 0);
        'body_spike_init': None,

        # Initialize the global operations
        # Example:
        # pop%(id)s._mean_r = 0.0;
        'body_globalops_init': None,

        # Updates the random numbers
        # Example:
        # // RD of pop%(id)s
        # #pragma omp parallel for
        # for(int i = 0; i < pop%(id)s.size; i++)
        # {
        #     pop%(id)s.rand_0[i] = pop%(id)s.dist_rand_0(rng[omp_get_thread_num()]);
        # }
        'body_random_dist_update': None,

        # Updates the neural variables
        # Example:
        # // Updating the local variables of population %(id)s
        # #pragma omp parallel for
        # for(int i = 0; i < pop%(id)s.size; i++){
        #     pop1.mp[i] += dt*(-pop%(id)s.mp[i] + pop%(id)s.rand_0[i] + pop%(id)s.sum_exc[i] + pop%(id)s.sum_inh[i])/pop%(id)s.tau;
        #     pop1.r[i] = clip(pop1.mp[i], 0.0, 1.0);
        # }
        'body_update_neuron': None,

        # Delays variables
        # Example:
        # // Enqueuing outputs of pop%(id)s
        # pop%(id)s._delayed_r.push_front(pop%(id)s.r);
        # pop%(id)s._delayed_r.pop_back();
        'body_delay_code': None,

        # Computes global operations
        # Example:
        # pop%(id)s._mean_r = mean_value(pop%(id)s.r;
        'body_update_globalops': None,

        # Record the variables
        # Example:
        # if(pop%(id)s.record_r
        #     pop%(id)s.recorded_r.push_back(pop%(id)s.r) ;
        'body_record': None,

        # Export of the C++ struct to Cython (must have an indent of 4)
        # Example:
        # cdef struct PopStruct%(id)s :
        #     int size
        #     # Global parameter tau
        #     double  tau 
        #     # Local variable r
        #     vector[double] r 
        #     vector[vector[double]] recorded_r 
        #     bool record_r 
        #     # Targets
        #     vector[double] sum_exc
        'pyx_pop_struct': None,

        # Wrapper class in Cython (no indentation)
        # Example:
        # cdef class pop%(id)s_wrapper :
        #     def __cinit__(self, size):
        #         pop%(id)s.size = size
        #         pop%(id)s.r = vector[double](size, 0.0)
        #         pop%(id)s.sum_exc = vector[double](size, 0.0)
        #     property size:
        #         def __get__(self):
        #             return pop%(id)s.size
        #     # Global parameter tau
        #     cpdef double get_tau(self):
        #         return pop%(id)s.tau
        #     cpdef set_tau(self, double value):
        #         pop%(id)s.tau = value
        #     # Local variable r
        #     cpdef np.ndarray get_r(self):
        #         return np.array(pop%(id)s.r)
        #     cpdef set_r(self, np.ndarray value):
        #         pop%(id)s.r = value
        #     cpdef double get_single_r(self, int rank):
        #         return pop%(id)s.r[rank]
        #     cpdef set_single_r(self, int rank, double value):
        #         pop%(id)s.r[rank] = value
        'pyx_pop_class': None,
    },
    'cuda': {} # TODO
}

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
        if isinstance(geometry, int): 
            # 1D
            self.geometry = (geometry, )
            self.width = geometry
            self.height = 1
            self.depth = 1
            self.dimension = 1
        else: 
            # a tuple is given, can be 1 .. N dimensional
            self.geometry = geometry
            self.width = geometry[0]
            if len(geometry)>=2:
                self.height = geometry[1]
            else:
                self.height = 1
            if len(geometry)>=3:                
                self.depth = geometry[2]
            else:
                self.depth = 1
                
            self.dimension = len(geometry)

        # Compute the size
        size = 1        
        for i in range(len(self.geometry)):
            size *= self.geometry[i]
            
        self.size = size
        
        # Store the neuron type
        if inspect.isclass(neuron):
            self.neuron_type = neuron()
        else:
            self.neuron_type = copy.deepcopy(neuron)
        self.neuron_type._analyse()
        self.generator = copy.deepcopy(pop_generator_template)
        
        # Store the stop condition
        self.stop_condition = stop_condition

        # Attribute a name if not provided
        self.id = len(Global._populations)
        self.class_name = 'pop'+str(self.id)
        
        if name:
            self.name = name
        else:
            self.name = self.class_name
                
        # Add the population to the global variable
        Global._populations.append(self)
        
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

        # Finalize initialization
        self.initialized = False
        self.cyInstance = None
        self._activated = True

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
        self.recorded_variables = {}
        
    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size)


    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        self.initialized = True  
        self.set(self.init)
        self.cyInstance.activate(self._activated)

        # If the spike population has a refractory period:    
        if self.neuron_type.type == 'spike' and self.neuron_type.description['refractory']:
            if isinstance(self.neuron_type.description['refractory'], str): # a global variable
                try:
                    self.refractory = eval('self.'+self.neuron_type.description['refractory'])
                except Exception, e:
                    Global._print(e, self.neuron_type.description['refractory'])
                    Global._error('The initialization for the refractory period is not valid.')
                    exit(0)
            else: # a value
                self.refractory = self.neuron_type.description['refractory']

    def reset(self):
        """
        Resets all parameters and variables to the value they had before the call to compile.
        """
        self._init_attributes()

    def enable(self):
        """
        Enables computations in this population (including the projections leading to it).
        """
        if self.initialized:
            self.cyInstance.activate(True)
        else:
            self._activated = True

    def disable(self):
        """
        Enables computations in this population (including the projections leading to it).
        """
        if self.initialized:
            self.cyInstance.activate(False)
        else:
            self._activated = False

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
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
        if not hasattr(self, 'initialized'): # Before the end of the constructor
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
        except Exception, e:
            print e
            Global._error('Error: the variable ' +  attribute +  ' does not exist in this population.')
        
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
        except Exception, e:
            print e
            Global._error('Error: either the variable ' +  attribute +  ' does not exist in this population, or the provided array does not have the right size.')
        
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
        for name, value in values.iteritems():
            self.__setattr__(name, value)
        
    def get(self, name):
        """
        Returns the value of neural variables and parameters.
        
        *Parameter*:
        
        * **name**: attribute name as a string.
        """
        return self.__getattr__(name)
            


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
            Global._error('rate-coded neurons do not have refractory periods...')
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
            Global._error('rate-coded neurons do not have refractory periods...')

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
        ...     print neur.r
            
        Alternatively, one could also benefit from the ``__iter__`` special command. The following code is equivalent:
        
        >>> for neur in pop:
        ...     print neur.r               
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
                    return None
                if not max(ranks) < self.size:
                    Global._error("Indices do not match the geometry of the population", self.geometry)
                    return None
                return PopulationView(self, ranks)
                
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
            Global._error('There is no neuron of coordinates', coord, 'in the population', self.name, self.geometry)
            exit(0)

        if rank > self.size:
            Global._warning('Error when accessing neuron', str(coord), ': the population' , self.name , 'has only', self.size, 'neurons (geometry '+ str(self.geometry) +').')
            exit(0)
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
            exit(0)
        
        try:
            coord = self._coord_from_rank( rank, self.geometry )
        except:
            Global._error('The given rank', str(rank), 'is larger than the size of the population', str(self.size) + '.')
            exit(0)
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
    def start_record(self, variable, period = None):
        """
        Start recording neural variables.
        
        Parameter:
            
            * **variable**: single variable name or list of variable names.  

            * **period**: delay between two recording (default: dt).

        Example::

            pop1.start_record('r')
            pop2.start_record(['mp', 'r'])      
        """
        if not period:
            period = Global.config['dt']
        _variable = []
        
        if isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        # set period and offset
        self.cyInstance.set_record_period( int(period/Global.config['dt']), Global.get_current_step() )

        # start recording of variables
        for var in _variable:
            if not var in self.variables + ['spike']:
                Global._error('start: ' + var, 'is not a recordable variable of the population.')
                return

            self.recorded_variables[var] = {'start': [Global.get_current_step()], 'stop': [-1], 'period': period}

            try:
                getattr(self.cyInstance, 'start_record_'+var)()
            except:
                Global._error(var + 'is not a recordable variable.')
                return

    def stop_record(self, variable=None):
        """
        Stops recording the previously defined variables.

        Parameter:
            
        * **variable**: single variable name or list of variable names. If no argument is provided, all recordings will stop.

        Example::

            pop1.stop_record('r')
            pop2.stop_record(['mp', 'r'])  
        """
        _variable = []
        if variable == None:
            _variable = self.recorded_variables.keys()
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')       
        
        for var in _variable:
            if not var in self.recorded_variables.keys():
                Global._warning(var, 'was not recording.')
                continue
            try:
                getattr(self.cyInstance, 'stop_record_'+var)()
            except:
                Global._error('stop: ' + var + 'is not a recordable variable.')
                return

            del self.recorded_variables[var]

    def pause_record(self, variable=None):
        """
        Pauses the recording of variables (can be resumed later with resume_record()).

        *Parameter*:
            
        * **variable**: single variable name or list of variable names. If no argument is provided all recordings will pause.

        Example::

            pop1.pause_record('r')
            pop2.pause_record(['mp', 'r'])  
        """
        _variable = []
        if variable == None:
            _variable = self.recorded_variables
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')       
        
        for var in _variable:
            if not var in self.recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue
            
            try:
                getattr(self.cyInstance, 'stop_record_'+var)()

            except:
                Global._error('pause: ' + var + 'is not a recordable variable.')
                return

            self.recorded_variables[var]['stop'][-1] = Global.get_current_step()
            self.recorded_variables[var]['start'].append(-1)
            self.recorded_variables[var]['stop'].append(-1)

            if Global.config['verbose']:
                print('pause record of', var, '(', self.name, ')')

    def resume_record(self, variable=None):
        """
        Resume recording the previous defined variables.
        
        *Parameter*:
            
            * **variable**: single variable name or list of variable names.  

        Example::

            pop1.resume_record('r')
            pop2.resume_record(['mp', 'r'])        
        """
        _variable = []
        
        if variable == None:
            _variable = self.recorded_variables
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        for var in _variable:
            if not var in self.recorded_variables.keys():
                print(var, 'is not a recordable variable of', self.name)
                continue
            
            # set period and offset
            self.cyInstance.set_record_period( int(self.recorded_variables[var]['period']/Global.config['dt']), Global.get_current_step() )

            try:
                getattr(self.cyInstance, 'start_record_'+var)()

            except:
                Global._error('resume: ' + var + 'is not a recordable variable.')
                return

            self.recorded_variables[var]['start'][-1] = Global.get_current_step()
            
            if Global.config['verbose']:
                Global._print('resume record of', var, '(' , self.name, ')')

    def get_record(self, variable=None, reshape=False):
        """
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
        
        _variable = []
        if variable == None:
            _variable = self.recorded_variables.keys()
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')

        data = {}

        for var in _variable:
            if not var in self.variables + ['spike']:
                Global._warning(var, 'is not a recordable variable of the population.')
                continue
            if not var in self.recorded_variables.keys():
                Global._warning(var, 'was not recording.')
                continue
        
            try:
                var_data = getattr(self.cyInstance, 'get_record_'+var)()
            except:
                Global._error('get: ' + var + 'is not a recordable variable.')
                return {}

            if self.recorded_variables[var]['stop'][-1] == -1:
                self.recorded_variables[var]['stop'][-1] = Global.get_current_step()


            if not reshape:
                data[var] = {
                    'start': self.recorded_variables[var]['start'] if len(self.recorded_variables[var]['start']) >1 else self.recorded_variables[var]['start'][0],
                    'stop' : self.recorded_variables[var]['stop'] if len(self.recorded_variables[var]['stop']) >1 else self.recorded_variables[var]['stop'][0] ,
                    'data' : np.array(var_data).T if not var == 'spike' else np.array(var_data),
                    'period' : self.recorded_variables[var]['period']
                }
            else:
                if not var == 'spike':
                    var_data = np.array(var_data)
                    mat1 = var_data.reshape((var_data.shape[0],)+self.geometry)

                    data[var] = {
                        'start': self.recorded_variables[var]['start'] if len(self.recorded_variables[var]['start']) >1 else self.recorded_variables[var]['start'][0],
                        'stop' : self.recorded_variables[var]['stop'] if len(self.recorded_variables[var]['stop']) >1 else self.recorded_variables[var]['stop'][0] ,
                        'data' : np.transpose(mat1, tuple( range(1, self.dimension+1)+[0])),
                        'period' : self.recorded_variables[var]['period']
                        }
                else:
                    Global._error("reshape=true is invalid for get_record('spike')")


            self.recorded_variables[var]['start'] = [Global.get_current_step()]
            self.recorded_variables[var]['stop'] = [-1]

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
            return
            
        for key, val in value.iteritems():
            if val == '': # a flag
                try:
                    self.neuron_type.description['variables'][rk_var]['flags'].index(key)
                except: # the flag does not exist yet, we can add it
                    self.neuron_type.description['variables'][rk_var]['flags'].append(key)
            elif val == None: # delete the flag
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
            return         
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
                Global._error('Can not save the attribute ' + var + 'in the population ' + self.name + '.')              
        return desc 

    def save(self, filename):
        """
        Saves all information about the population (structure, current value of parameters and variables) into a file.

        * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * Otherwise, the data will be pickled into a simple binary text file using cPickle.
        
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
