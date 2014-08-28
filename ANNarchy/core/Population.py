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

from ANNarchy.core.Random import RandomDistribution

import numpy as np



class Population(object):
    """
    Represents a population of neurons.
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
        self.neuron = neuron
        
        # Attribute a name if not provided
        self.id = len(Global._populations)
        self.class_name = 'pop'+str(self.id)
        
        if name:
            self.name = name
        else:
            self.name = self.class_name
                
        # Add the population to the global variable
        Global._populations[self.class_name] = self
        
        # Get a list of parameters and variables
        self.parameters = []
        self.variables = []
        for param in self.neuron.description['parameters']:
            self.parameters.append(param['name'])
        for var in self.neuron.description['variables']:
            self.variables.append(var['name'])
        self.attributes = self.parameters + self.variables

        # Store initial values
        self.init = {}
        for param in self.neuron.description['parameters']:
            self.init[param['name']] = param['init']
        for var in self.neuron.description['variables']:
            self.init[var['name']] = var['init']
        
        # List of targets actually connected
        self.targets = []

        # List of global operations needed by connected projections
        self.global_operations = []
                
        # Allow recording of variables

        # Finalize initialization
        self.initialized = False
        self.cyInstance = None

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
        
    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size)
        
        # Create the attributes and actualize the initial values
        self._init_attributes()

        # If the spike population has a refractory period:        
        if self.neuron.type == 'spike' and self.neuron.description['refractory']:
            if isinstance(self.description['refractory'], str): # a global variable
                try:
                    self.refractory = eval('self.'+self.description['refractory'])
                except:
                    Global._print(self.description['refractory'])
                    Global._error('The initialization for the refractory period is not valid.')
                    exit(0)
            else: # a value
                self.refractory = self.neuron.description['refractory']


    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        self.initialized = True  
        self.set(self.init)

    def reset(self):
        """
        Resets all parameters and variables to the value they had before the call to compile.
        """
        self._init_attributes()

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if self.initialized: # access after compile()
                    return self._get_cython_attribute(name)
                else: # access before compile()
                    if name in self.neuron.description['local']:
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
            if attribute in self.neuron.description['local']:
                return np.array(getattr(self.cyInstance, attribute)).reshape(self.geometry)
            else:
                return getattr(self.cyInstance, attribute)
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
            if attribute in self.neuron.description['local']:
                if isinstance(value, np.ndarray):
                    setattr(self.cyInstance, attribute, value.reshape(self.size))
                elif isinstance(value, list):
                    setattr(self.cyInstance, attribute, np.array(value).reshape(self.size))
                else:
                    setattr(self.cyInstance, attribute, np.array( [value]*self.size ))
            else:
                setattr(self.cyInstance, attribute, value)
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
        if self.description['type'] == 'spike':
            if self.initialized:
                return self.cyInstance._get_refractory()
            else :
                return self.description['refractory']
        else:
            Global._error('rate-coded neurons do not have refractory periods...')
            return None

    @refractory.setter
    def refractory(self, value):
        if self.description['type'] == 'spike':
            if self.initialized:
                if isinstance(value, RandomDistribution):
                    refs = (value.get_values(self.size)/Global.config['dt']).astype(int)
                elif isinstance(value, np.ndarray):
                    refs = (value / Global.config['dt']).astype(int).reshape(self.size)
                else:
                    refs = (value/ Global.config['dt']*np.ones(self.size)).astype(int)
                # TODO cast into int
                self.cyInstance._set_refractory(refs)
            else: # not initialized yet, saving for later
                self.description['refractory'] = value
        else:
            Global._error('rate-coded neurons do not have refractory periods...')

