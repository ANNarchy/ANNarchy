"""
    Dendrite.py
    
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
#from .Descriptor import Descriptor, Attribute
import ANNarchy4.core.Global as Global
from ANNarchy4.core.Neuron import RateNeuron
from ANNarchy4.core.Random import RandomDistribution
import numpy as np
import traceback

from ANNarchy4.core.Record import Record


class Dendrite(object):
    """
    A dendrite encapsulates all synapses of one neuron coming from a single projection.
    
    *Hint*: this class will be created from projection class.
    
    Parameter:
    
    * *proj*: projection instance
    * *post_rank*: rank of the postsynaptic neuron
    * *cython_instance*: instance of the cythonized dendrite class.
    """
    def __init__(self, proj, post_rank, ranks=None, weights=None, delays=None):

        self.post_rank = post_rank
        self.proj = proj
        self.pre = proj.pre
        
        self.parameters = self.proj.parameters
        self.variables = self.proj.variables
        self.attributes = self.proj.attributes

        cython_module = __import__('ANNarchyCython') 
        proj_class_name = 'Local' + self.proj.name
        local_proj = getattr(cython_module, proj_class_name)
        
        if isinstance(self.proj.pre.neuron_type, RateNeuron) and isinstance(self.proj.post.neuron_type, RateNeuron): 
            self.cy_instance = local_proj(
                self.proj._id, 
                self.proj.pre.rank, 
                self.proj.post.rank, 
                post_rank, 
                self.proj.post.targets.index(self.proj.target),
                False 
            )
        else:
            self.cy_instance = local_proj(
                self.proj._id, 
                self.proj.pre.rank, 
                self.proj.post.rank, 
                post_rank, 
                self.proj.post.targets.index(self.proj.target),
                True
            )
                    
        self.cy_instance.rank = ranks
        self.cy_instance.value = weights
        if delays != None:
            self.cy_instance.delay = delays
            max_delay = np.amax(delays)
            self.proj.pre.cyInstance.set_max_delay(int(max_delay))

        self._recorded_variables = {}        
        for var in self.variables:
            self._recorded_variables[var] = Record(var)

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'proj':
            return object.__getattribute__(self, name)
        elif name == 'attributes':
            return object.__getattribute__(self, 'attributes')
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                return getattr(self.cy_instance, name)
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name == 'proj':
            object.__setattr__(self, 'proj', value)
        elif name == 'attributes':
            object.__setattr__(self, name, value)
        elif hasattr(self, 'attributes'):
            if name in self.proj.attributes:
                setattr(self.cy_instance, name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)
    
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
                print('variable',var,'is not inited.')
                continue
            
            try:
                getattr(self.cy_instance, '_start_record_'+var)()
    
                if Global.config['verbose']:
                    print('start record of', var, '(', self.proj.name, ')')
                    
                self._recorded_variables[var].start()
            except:
                #TODO:
                print self.cy_instance
                print '_start_record_'+var
                print "Error (start_record): only possible after compilation."
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
                getattr(self.cy_instance, '_stop_record_'+var)()

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
                getattr(self.cy_instance, '_start_record_'+var)()
                
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
                Global._print(var, 'is not a recordable variable of', self.name)
                continue
            
            if self._recorded_variables[var].is_running:
                self.pause_record(var)
            
            try:
                if Global.config['verbose']:
                    Global._print('get record of '+var+' ('+self.proj.name+')')
                    
                data = getattr(self.cy_instance, '_get_recorded_'+var)()
                
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
                    print "Please use the as1D function."
                    return []
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
    
    def set(self, value):
        """
        Update dendrite variable/parameter.
        
        Parameter:
        
            * *value*: value need to be update
            
                .. code-block:: python
                
                    set( 'tau' : 20, 'value'= np.random.rand(8,8) } )
        """
        for val_key in value.keys():
            if hasattr(self.cy_instance, val_key):
                # Check the type of the data!!
                if isinstance(value[val_key], RandomDistribution):
                    val = value[val_key].getValues(self.size) 
                else: 
                    val = value[val_key]           
                # Set the value
                setattr(self.cy_instance, val_key, val)
            else:
                Global._error("dendrite has no parameter/variable called", val_key)
                
    def get(self, value):
        """
        Get current variable/parameter value
        
        Parameter:
        
            * *value*: value name as string
        """
        if value in self.variables:
            return self.get_variable(value)
        elif value in self.parameters:
            return self.get_parameter(value)
        else:
            Global._error("dendrite has no parameter/variable called", value)     
    
    @property
    def size(self):
        """
        Number of synapses.
        """
        return self.cy_instance.size
    
    def __len__(self):
        """
        Number of synapses.
        """
        return self.cy_instance.size
        
    @property
    def target(self):
        """
        Connection type id.
        """
        return self.cy_instance.get_target()
        
    def get_variable(self, variable):
        """
        Returns the value of the given variable for all synapses in the dendrite, as a NumPy array having the same geometry as the presynaptic population.
        
        Parameter:
        
        * *variable*:    a string representing the variable's name.
        """
        if hasattr(self.cy_instance, variable):
            return getattr(self.cy_instance, variable)
        else:
            Global._error("variable", variable, "does not exist in this dendrite.")
            Global._print(traceback.print_stack())

    def get_parameter(self, parameter):
        """
        Returns the value of the given parameter, which is common for all synapses in the dendrite.
        
        Parameter:
        
        * *parameter*:    a string representing the parameter's name.
        """
        if hasattr(self.cy_instance, parameter):
            return getattr(self.cy_instance, parameter)
        else:
            Global._error("parameter", parameter, "does not exist in this dendrite.")
            Global._print(traceback.print_stack())

    def add_synapse(self, rank, value, delay=0):
        """
        Adds a synapse to the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
            * *value*:    synaptic weight
            * *delay*:    additional delay of the synapse (as default a delay of 1ms is assumed)
        """
        self.cy_instance.add_synapse(rank, value, delay)
    
    def remove_synapse(self, rank):
        """
        Removes the synapse from the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
        """
        self.cy_instance.remove_synapse(rank)

