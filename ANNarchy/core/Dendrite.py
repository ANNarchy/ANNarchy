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
import ANNarchy.core.Global as Global
from ANNarchy.core.Neuron import RateNeuron
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Record import Record

import numpy as np
import traceback


class Dendrite(object):
    """
    A dendrite encapsulates all synapses of one neuron coming from a single projection.
    
    *Hint*: this class is only a data wrapper  created from projection class.
    
    Parameter:
    
    * *proj*: projection instance
    * *post_rank*: rank of the postsynaptic neuron
    * *cython_instance*: instance of the cythonized dendrite class.
    """
    def __init__(self, proj, post_rank):

        self.post_rank = post_rank
        self.proj = proj
        self.pre = proj.pre

        self.target = self.proj.target
        
        self.parameters = self.proj.parameters
        self.variables = self.proj.variables

    def size(self):
        """
        Number of synapses.
        """
        if self.proj.cyInstance:
            return self.proj.cyInstance._nb_synapses(self.post_rank)
        return 0

    def __len__(self):
        """
        Number of synapses.
        """
        return self.size

    #########################
    ### Access to attributes
    #########################
    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'proj':
            return object.__getattribute__(self, name)
        elif name == 'attributes':
            return object.__getattribute__(self, name)
        elif hasattr(self, 'proj'):
            if name in self.proj.attributes+['rank', 'delay']:
                return getattr(self.proj.cyInstance, '_get_'+name)(self.post_rank)
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name == 'proj':
            object.__setattr__(self, 'proj', value)
        elif name == 'attributes':
            object.__setattr__(self, 'attributes', value)
        elif hasattr(self, 'proj'):
            if name in self.proj.attributes:
                getattr(self.proj.cyInstance, '_set_'+name)(self.post_rank, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def set(self, value):
        """
        Update dendrite variable/parameter.
        
        Parameter:
        
            * *value*: value need to be update
            
                .. code-block:: python
                
                    set( 'tau' : 20, 'value'= np.random.rand(8,8) } )
        """
        for val_key in value.keys():
            if hasattr(self.proj.cy_instance, val_key):
                # Check the type of the data!!
                if isinstance(value[val_key], RandomDistribution):
                    val = value[val_key].getValues(self.size) 
                else: 
                    val = value[val_key]           
                # Set the value
                getattr(self.proj.cyInstance, '_set_'+val_key)(self.post_rank, val)
            else:
                Global._error("Dendrite has no parameter/variable called", val_key)
                
    def get(self, value):
        """
        Get current variable/parameter value
        
        Parameter:
        
            * *value*: value name as string
        """
        if value in self.attributes:
            return getattr(self.proj.cyInstance, '_get_'+value)(self.post_rank)
        else:
            Global._error("Dendrite has no parameter/variable called", value)     
    
    
    def receptive_field(self, variable = 'value'):
        """
        Get the variable data as receptive field.
        
        Parameter:
        
        * *variable*: name of the variable (default = 'value')
        """
        values = getattr(self.proj.cyInstance, '_get_'+variable)(self.post_rank)
        ranks = self.proj.cyInstance._get_rank( self.post_rank )
             
        m = np.zeros( self.pre.size )
        m[ranks] = values

        return m.reshape(self.pre.geometry)

        
    # @property 
    # def learnable(self):
    #     return False
    
    # @learnable.setter
    # def learnable(self, value):
    #     pass
    #     #self.cy_instance.learnable = value

    # @property 
    # def learn_offset(self):
    #     return self.cy_instance.learn_offset
    
    # @learn_offset.setter
    # def learn_offset(self, value):
    #     self.cy_instance.learn_offset = value

    # @property 
    # def learn_frequency(self):
    #     return self.cy_instance.learn_frequency
    
    # @learn_frequency.setter
    # def learn_frequency(self, value):
    #     self.cy_instance.learn_frequency = value
        
        

    #########################
    ### Structural plasticity
    ######################### 
    def add_synapse(self, rank, value, delay=0):
        """
        Adds a synapse to the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
            * *value*:    synaptic weight
            * *delay*:    additional delay of the synapse (as default a delay of 1ms is assumed)
        """
        print 'TODO: not implemented yet'
        #self.cy_instance.add_synapse(rank, value, delay)
    
    def remove_synapse(self, rank):
        """
        Removes the synapse from the dendrite.
        
        Parameters:
        
            * *rank*:     rank of the presynaptic neuron
        """
        print 'TODO: not implemented yet'
        #self.cy_instance.remove_synapse(rank)

    #########################
    ### Recording
    #########################   
    def start_record(self, variable):
        """
        Start recording the given variables.
        
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
            
            if not var in self.proj._recordable_variables:
                print(var, 'is not a recordable variable of', self.proj.name)
                continue

            if not self.post_rank in self.proj._recorded_variables.keys(): # dendrite not recorded yet
                self.proj._recorded_variables[self.post_rank] = {var: Record(var)}
            elif not var in self.proj._recorded_variables[self.post_rank].keys(): 
                self.proj._recorded_variables[self.post_rank][var] = Record(var)
                
            self.proj._recorded_variables[self.post_rank][var].start()

            # if not self.proj._recorded_variables[var].is_inited:
            #     print('variable',var,'is not initialized.')
            #     continue
            
            try:
                getattr(self.proj.cyInstance, '_start_record_'+var)(self.post_rank)
    
                if Global.config['verbose']:
                    print('start recording of', var, '(', self.proj.name, ')')
                    
            except:
                print '_start_record_'+var
                print "Error (start_record): only possible after compilation."
                pass

    def stop_record(self, variable=None):
        """
        Stops recording the defined variables.

        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument is provided all records will stop.
        """
        _variable = []
        if variable == None:
            _variable = self.proj._recorded_variables[self.post_rank].keys() # TODO: what is it?
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print 'Error: variable must be either a string or list of strings.'       
        
        for var in _variable:
            
            if not var in self.proj._recordable_variables:
                print var, 'is not a recordable variable of', self.proj.name
                continue

            if not self.post_rank in self.proj._recorded_variables.keys() or \
               not var in self.proj._recorded_variables[self.post_rank].keys() or \
               not self.proj._recorded_variables[self.post_rank][var].is_running :
                print 'The recording of', var, 'was not running on projection', self.proj.name
                continue
            
            try:
                getattr(self.proj.cyInstance, '_stop_record_'+var)(self.post_rank)

                if Global.config['verbose']:
                    print('stop record of', var, '(', self.name, ')')
                self.proj._recorded_variables[self.post_rank][var].pause()
            except:
                Global._error('(pause_record): only possible after compilation.')

    def pause_record(self, variable=None):
        """
        Pause in recording the defined variables.

        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument is provided all records will stop.
        """
        _variable = []
        if variable == None:
            _variable = self.proj._recorded_variables[self.post_rank].keys() # TODO: what is it?
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')       
        
        for var in _variable:
            
            if not var in self.proj._recordable_variables:
                print(var, 'is not a recordable variable of', self.proj.name)
                continue

            if not self.post_rank in self.proj._recorded_variables.keys() or \
               not var in self.proj._recorded_variables[self.post_rank].keys() or \
               not self.proj._recorded_variables[self.post_rank][var].is_running :
                print('The recording of', var, 'was not running on projection', self.proj.name)
                continue
            
            try:
                getattr(self.proj.cyInstance, '_stop_record_'+var)(self.post_rank)

                if Global.config['verbose']:
                    print('pause record of', var, '(', self.name, ')')
                self.proj._recorded_variables[self.post_rank][var].pause()
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
            
            if not var in var in self.proj._recordable_variables:
                print(var, 'is not a recordable variable of', self.proj.name)
                continue
            
            if not self.post_rank in self.proj._recorded_variables.keys() or \
               not var in self.proj._recorded_variables[self.post_rank].keys() :
                print('The recording of', var, 'was not running on projection', self.proj.name)
                continue


            if not self.proj._recorded_variables[self.post_rank][var].is_running:
                print('Recording of', var, 'is already running on projection', self.proj.name)
                continue
            
            try:
                getattr(self.proj.cyInstance, '_start_record_'+var)(self.post_rank)
                
                if Global.config['verbose']:
                    print('resume record of', var, '(' , self.proj.name, ')')

                self.proj._recorded_variables[self.post_rank][var].start()
            except:
                print("Error: only possible after compilation.")
                
    def get_record(self, variable=None):
        """
        Returns the recorded data as one matrix or a dictionary if more then one variable is requested. 
        The first dimension is the neuron index, the last dimension represents the number of simulation steps.
        
        Parameter:
            
        * *variable*: single variable name or list of variable names. If no argument provided, the remaining recorded data is returned.  
        """        
        _variable = []
        if variable == None:
            _variable = self.proj._recorded_variables[self.post_rank].keys()
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        data_dict = {}
        
        for var in _variable:

            if not var in self.proj._recordable_variables:
                Global._print(var, 'is not a recordable variable of', self.proj.name)
                data_dict[var] = { 'start': -1, 'stop': -1, 'data': None }
                continue
            
            if not self.post_rank in self.proj._recorded_variables.keys() or \
               not var in self.proj._recorded_variables[self.post_rank].keys() :
                print('The recording of', var, 'was not running on projection', self.proj.name)
                continue

            if self.proj._recorded_variables[self.post_rank][var].is_running:
                self.pause_record(var)
            
            try:
                if Global.config['verbose']:
                    Global._print('get record of '+var+' ('+self.proj.name+')')
                    
                data = getattr(self.proj.cyInstance, '_get_recorded_'+var)(self.post_rank)
                
                data_dict[var] = { 
                    'data': data.T,
                    'start': self.proj._recorded_variables[self.post_rank][var].start_time,
                    'stop': self.proj._recorded_variables[self.post_rank][var].stop_time
                }
                                
                self.proj._recorded_variables[self.post_rank][var].reset()
            except:
                print("Error: only possible after compilation.")

        if( len(_variable)==1 and variable!=None):
            return data_dict[_variable[0]]
        else:
            return data_dict          
    


