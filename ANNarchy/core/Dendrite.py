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

import numpy as np

class Dendrite(object):
    """    
    A ``Dendrite`` is a sub-group of a ``Projection``, gathering the synapses between the pre-synaptic population and a single post-synaptic neuron. 

    It can not be created directly, only through a call to ``Projection.dendrite(rank)``::

        dendrite = proj.dendrite(6)
    """
    def __init__(self, proj, post_rank):

        self.post_rank = post_rank
        self.proj = proj
        self.pre = proj.pre

        self.target = self.proj.target
        
        self.parameters = self.proj.parameters
        self.variables = self.proj.variables

    @property
    def size(self):
        """
        Number of synapses.
        """
        if self.proj.cyInstance:
            return self.proj.cyInstance.nb_synapses(self.post_rank)
        return 0

    def __len__(self):
        """
        Number of synapses.
        """
        return self.size
    @property
    def synapses(self):
        """
        Iteratively returns the synapses corresponding to this dendrite.
        """
        for n in self.rank:
            yield Synapse(self, n)
    
    def synapse(self, pos):
        """
        Returns the synapse coming from the corresponding presynaptic neuron.

        *Parameters*:

            * **pos**: can be either the rank or the coordinates of the presynaptic neuron
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.proj.pre.rank_from_coordinates(pos)
            
        if rank in self.rank:
            return IndividualSynapse(self, rank)
        else:
            Global._error(" The neuron of rank "+ str(rank) + " has no synapse in this dendrite.")
            return None
    

    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns the synapse of the given position in the presynaptic population. 
        
        If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        if len(args) == 1:
            return self.synapse(args[0])
        return self.synapse(args)
        
    def __iter__(self):
        " Returns iteratively each synapse in the dendrite in ascending presynaptic rank order."
        for n in self.rank:
            yield IndividualSynapse(self, n)

    #########################
    ### Access to attributes
    #########################
    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if name == 'proj':
            return object.__getattribute__(self, name)
        elif hasattr(self, 'proj'):
            if name == 'rank':
                return self.proj.cyInstance.pre_rank(self.post_rank)
            elif name in self.proj.attributes:
                return getattr(self.proj.cyInstance, 'get_dendrite_'+name)(self.post_rank)
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
                if isinstance(value, (np.ndarray, list)):
                    getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.post_rank, value)
                else :
                    getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.post_rank, value * np.ones(self.size))
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def set(self, value):
        """
        Sets the value of a parameter/variable of all synapses.
        
        *Parameter*:
        
            * **value**: a dictionary containing the parameter/variable names as keys::
            
                dendrite.set( 'tau' : 20, 'w'= Uniform(0.0, 1.0) } )
        """
        for val_key in value.keys():
            if hasattr(self.proj.cy_instance, val_key):
                # Check the type of the data!!
                if isinstance(value[val_key], RandomDistribution):
                    val = value[val_key].getValues(self.size) 
                else: 
                    val = value[val_key]           
                # Set the value
                getattr(self.proj.cyInstance, 'set_dendrite_'+val_key)(self.post_rank, val)
            else:
                Global._error("Dendrite has no parameter/variable called", val_key)
                
    def get(self, name):
        """
        Returns the value of a variable/parameter.
        
        *Parameter*:
        
            * *name*: name of the parameter/variable::

                dendrite.get('w')
        """
        if name == 'rank':
            return self.proj.cyInstance.pre_rank(self.post_rank)
        elif name in self.attributes:
            return getattr(self.proj.cyInstance, 'get_dendrite_'+name)(self.post_rank)
        else:
            Global._error("Dendrite has no parameter/variable called", name)     
    

    #########################
    ### Formatting
    #########################     
    def receptive_field(self, variable = 'w', fill = 0.0):
        """
        Returns the given variable as a receptive field.

        A Numpy array of the same geometry as the pre-synaptic population is returned. Non-existing synapses are replaced by zeros (or the value ``fill``).
        
        *Parameter*:
        
        * **variable**: name of the variable (default = 'w')
        * **fill**: value to use when a synapse does not exist.
        """
        values = getattr(self.proj.cyInstance, 'get_dendrite_'+variable)(self.post_rank)
        ranks = self.proj.cyInstance._get_rank( self.post_rank )
             
        m = fill * np.ones( self.pre.size )
        m[ranks] = values

        return m.reshape(self.pre.geometry)

    #########################
    ### Structural plasticity
    ######################### 
    def add_synapse(self, rank, w, delay=0):
        """
        Adds a synapse to the dendrite.
        
        *Parameters*:
        
        * **rank**: rank of the pre-synaptic neuron
        * **w**: synaptic weight
        * **delay**: synaptic delay (default = dt)
        """
        if not Global.config['structural_plasticity']:
            Global._error('"structural_plasticity" has not been set to True in setup(), can not add the synapse.')
            return 

        if rank in self.rank:
            Global._error('the synapse of rank ' + str(rank) + ' already exists.')
            return             

        # Set default values for the additional variables
        extra_attributes = {}
        for var in self.proj.synapse.description['parameters'] + self.proj.synapse.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in self.proj.synapse.description['local']:
                if not isinstance(self.proj.init[var['name']], (int, float, bool)):
                    init = var['init']
                else:
                    init = self.proj.init[var['name']]
                extra_attributes[var['name']] = init

        self.proj.cyInstance.add_synapse(self.post_rank, rank, w, int(delay/Global.config['dt']), **extra_attributes)
    
    def remove_synapse(self, rank):
        """
        Removes the synapse from the dendrite.
        
        *Parameters*:
        
        * **rank**: rank of the pre-synaptic neuron
        """
        if not Global.config['structural_plasticity']:
            Global._error('"structural_plasticity" has not been set to True in setup(), can not remove the synapse.')
            return 

        if not rank in self.rank:
            Global._error('the synapse of rank ' + str(rank) + ' did not already exist.')
            return   

        self.proj.cyInstance.remove_synapse(self.post_rank, rank)

    #########################
    ### Recording
    #########################   
    def start_record(self, variable):
        """
        Starts recording the given variables.
        
        **Parameter**:
            
        * **variable**: single variable name or list of variable names.        
        """
        _variable = []
        
        if isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        for var in _variable:            
            try:
                getattr(self.proj.cyInstance, 'start_record_'+var)(self.post_rank)
                    
            except Exception, e:
                print e
                print "Error (start_record): only possible after compilation."
                return


            if not self.post_rank in self.proj.recorded_variables.keys():
                self.proj.recorded_variables[self.post_rank] = {}
            
            self.proj.recorded_variables[self.post_rank][var]= {'start': [Global.get_current_step()], 'stop': [-1]}

            if Global.config['verbose']:
                print('start recording of', var, '(', self.proj.name, ')')

    def stop_record(self, variable=None):
        """
        Stops recording the defined variables.

        *Parameter*:
            
        * **variable**: single variable name or list of variable names. If no argument is provided all recordings will stop.
        """
        _variable = []
        if variable == None:
            _variable = self.proj.recorded_variables[self.post_rank].keys() 
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print 'Error: variable must be either a string or list of strings.'       
        
        for var in _variable:
            
            try:
                getattr(self.proj.cyInstance, 'stop_record_'+var)(self.post_rank)

            except:
                Global._error('(stop_record): only possible after compilation.')

    # def pause_record(self, variable=None):
    #     """
    #     Pause in recording the defined variables.

    #     *Parameter*:
            
    #     * **variable**: single variable name or list of variable names. If no argument is provided all records will stop.
    #     """
    #     _variable = []
    #     if variable == None:
    #         _variable = self.proj._recorded_variables[self.post_rank].keys() # TODO: what is it?
    #     elif isinstance(variable, str):
    #         _variable.append(variable)
    #     elif isinstance(variable, list):
    #         _variable = variable
    #     else:
    #         print('Error: variable must be either a string or list of strings.')       
        
    #     for var in _variable:
            
    #         if not var in self.proj._recordable_variables:
    #             print(var, 'is not a recordable variable of', self.proj.name)
    #             continue

    #         if not self.post_rank in self.proj._recorded_variables.keys() or \
    #            not var in self.proj._recorded_variables[self.post_rank].keys() or \
    #            not self.proj._recorded_variables[self.post_rank][var].is_running :
    #             print('The recording of', var, 'was not running on projection', self.proj.name)
    #             continue
            
    #         try:
    #             getattr(self.proj.cyInstance, '_stop_record_'+var)(self.post_rank)

    #             if Global.config['verbose']:
    #                 print('pause record of', var, '(', self.name, ')')
    #             self.proj._recorded_variables[self.post_rank][var].pause()
    #         except:
    #             print("Error (pause_record): only possible after compilation.")

    # def resume_record(self, variable):
    #     """
    #     Resume recording the previous defined variables.
        
    #     *Parameter*:
            
    #     * **variable**: single variable name or list of variable names.        
    #     """
    #     _variable = []
        
    #     if isinstance(variable, str):
    #         _variable.append(variable)
    #     elif isinstance(variable, list):
    #         _variable = variable
    #     else:
    #         print('Error: variable must be either a string or list of strings.')
        
    #     for var in _variable:
            
    #         if not var in var in self.proj._recordable_variables:
    #             print(var, 'is not a recordable variable of', self.proj.name)
    #             continue
            
    #         if not self.post_rank in self.proj._recorded_variables.keys() or \
    #            not var in self.proj._recorded_variables[self.post_rank].keys() :
    #             print('The recording of', var, 'was not running on projection', self.proj.name)
    #             continue


    #         if not self.proj._recorded_variables[self.post_rank][var].is_running:
    #             print('Recording of', var, 'is already running on projection', self.proj.name)
    #             continue
            
    #         try:
    #             getattr(self.proj.cyInstance, '_start_record_'+var)(self.post_rank)
                
    #             if Global.config['verbose']:
    #                 print('resume record of', var, '(' , self.proj.name, ')')

    #             self.proj._recorded_variables[self.post_rank][var].start()
    #         except:
    #             print("Error: only possible after compilation.")
                
    def get_record(self, variable=None):
        """
        Returns the recorded data as one matrix or a dictionary if more then one variable is requested. 
        The first dimension is the neuron index, the last dimension represents the number of simulation steps.
        
        *Parameter*:
            
        * **variable**: single variable name or list of variable names. If no argument provided, the remaining recorded data is returned.  
        """        
        _variable = []
        if variable == None:
            _variable = self.proj.recorded_variables[self.post_rank].keys() 
        elif isinstance(variable, str):
            _variable.append(variable)
        elif isinstance(variable, list):
            _variable = variable
        else:
            print('Error: variable must be either a string or list of strings.')
        
        data_dict = {}
        
        for var in _variable:
            try:                    
                data = getattr(self.proj.cyInstance, 'get_recorded_'+var)(self.post_rank)
            except Exception, e:
                print e
                print "Error: only possible after compilation."
                return

            self.proj.recorded_variables[self.post_rank][var]['stop'][-1] = Global.get_current_step()

            data_dict[var] = {
                'start': self.proj.recorded_variables[self.post_rank][var]['start'] \
                        if len(self.proj.recorded_variables[self.post_rank][var]['start']) > 2 \
                        else self.proj.recorded_variables[self.post_rank][var]['start'][0],
                'stop' : self.proj.recorded_variables[self.post_rank][var]['stop'] \
                        if len(self.proj.recorded_variables[self.post_rank][var]['stop']) > 1 \
                        else self.proj.recorded_variables[self.post_rank][var]['stop'][0] ,
                'data' : data
            }

            self.proj.recorded_variables[self.post_rank][var]['start'][-1] = Global.get_current_step()
            
        return data_dict          
    

class IndividualSynapse(object):

    def __init__(self, dendrite, rank):
        self.dendrite = dendrite
        self.rank = rank
        self.idx = self.dendrite.rank.index(rank)
        self.attributes = self.dendrite.proj.synapse.description['local']

    def __getattr__(self, name):
        " Method called when accessing an attribute."

        if name in ['dendrite', 'attributes', 'rank', 'idx']:
            return object.__getattribute__(self, name)
        if name in self.attributes:
            return getattr(self.dendrite.proj.cyInstance, 'get_synapse_'+name)(self.dendrite.post_rank, self.idx)
        return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name in ['dendrite', 'attributes', 'rank', 'idx']:
            object.__setattr__(self, name, value)
        elif name in self.attributes:
                getattr(self.dendrite.proj.cyInstance, 'set_synapse_'+name)(self.dendrite.post_rank, self.idx, value)
        else:
            object.__setattr__(self, name, value)

