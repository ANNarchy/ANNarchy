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
from Descriptor import Descriptor, Attribute
from ANNarchy4.core.Random import RandomDistribution

import traceback

class Dendrite(Descriptor):
    """
    A dendrite encapsulates all synapses of one neuron.
    
    *Hint*: this class will be created from projection class.
    
    Parameter:
    
    * *proj*: projection instance
    * *post_rank*: rank of the postsynaptic neuron
    * *cython_instance*: instance of the cythonized dendrite class.
    """
    def __init__(self, proj, post_rank, cython_instance=None, ranks=None, weights=None, delays=None):
        Descriptor.__init__(self)
        
        self.post_rank = post_rank
        self.proj = proj
        self.pre = proj.pre

        if cython_instance != None:
            self.cy_instance = cython_instance
        else:
            cython_module = __import__('ANNarchyCython')
            proj_id = self.proj.generator.proj_class['ID']   
            proj_class_name = 'LocalProjection'+str(proj_id)
            local_proj = getattr(cython_module, proj_class_name)
            
            self.cy_instance = local_proj(
                proj_id, 
                self.proj.pre.rank, 
                self.proj.post.rank, 
                post_rank, 
                self.proj.post.generator.targets.index(self.proj.target) 
            )

            self.cy_instance.rank = ranks
            self.cy_instance.value = weights
            if delays != None:
                self.cy_instance.delay = delays
            
        # synapse variables           
        for value in self.variables + self.parameters:
            setattr(self, value, Attribute(value))   
    
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
                print "Error: dendrite has no parameter/variable called", val_key+"."    
                
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
            print "Error: dendrite has no parameter/variable called", value+"."     
               
    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        return self.proj.variables

    @property
    def parameters(self):
        """
        Returns a list of all parameter names.
        """
        return self.proj.parameters
    
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
            print 'Error: variable', variable, 'does not exist in this dendrite.'
            print traceback.print_stack()

    def get_parameter(self, parameter):
        """
        Returns the value of the given parameter, which is common for all synapses in the dendrite.
        
        Parameter:
        
        * *parameter*:    a string representing the parameter's name.
        """
        if hasattr(self.cy_instance, parameter):
            return getattr(self.cy_instance, parameter)
        else:
            print 'Error: parameter', parameter, 'does not exist in this dendrite.'
            print traceback.print_stack()

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

