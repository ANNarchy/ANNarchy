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
                self.proj.post.targets.index(self.proj.target) 
            )
        else:
            self.cy_instance = local_proj(
                self.proj._id, 
                self.proj.post.rank, 
                self.proj.pre.rank, 
                post_rank, 
                self.proj.post.targets.index(self.proj.target) 
            )
                    
        self.cy_instance.rank = ranks
        self.cy_instance.value = weights
        if delays != None:
            self.cy_instance.delay = delays
            max_delay = np.amax(delays)
            self.proj.pre.cyInstance.set_max_delay(int(max_delay))

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

