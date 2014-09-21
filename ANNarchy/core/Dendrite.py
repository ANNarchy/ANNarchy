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
                if isinstance(value, np.ndarray):
                    getattr(self.proj.cyInstance, 'set_dendrite_'+name)(self.post_rank, value)
                elif isinstance(value, list):
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