"""

    Projection.py
    
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
import numpy as np

import Global
from Dendrite import Dendrite
from ANNarchy4 import generator
from ANNarchy4.core.Random import RandomDistribution
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Descriptor import Descriptor, Attribute

class Projection(Descriptor):
    """
    Python class representing the projection between two populations.
    """

    def __init__(self, pre, post, target, connector, synapse=None):
        """
        Constructor of a Projection object.

        Parameters:
                
            * *pre*: pre synaptic layer (either name or Population object)
            * *post*: post synaptic layer (either name or Population object)
            * *target*: connection type
            * *connector*: connection pattern object
            * *synapse*: synapse object
        """
        
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object 
        if isinstance(pre, str):
            for pop in Global._populations:
                if pop.name == pre:
                    self.pre = pop
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for pop in Global._populations:
                if pop.name == post:
                    self.post = pop
        else:
            self.post = post
            
        self.post.generator._add_target(target)
        self.target = target
        self.connector = connector
        self.connector.proj = self # set reference to projection
        self.synapse = synapse
        self._dendrites = []
        self._post_ranks = []

        self.name = 'Projection'+str(len(Global._projections))
        self.generator = generator.Projection(self, self.synapse)
        Global._projections.append(self)
        self.initialized = True
        
    def _init_attributes(self):
        """ Method used after compilation to initialize the attributes."""
        for var in self.variables + self.parameters:
            setattr(self, var, Attribute(var))
            
    def get(self, name):
        """ Returns a list of parameters/variables values for each dendrite in the projection.
        
        The list will have the same length as the number of actual dendrites (self.size), so it can be smaller than the size of the postsynaptic population. Use self.post_ranks to indice it.
        
        """
        ret=[]
        for dendrite in self.dendrites:
            ret.append(dendrite.get(name))        
        return np.array(ret)
            
    def set(self, value):
        """ Sets the parameters/variables values for each dendrite in the projection.
        
        For parameters, you can provide:
        
            * a single value, which will be the same for all dendrites.
            
            * a list or 1D numpy array of the same length as the number of actual dendrites (self.size).
            
        For variables, you can provide:
        
            * a single value, which will be the same for all synapses of all dendrites.
            
            * a list or 1D numpy array of the same length as the number of actual dendrites (self.size). The synapses of each postsynaptic neuron will have the same value.
            
            * a list of lists or 2D numpy array representing for each connected postsynaptic neuron, the value to be taken by each synapse. The first dimension must be self.size, while the second must correspond to the number of synapses in each particular dendrite.
            
        .. hint::
        
            In the latter case, it would be less error-prone to iterate over all dendrites in the projection:
            
            .. code-block:: python
            
                for dendrite in proj.dendrites:
                    dendrite.set( ... )    
        
        """
        
        for val_key in value.keys():
            if isinstance(value[val_key], list) or isinstance(value[val_key], np.ndarray):
                for rk in range(len(self._dendrites)):
                    self._dendrites[rk].set({val_key: value[val_key][rk] })
            else:
                for dendrite in self._dendrites:
                    dendrite.set({val_key: value[val_key] })
            
                
    def dendrite(self, pos):
        """
        Returns the dendrite of a postsynaptic neuron according to its rank.

        Parameters:

            * *pos*: could be either rank or coordinate of the requested postsynaptic neuron
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.post.rank_from_coordinates(pos)
        if rank in self._post_ranks:
            return self._dendrites[rank]
        else:
            print 'Error: neuron of rank', str(rank), 'has no synapse in this projection.'
            return None
    
    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns dendrite of the given position in the postsynaptic population. 
        
        If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        return self.dendrite(args[0])
        
    def __iter__(self):
        " Returns iteratively each dendrite in the population in ascending rank order."
        for n in range(self.size):
            yield self._dendrites(n)  
        
    @property
    def size(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return len(self._dendrites)
        
    def __len__(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return len(self._dendrites)
        
    @property
    def dendrites(self):
        """
        List of dendrites corresponding to this projection.
        """
        return self._dendrites
        
    @property
    def post_ranks(self):
        """
        List of postsynaptic neuron ranks having synapses in this projection.
        """
        return self._post_ranks

    def _parsed_variables(self):
        """
        Returns parsed variables in case of an attached synapse.
        """
        if self.synapse:
            return self.generator.parsed_variables
        else:
            return []
            
    @property
    def variables(self):
        """
        Returns a list of all variable names.
        """
        ret_var = ['rank','value', 'delay']
        
        # check for additional variables        
        for var in self._parsed_variables():
            if not var['type'] == 'parameter' and not var['name'] in ret_var:
                ret_var.append(var['name'])        
        return ret_var

    @property
    def parameters(self):
        """
        Returns a list of all parameter names.
        """
        ret_par = []
                
        for var in self._parsed_variables():
            if var['type'] == 'parameter' and not var['name'] in ret_par:
                ret_par.append(var['name'])    
                
        return ret_par

    def connect(self):
        
        self._dendrites, self._post_ranks = self.connector.connect()

    def gather_data(self, variable):
        blank_col=np.zeros((self.pre.geometry[1], 1))
        blank_row=np.zeros((1,self.post.geometry[0]*self.pre.geometry[0]+self.post.geometry[0] +1))
        
        m_ges = None
        i=0
        
        for y in xrange(self.post.geometry[1]):
            m_row = None
            
            for x in xrange(self.post.geometry[0]):
                m = self._dendrites[i].get_variable(variable)
                
                if m_row == None:
                    m_row = np.ma.concatenate( [ blank_col, m.reshape(self.pre.geometry[1], self.pre.geometry[0]) ], axis = 1 )
                else:
                    m_row = np.ma.concatenate( [ m_row, m.reshape(self.pre.geometry[1], self.pre.geometry[0]) ], axis = 1 )
                m_row = np.ma.concatenate( [ m_row , blank_col], axis = 1 )
                
                i += 1
            
            if m_ges == None:
                m_ges = np.ma.concatenate( [ blank_row, m_row ] )
            else:
                m_ges = np.ma.concatenate( [ m_ges, m_row ] )
            m_ges = np.ma.concatenate( [ m_ges, blank_row ] )
        
        return m_ges
