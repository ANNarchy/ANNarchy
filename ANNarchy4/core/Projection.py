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
import traceback
import numpy as np
import math

import pyximport
pyximport.install()
import cy_functions

from ANNarchy4.core import Global
from ANNarchy4.core.Neuron import RateNeuron, SpikeNeuron
from ANNarchy4.core.Synapse import RateSynapse, SpikeSynapse
from ANNarchy4.parser.Analyser import analyse_projection
from ANNarchy4.core.Dendrite import Dendrite

class Projection(object):
    """
    Python class representing the projection between two populations.
    """

    def __init__(self, pre, post, target, synapse=None):
        """
        Constructor of a Projection object.

        Parameters:
                
            * *pre*: pre synaptic layer (either name or Population object)
            * *post*: post synaptic layer (either name or Population object)
            * *target*: connection type
            * *synapse*: synapse description
        """
        if Global.get_projection(pre, post, target, suppress_error=True) != None:
            Global._error('population from', pre.name, 'to', post.name, 'with target', target, 'already exists.')
            exit(0)
        
        # Store the pre and post synaptic populations
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
            
        # Store the arguments
        self.target = target
        
        # Create a default name
        self._id = len(Global._projections)
        self.name = 'Projection'+str(self._id)
        
        # Add the target to the postsynaptic population
        self.post.targets.append(self.target)
        
        #
        # check if a synapse description is attached
        if not synapse:
            # No synapse attached assume default synapse based on
            # presynaptic population.
            if isinstance(self.pre.neuron_type, RateNeuron):
                self.synapse_type = RateSynapse(parameters = "", equations = "")
            else:
                self.synapse_type = SpikeSynapse(parameters = "", equations = "", pre_spike="g_target += value", post_spike="")
        else:
            self.synapse_type = synapse
            
        self._synapses = {}
        self._connector = None
        self._dendrites = []
        self._post_ranks = []

        # Get a list of parameters and variables
        self.description = analyse_projection(self)
        self.parameters = []
        self.init = {}
        for param in self.description['parameters']:
            self.parameters.append(param['name'])
            
            # TODO: 
            # pre evaluate init to 
            # transform expressions into their value
            self.init[param['name']] = param['init']
        self.variables = []
        for var in self.description['variables']:
            self.variables.append(var['name'])

            # TODO: 
            # pre evaluate init to 
            # transform expressions into their value
            self.init[var['name']] = var['init']
        self.attributes = self.parameters + self.variables
        
        
        # Add the population to the global variable
        Global._projections.append(self)
        
        # Finalize initialization
        self.initialized = False

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
            Global._error("neuron of rank", str(rank),"has no synapse in this projection.")
            return None
    
    def connect_one_to_one(self, weights=1.0, delays=0.0):
        """
        Establish one to one connections within the two projections.
        
        Parameters:
        
            * *weights*: synaptic value, either one value or a random distribution.
            * *delays*: synaptic delay, either one value or a random distribution.
        """
        synapses = {}
    
        for pre_neur in xrange(self.pre.size):
            try:
                w = weights.get_value()
            except:
                w = weights
                
            try:
                d = delays.get_value()
            except:
                d = delays
            self._synapses[(pre_neur, pre_neur)] = { 'w': w, 'd': d }
            
        return self
    
    def connect_all_to_all(self, weights, delays=0.0, allow_self_connections=False):
        """
        Establish all to all connections within the two projections.
        
        Parameters:
        
            * *weights*: synaptic value, either one value or a random distribution.
            * *delays*: synaptic delay, either one value or a random distribution.
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        allow_self_connections = (self.pre!=self.post) and not allow_self_connections
    
        for post_neur in xrange(self.post.size):
            for pre_neur in xrange(self.pre.size):
                if (pre_neur == post_neur) and not allow_self_connections:
                    continue
                
                try:
                    w = weights.get_value()
                except:
                    w = weights
                    
                try:
                    d = delays.get_value()
                except:
                    d = delays
    
                self._synapses[(pre_neur, post_neur)] = { 'w': w, 'd': d }
        
        return self

    def connect_gaussian(self, sigma, amp, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Establish all to all connections within the two projections.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same rank and width weights following a gaussians distribution.
        
        Parameters:
        
            * *weights*: synaptic value, either one value or a random distribution.
            * *delays*: synaptic delay, either one value or a random distribution.
            * *sigma*: sigma value
            * *amp*: amp value
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        allow_self_connections = (self.pre!=self.post) and not allow_self_connections
        synapses = {}
        
        for post_neur in xrange(self.post.size):
            normPost = self.post.normalized_coordinates_from_rank(post_neur)
            
            for pre_neur in range(self.pre.size):
                if (pre_neur == post_neur) and not allow_self_connections:
                    continue
    
                normPre = self.pre.normalized_coordinates_from_rank(pre_neur)
    
                dist = self._comp_dist(normPre, normPost)
                dist2 = cy_functions.comp_distND( normPre, normPost )
                
                value = amp * np.exp(-dist/2.0/sigma/sigma)
                if (abs(value) > limit * abs(amp)):
                        
                    try:
                        d = delays.get_value()
                    except:
                        d = delays
                    self._synapses[(pre_neur, post_neur)] = { 'w': value, 'd': d }   
                         
        return self

    @profile
    def connect_dog(self, sigma_pos, sigma_neg, amp_pos, amp_neg, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Establish all to all connections within the two projections.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same rank and width weights following a difference-of-gaussians distribution.
        
        Parameters:
        
            * *weights*: synaptic value, either one value or a random distribution.
            * *delays*: synaptic delay, either one value or a random distribution.
            * *sigma_pos*: sigma of positive gaussian function
            * *sigma_neg*: sigma of negative gaussian function
            * *amp_pos*: amp of positive gaussian function
            * *amp_neg*: amp of negative gaussian function
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        #
        # create pattern
        for post_neur in xrange(self.post.size):
            normPost = self.post.normalized_coordinates_from_rank(post_neur)
            
            for pre_neur in range(self.pre.size):
                if (pre_neur == post_neur) and not allow_self_connections:
                    continue
    
                normPre = self.pre.normalized_coordinates_from_rank(pre_neur)
    
                dist = self._comp_dist(normPre, normPost)
    
                value = amp_pos * np.exp(-dist/2.0/sigma_pos/sigma_pos) - amp_neg * np.exp(-dist/2.0/sigma_neg/sigma_neg)
                if ( abs(value) > limit * abs( amp_pos - amp_neg ) ):
                        
                    try:
                        d = delays.get_value()
                    except:
                        d = delays
                    self._synapses[(pre_neur, post_neur)] = { 'w': value, 'd': d }

        return self
    
    
    @profile
    def connect_dog_optimized(self, sigma_pos, sigma_neg, amp_pos, amp_neg, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Establish all to all connections within the two projections.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same rank and width weights following a difference-of-gaussians distribution.
        
        Parameters:
        
            * *weights*: synaptic value, either one value or a random distribution.
            * *delays*: synaptic delay, either one value or a random distribution.
            * *sigma_pos*: sigma of positive gaussian function
            * *sigma_neg*: sigma of negative gaussian function
            * *amp_pos*: amp of positive gaussian function
            * *amp_neg*: amp of negative gaussian function
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        #
        # preparation of comparison functions and        
        # choose the right function depending on population dimension
        comp_dict={
            1: cy_functions.comp_dist1D,
            2: cy_functions.comp_dist2D,
            3: cy_functions.comp_dist3D
        }
        try:
            # dim = (1 .. 3)
            comp_func = comp_dict[self.pre.dimension]
        except KeyError:
            # 4 and higher dimensionality
            comp_func = cy_functions.comp_distND

        if isinstance(delays, (int, float)):
            delay_values = [ delays for n in range(self.pre.size) ]
        
        #
        # create dog pattern by iterating over both populations
        #
        # 1st post ranks
        for post_neur in xrange(self.post.size):
            normPost = self.post.normalized_coordinates_from_rank_optimized(post_neur)

            # get a new array of random values
            if not isinstance(delays, (int, float)):
                delay_values = delays.get_values((self.pre.size,1))
                
            # set iterator on begin of the array
            delay_iter = iter(delay_values)
            
            # 2nd pre ranks
            for pre_neur in range(self.pre.size):
                if (pre_neur == post_neur) and not allow_self_connections:
                    continue
    
                normPre = self.pre.normalized_coordinates_from_rank_optimized(pre_neur)
                dist = comp_func( normPre, normPost )
    
                value = amp_pos * math.exp(-dist/2.0/sigma_pos/sigma_pos) - amp_neg * math.exp(-dist/2.0/sigma_neg/sigma_neg)
                
                #
                # important: math.fabs is only a well solution in float case
                if ( math.fabs(value) > limit * math.fabs( amp_pos - amp_neg ) ):
                    self._synapses[(pre_neur, post_neur)] = { 'w': value, 'd': next(delay_iter) }

        return self
        
    def connect_with_func(self, method, **args):
        """
        Establish connections provided by user defined function.

        * *method*: function handle. The function **need** to return a dictionary of synapses.
        * *args*: list of arguments needed by the function 
        """
        self._connector = method
        self._connector_params = args
        self._synapses = self._connector(self.pre, self.post, **self._connector_params)

        return self
      
    def get(self, name):
        """ Returns a list of parameters/variables values for each dendrite in the projection.
        
        The list will have the same length as the number of actual dendrites (self.size), so it can be smaller than the size of the postsynaptic population. Use self.post_ranks to indice it.        
        """       
        return self.__getattr__(name)
            
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
        
        for name, val in value:
            self.__setattr__(name, val)

    def _connect(self):
        """
        build up dendrites
        """
        self._dendrites, self._post_ranks = self._build_pattern_from_dict()

    def _comp_dist(self, pre, post):
        """
        Compute euklidean distance between two coordinates. 
        """
        res = 0.0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res
      
    def _build_pattern_from_dict(self):
        """
        build up the dendrites from the list of synapses
        """
        #
        # the synapse objects are stored as pre-post pairs.
        dendrites = {} 
        
        for conn, data in self._synapses.iteritems():
            try:
                dendrites[conn[1]]['rank'].append(conn[0])
                dendrites[conn[1]]['weight'].append(data['w'])
                dendrites[conn[1]]['delay'].append(data['d'])
            except KeyError:
                dendrites[conn[1]] = { 'rank': [conn[0]], 'weight': [data['w']], 'delay': [data['d']] }
        
        ret_value = []
        ret_ranks = []
        for post_id, data in dendrites.iteritems():
            ret_value.append(Dendrite(self, post_id, ranks = data['rank'], weights = data['weight'], delays = data['delay']))
            ret_ranks.append(post_id)
        
        return ret_value, ret_ranks

    def _gather_data(self, variable):
        """ 
        Gathers synaptic values for visualization
        
        *Paramater*:
        
            * *variable*: name of variable
        """
        blank_col=np.zeros((self.pre.geometry[1], 1))
        blank_row=np.zeros((1,self.post.geometry[0]*self.pre.geometry[0]+self.post.geometry[0] +1))
        
        m_ges = None
        i=0
        
        for y in xrange(self.post.geometry[1]):
            m_row = None
            
            for x in xrange(self.post.geometry[0]):
                m = self._dendrites[i].cy_instance.value
                
                if m.shape != self.pre.geometry:
                    new_m = np.zeros(self.pre.geometry[0]*self.pre.geometry[1])
                    
                    j = 0
                    for r in self._dendrites[i].cy_instance.rank:
                        new_m[r] = m[j]
                        j+=1 
                    m = new_m
                
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
                
    def _init_attributes(self):
        """ 
        Method used after compilation to initialize the attributes.
        """
        self.initialized = True  
        for attr in self.attributes:
            if attr in self.description['local']: # Only local variables are not directly initialized in the C++ code
                if isinstance(self.init[attr], list) or isinstance(self.init[attr], np.ndarray):
                    self._set_cython_attribute(attr, self.init[attr])

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif name == 'attributes':
            return object.__getattribute__(self, 'attributes')
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
                    if name in self.description['local']:
                        return self.init[name] # Dendrites are not initialized
                    else:
                        return self.init[name]
                else:
                    return self._get_cython_attribute( name)
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            object.__setattr__(self, name, value)
        elif name == 'attributes':
            object.__setattr__(self, name, value)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
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
        return np.array([getattr(dendrite, attribute) for dendrite in self._dendrites])
        
    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all neurons in the population, 
        as a NumPy array having the same geometry as the population if it is local.
        
        Parameter:
        
        * *attribute*: should be a string representing the variables's name.
        
        """
        if isinstance(value, np.ndarray):
            if value.dim == 1:
                if value.shape == (self.size, ):
                    for n in range(self.size):
                        setattr(self._dendrites[n], attribute, value[n])
                else:
                    Global._error('The projection has '+self.size+ ' dendrites.')
        elif isinstance(value, list):
            if len(value) == self.size:
                for n in range(self.size):
                    setattr(self._dendrites[n], attribute, value[n])
            else:
                Global._error('The projection has '+self.size+ ' dendrites.')
        else:
            for dendrite in self._dendrites:
                setattr(dendrite, attribute,  value)
           
    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns dendrite of the given position in the postsynaptic population. 
        
        If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        return self.dendrite(args[0])
        
    def __iter__(self):
        " Returns iteratively each dendrite in the population in ascending rank order."
        for n in range(self.size):
            yield self._dendrites[n] 
