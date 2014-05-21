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

from ANNarchy.core import Global
from ANNarchy.core.Neuron import RateNeuron, SpikeNeuron
from ANNarchy.core.Synapse import RateSynapse, SpikeSynapse
from ANNarchy.parser.Analyser import analyse_projection
from ANNarchy.core.Dendrite import Dendrite
from ANNarchy.core.Record import Record

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

        self.type_prefix = "Rate" if isinstance(self.synapse_type, RateSynapse) else "Spike"

        # Create a default name
        self._id = len(Global._projections)
        self.name = self.type_prefix+'Projection'+str(self._id)
            
        self._synapses = {}
        self._connector = None
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
        
        # Allow recording of variables
        self._recorded_variables = {}  
        self._recordable_variables = list(set(self.variables + ['value']))

        # Finalize initialization
        self.initialized = False

        # Cython instance
        self.cyInstance = None

    ################################
    ## Dendrite access
    ################################
    @property
    def size(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        if self.cyInstance:
            return self.cyInstance._nb_dendrites
        else:
            return 0
        
    def __len__(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return self.size

    @property
    def dendrites(self):
        """
        Iteratively returns a list of dendrites corresponding to this projection.
        """
        for n in self._post_ranks:
            yield Dendrite(self, n)
        
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

            * *pos*: could be either the rank or the coordinates of the requested postsynaptic neuron
        """
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.post.rank_from_coordinates(pos)
        if rank in self._post_ranks:
            return Dendrite(self, rank)
        else:
            Global._error(" The neuron of rank "+ str(rank) + " has no dendrite in this projection.")
            return None
    

    # Iterators
    def __getitem__(self, *args, **kwds):
        """ Returns dendrite of the given position in the postsynaptic population. 
        
        If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        if len(args) == 1:
            return self.dendrite(args[0])
        return self.dendrite(args)
        
    def __iter__(self):
        " Returns iteratively each dendrite in the population in ascending postsynaptic rank order."
        for n in self._post_ranks:
            yield Dendrite(self, n)

    ################################
    ## Access to attributes
    ################################

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
                    return self._get_cython_attribute( name )
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
        return np.array([getattr(self.cyInstance, '_get_'+attribute)(i) for i in self._post_ranks])
        
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
                    for n in self._post_ranks:
                        getattr(self.cyInstance, '_set_'+attribute)(n, value)
                else:
                    Global._error('The projection has '+self.size+ ' dendrites.')
        elif isinstance(value, list):
            if len(value) == self.size:
                for n in self._post_ranks:
                    getattr(self.cyInstance, '_set_'+attribute)(n, value)
            else:
                Global._error('The projection has '+self.size+ ' dendrites.')
        else: # a single value
            if attribute in self.description['local']:
                for i in self._post_ranks:
                    getattr(self.cyInstance, '_set_'+attribute)(i, value*np.ones(self.size))
            else:
                for i in self._post_ranks:
                    getattr(self.cyInstance, '_set_'+attribute)(i, value)

 
    ################################
    ## Variable flags
    ################################           
    def set_variable_flags(self, name, value):
        """ Sets the flags of a variable for the projection.
        
        If the variable ``rate`` is defined in the Synapse description through:
        
            value = pre.rate * post.rate : max=1.0  
            
        one can change its maximum value with:
        
            proj.set_variable_flags('value', {'max': 2.0})
            
        For valued flags (init, min, max), ``value`` must be a dictionary containing the flag as key ('init', 'min', 'max') and its value. 
        
        For positional flags (postsynaptic, implicit), the value in the dictionary must be set to the empty string '':
        
            proj.set_variable_flags('value', {'implicit': ''})
        
        A None value in the dictionary deletes the corresponding flag:
        
            proj.set_variable_flags('value', {'max': None})
            
        """
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The projection '+self.name+' has no variable called ' + name)
            return
            
        for key, val in value.iteritems():
            if val == '': # a flag
                try:
                    self.description['variables'][rk_var]['flags'].index(key)
                except: # the flag does not exist yet, we can add it
                    self.description['variables'][rk_var]['flags'].append(key)
            elif val == None: # delete the flag
                try:
                    self.description['variables'][rk_var]['flags'].remove(key)
                except: # the flag did not exist, check if it is a bound
                    if has_key(self.description['variables'][rk_var]['bounds'], key):
                        self.description['variables'][rk_var]['bounds'].pop(key)
            else: # new value for init, min, max...
                if key == 'init':
                    self.description['variables'][rk_var]['init'] = val 
                    self.init[name] = val              
                else:
                    self.description['variables'][rk_var]['bounds'][key] = val
                
       
            
    def set_variable_equation(self, name, equation):
        """ Changes the equation of a variable for the projection.
        
        If the variable ``value`` is defined in the Synapse description through:
        
            eta * dvalue/dt = pre.rate * post.rate 
            
        one can change the equation with:
        
            proj.set_variable_equation('value', 'eta * dvalue/dt = pre.rate * (post.rate - 0.1) ')
            
        Only the equation should be provided, the flags have to be changed with ``set_variable_flags()``.
        
        .. warning::
            
            This method should be used with great care, it is advised to define another Synapse object instead. 
            
        """         
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The projection '+self.name+' has no variable called ' + name)
            return               
        self.description['variables'][rk_var]['eq'] = equation    
            
            
    def _find_variable_index(self, name):
        " Returns the index of the variable name in self.description['variables']"
        for idx in range(len(self.description['variables'])):
            if self.description['variables'][idx]['name'] == name:
                return idx
        return -1

    ################################
    ## Learning flags
    ################################
    def enable_learning(self, params={ 'freq': 1, 'offset': 0} ):
        """
        Enable the learning for all attached dendrites
        
        Parameter:
        
            * *params*: optional parameter to configure the learning
        """
        self.cyInstance._set_learning(True)
        self.cyInstance._set_learn_frequency(params['freq'])
        self.cyInstance._set_learn_offset(params['offset'])
            
    def disable_learning(self):
        """
        Disable the learning for all attached dendrites
        """
        self.cyInstance._set_learning(False)


    
    ################################
    ## Connector methods
    ################################
    
    def connect_one_to_one(self, weights=1.0, delays=0.0):
        """
        Establish a one-to-one connections between the two populations.
        
        Parameters:
        
            * *weights*: initial synaptic values, either a single value (float) or a random distribution object.
            * *delays*: synaptic delays, either one value (float or int) or a random distribution object.
        """
        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.one_to_one(self.pre.size, self.post.size, weights, delays)
        return self
    
    def connect_all_to_all(self, weights, delays=0.0, allow_self_connections=False):
        """
        Establishes an all-to-all connection pattern between the two populations.
        
        Parameters:
        
            * *weights*: synaptic values, either one value (float) or a random distribution object.
            * *delays*: synaptic delays, either one value (float or int) or a random distribution object.
            * *allow_self_connections*: if True, self-connections between a neuron and itself are allowed (default=False).
        """
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.all_to_all(self.pre.size, self.post.size, weights, delays, allow_self_connections)

        return self

    def connect_gaussian(self, amp, sigma, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Establish a Guassian conneciton pattern between the two populations.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same normalized coordinates using a Gaussian distribution.
        
        Parameters:
        
            * *amp*: amp value
            * *sigma*: sigma value
            * *delays*: synaptic delay, either a single value or a random distribution object (default : 0.0).
            * *limit*: proportion of *amp* below which synapses are not created (default: 0.01)
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.gaussian(self.pre.geometry, self.post.geometry, amp, sigma, delays, limit, allow_self_connections)

        return self
    
    def connect_dog(self, amp_pos, sigma_pos, amp_neg, sigma_neg, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Establish all to all connections within the two projections.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same rank and width weights following a difference-of-gaussians distribution.
        
        Parameters:
        
            * *amp_pos*: amp of positive gaussian function
            * *sigma_pos*: sigma of positive gaussian function
            * *amp_neg*: amp of negative gaussian function
            * *sigma_neg*: sigma of negative gaussian function
            * *delays*: synaptic delay, either a single value or a random distribution.
            * *allow_self_connections*: set to True, if you want to allow connections within equal neurons in the same population.
        """
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.dog(self.pre.geometry, self.post.geometry, amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections)


        return self

    def connect_fixed_probability(self, probability, weights, delays=0.0, allow_self_connections=False):
        """ fixed_probability projection between two populations. 
    
        Each neuron in the postsynaptic population is connected to neurons of the presynaptic population with a fixed probability. Self-connections are avoided by default.
    
        Parameters:
        
        * probability: probability that an individual synapse is created.
        
        * weights: either a single value for all synapses or a RandomDistribution object.
        
        * delays: either a single value for all synapses or a RandomDistribution object (default = 0.0)
        
        * allow_self_connections : defines if self-connections are allowed (default=False).
        """         
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_probability(self.pre.size, self.post.size, probability, weights, delays, allow_self_connections)

        return self

    def connect_fixed_number_pre(self, number, weights, delays=0.0, allow_self_connections=False):
        """ fixed_number_pre projection between two populations.
    
        Each neuron in the postsynaptic population receives connections from a fixed number of neurons of the presynaptic population chosen randomly. 
    
        Parameters:
        
        * number: number of synapses per postsynaptic neuron.
        
        * weights: either a single value for all synapses or a RandomDistribution object.
        
        * delays: either a single value for all synapses or a RandomDistribution object (default = 0.0)
        
        * allow_self_connections : defines if self-connections are allowed (default=False).
        """
        if self.pre!=self.post:
            allow_self_connections = True
        
        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_number_pre(self.pre.size, self.post.size, number, weights, delays, allow_self_connections)

        return self
            
    def connect_fixed_number_post(self, number, weights=1.0, delays=0.0, allow_self_connections=False):
        """ fixed_number_post projection between two populations.
    
        Each neuron in the presynaptic population sends connections to a fixed number of neurons of the postsynaptic population chosen randomly. 
    
        Parameters:
        
        * number: number of synapses per presynaptic neuron.
        
        * weights: either a single value for all synapses or a RandomDistribution object.
        
        * delays: either a single value for all synapses or a RandomDistribution object (default = 0.0)
        
        * allow_self_connections : defines if self-connections are allowed (default=False).
        """
        if self.pre!=self.post:
            allow_self_connections = True
        
        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_number_post(self.pre.size, self.post.size, number, weights, delays, allow_self_connections)

        return self

    def connect_from_list(self, connection_list):
        """
        Initialize projection initilized by specific list.
        
        Expected format:
        
            * list of tuples ( pre, post, weight, delay )
            
        Example:
        
            >>> conn_list = [
            ...     ( 0, 0, 0.0, 0.1 )
            ...     ( 0, 1, 0.0, 0.1 )
            ...     ( 0, 2, 0.0, 0.1 )
            ...     ( 1, 5, 0.0, 0.1 )
            ... ]
            
            proj = Projection(pre_pop, post_pop, 'exc').connect_from_list( conn_list )
        """
        self._synapses = connection_list
                
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

    def save_connectivity_as_csv(self):
        """
        Save the projection pattern as csv format. 
        Please note, that only the pure connectivity data pre_rank, post_rank, value and delay are stored.
        """
        filename = self.pre.name + '_' + self.post.name + '_' + self.target+'.csv'
        
        with open(filename, mode='w') as w_file:
            
            for dendrite in self.dendrites:
                rank_iter = iter(dendrite.rank)
                value_iter = iter(dendrite.value)
                delay_iter = iter(dendrite.delay)
                post_rank = dendrite.post_rank

                for i in xrange(dendrite.size()):
                    w_file.write(str(next(rank_iter))+', '+
                                 str(post_rank)+', '+
                                 str(next(value_iter))+', '+
                                 str(next(delay_iter))+'\n'
                                 )
      

    def _connect(self):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """

        cython_module = __import__('ANNarchyCython') 
        proj = getattr(cython_module, 'py'+self.name)
        self.cyInstance = proj(self.pre._id, self.post._id, self.post.targets.index(self.target))
        
        # Sort the dendrites to be created based on _synapses
        import ANNarchy.core.cython_ext.Connector as Connector
        if ( isinstance(self._synapses, Connector.CSR) ):
            self.cyInstance.createFromCSR(self._synapses)
            self._post_ranks = self._synapses.keys()
            return
        elif ( isinstance(self._synapses, list) ):
        	dendrites = self._build_pattern_from_list()
        else:
        	dendrites = self._build_pattern_from_dict()

        # Store the keys of dendrites in _post_ranks
        self._post_ranks = dendrites.keys()

        # Create the dendrites in Cython
        self.cyInstance.createFromDict(dendrites)

        # Delete the _synapses array, not needed anymore
        del self._synapses
        self._synapses = None

    def _comp_dist(self, pre, post):
        """
        Compute euclidean distance between two coordinates. 
        """
        res = 0.0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res
      
    def _build_pattern_from_dict(self):
        """
        build up the dendrites from the dictionary of synapses
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
        
        return dendrites
    
    def _build_pattern_from_list(self):
        """
        build up the dendrites from the list of synapses
        """
        dendrites = {} 
        
        for conn in self._synapses:
            try:
                dendrites[conn[1]]['rank'].append(conn[0])
                dendrites[conn[1]]['weight'].append(conn[2])
                dendrites[conn[1]]['delay'].append(conn[3])
            except KeyError:
                dendrites[conn[1]] = { 'rank': [conn[0]], 'weight': [conn[2]], 'delay': [conn[3]] }

        return dendrites

    def receptive_fields(self, variable = 'value', in_post_geometry = True):
        """ 
        Gathers all receptive fields within this projection.
        
        *Parameters*:
        
            * *variable*: name of variable
            * *in_post_geometry*: if set to false, the data will be plotted as square grid. (default = True)
        """        
        if in_post_geometry:
            x_size = self.post.geometry[1]
            y_size = self.post.geometry[0]
        else:
            x_size = int( math.floor(math.sqrt(self.post.size)) )
            y_size = int( math.ceil(math.sqrt(self.post.size)) )
        


        def get_rf(rank):
            if rank in self._post_ranks:
                return self.dendrite(rank).receptive_field(variable)
            else:
                return np.zeros( self.pre.geometry )

        res = np.zeros((1, x_size*self.pre.geometry[1]))
        for y in xrange ( y_size ):
            row = np.concatenate(  [ get_rf(self.post.rank_from_coordinates( (y, x) ) ) for x in range ( x_size ) ], axis = 1)
            res = np.concatenate((res, row))
        
        return res
                

