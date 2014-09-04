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
import math

from ANNarchy.core import Global
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.PopulationView import PopulationView

class Projection(object):
    """
    Represents all synapses of the same type between two populations.
    """

    def __init__(self, pre, post, target, synapse=None):
        """
        *Parameters*:
                
            * **pre**: pre-synaptic population (either its name or a ``Population`` object).
            * **post**: post-synaptic population (either its name or a ``Population`` object).
            * **target**: type of the connection.
            * **synapse**: a ``Synapse`` instance.

        By default, the synapse only ensures synaptic transmission:

        * For rate-coded populations: ``psp = w * pre.r``
        * For spiking populations: ``g_target += w``

        """
        
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

        # check if a synapse description is attached
        if not synapse:
            # No synapse attached assume default synapse based on
            # presynaptic population.
            if self.pre.neuron.type == 'rate':
                self.synapse = Synapse(parameters = "w=0.0", equations = "")
            else:
                self.synapse = Synapse(parameters = "w=0.0", equations = "", pre_spike="g_target += w", post_spike="")
        else:
            self.synapse = synapse

        self.synapse._analyse()

        # Create a default name
        self.id = len(Global._projections)
        self.name = 'proj'+str(self.id)
            
        self._connector = None

        # Get a list of parameters and variables
        self.parameters = []
        self.init = {}
        for param in self.synapse.description['parameters']:
            self.parameters.append(param['name'])
            self.init[param['name']] = param['init']

        self.variables = []
        for var in self.synapse.description['variables']:
            self.variables.append(var['name'])
            self.init[var['name']] = var['init']

        self.attributes = self.parameters + self.variables 
        if not 'w' in self.attributes:
            self.attributes.append('w')
            self.variables.append('w')
            self.synapse.description['local'].append('w')
        
        # Add the population to the global variable
        Global._projections[self.name] = self

        # Finalize initialization
        self.initialized = False

        # Cython instance
        self.cyInstance = None

        # CSR object
        self._synapses = None

    def _instantiate(self, module):

        self._connect(module)

        self.initialized = True  

        self._init_attributes()

    def _init_attributes(self):
        """ 
        Method used after compilation to initialize the attributes.
        """
        for name, val in self.init.iteritems():
            if not name in ['w']:
                self.__setattr__(name, val)

    def reset(self, synapses=False):
        """
        Resets all parameters and variables to the value they had before the call to compile.

        *Parameters*:

        * **synapses**: if True, the connections will also be erased (default: False).

        .. note::

            Not implemented yet...
        """
        self._init_attributes()
        if synapses:
            Global._warning('not implemented yet...')
    
    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """        
        if not self._synapses:
            Global._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not instantiated.')
            exit(0)

        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self._synapses)

        # Access the list of postsynaptic neurons
        self.post_ranks = self._synapses.post_rank

        # Delete the _synapses array, not needed anymore
        del self._synapses
        self._synapses = None

    ################################
    ## Dendrite access
    ################################   
    @property
    def size(self):
        if self.cyInstance:
            return self.cyInstance.size
        else:
            return 0

    def __len__(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return self.size

    ################################
    ## Access to attributes
    ################################

    def get(self, name):
        """ 
        Returns a list of parameters/variables values for each dendrite in the projection.
        
        The list will have the same length as the number of actual dendrites (self.size), so it can be smaller than the size of the postsynaptic population. Use self.post_ranks to indice it. 

        *Parameters*:

        * **name**: the name of the parameter or variable       
        """       
        return self.__getattr__(name)
            
    def set(self, value):
        """ Sets the parameters/variables values for each dendrite in the projection.
        
        For parameters, you can provide:
        
            * a single value, which will be the same for all dendrites.
            
            * a list or 1D numpy array of the same length as the number of actual dendrites (self.size).
            
        For variables, you can provide:
        
            * a single value, which will be the same for all synapses of all dendrites.
            
            * a list or 1D numpy array of the same length as the number of actual dendrites (self.size). The synapses of each postsynaptic neuron will take the same value.
        
        .. warning::

            It not possible to set different values to each synapse using this method. One should iterate over the dendrites::

                for dendrite in proj.dendrites:
                    dendrite.w = np.ones(dendrite.size)

        *Parameters*:

        * **value**: a dictionary with the name of the parameter/variable as key. 

        """
        
        for name, val in value:
            self.__setattr__(name, val)

    def __getattr__(self, name):
        " Method called when accessing an attribute."
        if not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif name == 'attributes':
            return object.__getattribute__(self, 'attributes')
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
                    if name in self.synapse.description['local']:
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
        if attribute in self.synapse.description['local']:
            return list(getattr(self.cyInstance, 'get_'+attribute)())
        else:
            return getattr(self.cyInstance, 'get_'+attribute)()

        
    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all post-synaptic neurons in the projection, 
        as a NumPy array having the same geometry as the population if it is local.
        
        Parameter:
        
        * *attribute*: should be a string representing the variables's name.
        
        """
        if isinstance(value, list):
            if len(value) == len(self.post_ranks):
                for n in self.post_ranks:
                    if not len(value[n]) == self.cyInstance.nb_synapses(n):
                        Global._error('The postynaptic neuron ' + n + ' receives '+self.cyInstance.nb_synapses(n)+ ' synapses.')
                        exit(0)
                    getattr(self.cyInstance, 'set_dendrite_'+attribute)(n, value[n])
            else:
                Global._error('The projection has ' + self.size + ' post-synaptic neurons.')
        else: # a single value
            if attribute in self.synapse.description['local']:
                for n in self.post_ranks:
                    getattr(self.cyInstance, 'set_dendrite_'+attribute)(n, value*np.ones(self.cyInstance.nb_synapses(n)))
            else:
                getattr(self.cyInstance, 'set_'+attribute)(value)

 
    ################################
    ## Variable flags
    ################################           
    def set_variable_flags(self, name, value):
        """ Sets the flags of a variable for the projection.
        
        If the variable ``r`` is defined in the Synapse description through:
        
            w = pre.r * post.r : max=1.0  
            
        one can change its maximum value with:
        
            proj.set_variable_flags('w', {'max': 2.0})
            
        For valued flags (init, min, max), ``value`` must be a dictionary containing the flag as key ('init', 'min', 'max') and its value. 
        
        For positional flags (postsynaptic, implicit), the value in the dictionary must be set to the empty string '':
        
            proj.set_variable_flags('w', {'implicit': ''})
        
        A None value in the dictionary deletes the corresponding flag:
        
            proj.set_variable_flags('w', {'max': None})


        *Parameters*:

        * **name**: the name of the variable.

        * **value**: a dictionary containing the flags.
            
        """
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The projection '+self.name+' has no variable called ' + name)
            return
            
        for key, val in value.iteritems():
            if val == '': # a flag
                try:
                    self.synapse.description['variables'][rk_var]['flags'].index(key)
                except: # the flag does not exist yet, we can add it
                    self.synapse.description['variables'][rk_var]['flags'].append(key)
            elif val == None: # delete the flag
                try:
                    self.synapse.description['variables'][rk_var]['flags'].remove(key)
                except: # the flag did not exist, check if it is a bound
                    if has_key(self.synapse.description['variables'][rk_var]['bounds'], key):
                        self.synapse.description['variables'][rk_var]['bounds'].pop(key)
            else: # new value for init, min, max...
                if key == 'init':
                    self.synapse.description['variables'][rk_var]['init'] = val 
                    self.init[name] = val              
                else:
                    self.synapse.description['variables'][rk_var]['bounds'][key] = val
                
       
            
    def set_variable_equation(self, name, equation):
        """ Changes the equation of a variable for the projection.
        
        If the variable ``w`` is defined in the Synapse description through:
        
            eta * dw/dt = pre.r * post.r 
            
        one can change the equation with:
        
            proj.set_variable_equation('w', 'eta * dw/dt = pre.r * (post.r - 0.1) ')
            
        Only the equation should be provided, the flags have to be changed with ``set_variable_flags()``.
        
        .. warning::
            
            This method should be used with great care, it is advised to define another Synapse object instead. 
            
        *Parameters*:

        * **name**: the name of the variable.

        * **equation**: the new equation as string.
        """         
        rk_var = self._find_variable_index(name)
        if rk_var == -1:
            Global._error('The projection '+self.name+' has no variable called ' + name)
            return               
        self.synapse.description['variables'][rk_var]['eq'] = equation    
            
            
    def _find_variable_index(self, name):
        " Returns the index of the variable name in self.synapse.description['variables']"
        for idx in range(len(self.synapse.description['variables'])):
            if self.synapse.description['variables'][idx]['name'] == name:
                return idx
        return -1

    ################################
    ## Learning flags
    ################################
    def enable_learning(self, params={ 'freq': 1, 'offset': 0} ):
        """
        Enables learning for all the attached dendrites.
        
        *Parameter*:
        
        * **params**: optional dictionary to configure the learning behaviour:

            * the key 'freq' determines the frequency at which the synaptic plasticity-related methods will be called. 
            * the key 'offset' determines the frequency at which the synaptic plasticity-related methods will be called. 

        For example::

            { 'freq': 10, 'offset': 5}

        would call the learning methods at time steps 5, 15, 25, etc...

        The default behaviour is that the learning methods are called at each time step.
        """
        self.cyInstance._set_learning(True)
        self.cyInstance._set_learn_frequency(params['freq'])
        self.cyInstance._set_learn_offset(params['offset'])
            
    def disable_learning(self):
        """
        Disables the learning methods for all attached dendrites.

        When this method is called, synaptic plasticity is disabled (i.e the updating of any synaptic variable except g_target and psp) until the next call to ``enable_learning``.

        This method is useful when performing some tests on a learned network without messing with the learned weights.
        """
        self.cyInstance._set_learning(False)


    
    ################################
    ## Connector methods
    ################################
    
    def connect_one_to_one(self, weights=1.0, delays=0.0, shift=None):
        """
        Builds a one-to-one connection pattern between the two populations.
        
        *Parameters*:
        
            * **weights**: initial synaptic values, either a single value (float) or a random distribution object.
            * **delays**: synaptic delays, either a single value or a random distribution object (default=dt).
            * **shift**: specifies if the ranks of the presynaptic population should be shifted to match the start of the post-synaptic population ranks. Particularly useful for PopulationViews. Does not work yet for populations with geometry. Default: if the two populations have the same number of neurons, it is set to True. If not, it is set to False (only the ranks count).
        """
        if not shift:
            if self.pre.size == self.post.size:
                shift = True
            else:
                shift = False

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.one_to_one(self.pre, self.post, weights, delays, shift)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
        return self
    
    def connect_all_to_all(self, weights, delays=0.0, allow_self_connections=False):
        """
        Builds an all-to-all connection pattern between the two populations.
        
        *Parameters*:
        
            * **weights**: synaptic values, either a single value or a random distribution object.
            * **delays**: synaptic delays, either a single value or a random distribution object (default=dt).
            * **allow_self_connections**: if True, self-connections between a neuron and itself are allowed (default=False).
        """
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.all_to_all(self.pre, self.post, weights, delays, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)

        return self

    def connect_gaussian(self, amp, sigma, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Builds a Gaussian connection pattern between the two populations.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same normalized coordinates using a Gaussian profile.
        
        *Parameters*:
        
            * **amp***: amplitude of the Gaussian function
            * **sigma**: width of the Gaussian function
            * **delays**: synaptic delay, either a single value or a random distribution object (default=dt).
            * **limit**: proportion of *amp* below which synapses are not created (default: 0.01)
            * **allow_self_connections**: allows connections between a neuron and itself.
        """
        if self.pre!=self.post:
            allow_self_connections = True

        if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
            _error('gaussian connector is only possible on whole populations, not PopulationViews.')
            exit(0)

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.gaussian(self.pre.geometry, self.post.geometry, amp, sigma, delays, limit, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
        return self
    
    def connect_dog(self, amp_pos, sigma_pos, amp_neg, sigma_neg, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Builds a Difference-Of-Gaussians connection pattern between the two populations.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same normalized coordinates using a Difference-Of-Gaussians profile.
        
        *Parameters*:
        
            * **amp_pos***: amplitude of the positive Gaussian function
            * **sigma_pos**: width of the positive Gaussian function
            * **amp_neg***: amplitude of the negative Gaussian function
            * **sigma_neg**: width of the negative Gaussian function
            * **delays**: synaptic delay, either a single value or a random distribution object (default=dt).
            * **limit**: proportion of *amp* below which synapses are not created (default: 0.01)
            * **allow_self_connections**: allows connections between a neuron and itself.
        """
        if self.pre!=self.post:
            allow_self_connections = True

        if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
            _error('DoG connector is only possible on whole populations, not PopulationViews.')
            exit(0)

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.dog(self.pre.geometry, self.post.geometry, amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)


        return self

    def connect_fixed_probability(self, probability, weights, delays=0.0, allow_self_connections=False):
        """ 
        Builds a probabilistic connection pattern between the two populations.
    
        Each neuron in the postsynaptic population is connected to neurons of the presynaptic population with the given probability. Self-connections are avoided by default.
    
        *Parameters*:
        
        * **probability**: probability that a synapse is created.
        
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        
        * **allow_self_connections** : defines if self-connections are allowed (default=False).
        """         
        if self.pre!=self.post:
            allow_self_connections = True

        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_probability(self.pre, self.post, probability, weights, delays, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
        return self

    def connect_fixed_number_pre(self, number, weights, delays=0.0, allow_self_connections=False):
        """ 
        Builds a connection pattern between the two populations with a fixed number of pre-synaptic neurons.
    
        Each neuron in the postsynaptic population receives connections from a fixed number of neurons of the presynaptic population chosen randomly. 
    
        *Parameters*:
        
        * **number**: number of synapses per postsynaptic neuron.
        
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        
        * **allow_self_connections** : defines if self-connections are allowed (default=False).
        """
        if self.pre!=self.post:
            allow_self_connections = True
        
        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_number_pre(self.pre, self.post, number, weights, delays, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)

        return self
            
    def connect_fixed_number_post(self, number, weights=1.0, delays=0.0, allow_self_connections=False):
        """
        Builds a connection pattern between the two populations with a fixed number of post-synaptic neurons.
    
        Each neuron in the pre-synaptic population sends connections to a fixed number of neurons of the post-synaptic population chosen randomly.
    
        *Parameters*:
        
        * **number**: number of synapses per pre-synaptic neuron.
        
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        
        * **allow_self_connections** : defines if self-connections are allowed (default=False).

        """
        if self.pre!=self.post:
            allow_self_connections = True
        
        import ANNarchy.core.cython_ext.Connector as Connector
        self._synapses = Connector.fixed_number_post(self.pre, self.post, number, weights, delays, allow_self_connections)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
        return self

    # def connect_from_list(self, connection_list):
    #     """
    #     Initialize projection initilized by specific list.
        
    #     Expected format:
        
    #         * list of tuples ( pre, post, weight, delay )
            
    #     Example:
        
    #         >>> conn_list = [
    #         ...     ( 0, 0, 0.0, 0.1 )
    #         ...     ( 0, 1, 0.0, 0.1 )
    #         ...     ( 0, 2, 0.0, 0.1 )
    #         ...     ( 1, 5, 0.0, 0.1 )
    #         ... ]
            
    #         proj = Projection(pre_pop, post_pop, 'exc').connect_from_list( conn_list )
    #     """
    #     self._synapses = connection_list
                
    def connect_with_func(self, method, **args):
        """
        Builds a connection pattern based on a user-defined method.

        *Parameters*:

        * **method**: method to call. The method **must** return a CSR object.
        * **args**: list of arguments needed by the function 
        """
        self._connector = method
        self._connector_params = args
        self._synapses = self._connector(self.pre, self.post, **args)
        self.max_delay = self._synapses.get_max_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)

        return self

    def _save_connectivity_as_csv(self):
        """
        Saves the projection pattern in the csv format. 

        Please note, that only the pure connectivity data pre_rank, post_rank, w and delay are stored.
        """
        filename = self.pre.name + '_' + self.post.name + '_' + self.target+'.csv'
        
        with open(filename, mode='w') as w_file:
            
            for dendrite in self.dendrites:
                rank_iter = iter(dendrite.rank)
                w_iter = iter(dendrite.w)
                delay_iter = iter(dendrite.delay)
                post_rank = dendrite.post_rank

                for i in xrange(dendrite.size):
                    w_file.write(str(next(rank_iter))+', '+
                                 str(post_rank)+', '+
                                 str(next(w_iter))+', '+
                                 str(next(delay_iter))+'\n'
                                 )

    def _comp_dist(self, pre, post):
        """
        Compute euclidean distance between two coordinates. 
        """
        res = 0.0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res
      

    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        """ 
        Gathers all receptive fields within this projection.
        
        *Parameters*:
        
            * **variable**: name of variable
            * **in_post_geometry**: if set to false, the data will be plotted as square grid. (default = True)
        """        
        if in_post_geometry:
            x_size = self.post.geometry[1]
            y_size = self.post.geometry[0]
        else:
            x_size = int( math.floor(math.sqrt(self.post.size)) )
            y_size = int( math.ceil(math.sqrt(self.post.size)) )
        


        def get_rf(rank): # TODO: IMPROVE
            res = np.zeros( self.pre.size )
            for n in xrange(len(self.post_ranks)):
                if self.post_ranks[n] == n:
                    pre_ranks = self.cyInstance.pre_rank(n)
                    data = getattr(self.cyInstance, 'get_dendrite_'+variable)(rank)
                    for j in xrange(len(pre_ranks)):
                        res[pre_ranks[j]] = data[j]  
            return res.reshape(self.pre.geometry)

        res = np.zeros((1, x_size*self.pre.geometry[1]))
        for y in xrange ( y_size ):
            row = np.concatenate(  [ get_rf(self.post.rank_from_coordinates( (y, x) ) ) for x in range ( x_size ) ], axis = 1)
            res = np.concatenate((res, row))
        
        return res

    ################################
    ## Save/load methods
    ################################

    def _data(self):
        desc = {}
        desc['post_ranks'] = self._post_ranks
        desc['attributes'] = self.attributes
        desc['parameters'] = self.parameters
        desc['variables'] = self.variables

        synapse_count = []
        dendrites = []  
        
        for d in self.post_ranks:
            dendrite_desc = {}
            # Number of synapses in the dendrite
            synapse_count.append(self.dendrite(d).size)
            # Postsynaptic rank
            dendrite_desc['post_rank'] = d
            # Attributes
            attributes = self.attributes
            if not 'w' in self.attributes:
                attributes.append('w')
            if not 'rank' in self.attributes:
                attributes.append('rank')
            if not 'delay' in self.attributes:
                attributes.append('delay')
            # Save all attributes           
            for var in attributes:
                try:
                    dendrite_desc[var] = getattr(self.cyInstance, '_get_'+var)(d) 
                except:
                    Global._error('Can not save the attribute ' + var + 'in the projection.')               
            dendrites.append(dendrite_desc)
        
        desc['dendrites'] = dendrites
        desc['number_of_synapses'] = synapse_count
        return desc
    
    def save(self, filename):
        """
        Saves all information about the projection (connectivity, current value of parameters and variables) into a file.

        * If the extension is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * If the extension ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * Otherwise, the data will be pickled into a simple binary text file using cPickle.
        
        *Parameter*:
        
        * **filename**: filename, may contain relative or absolute path.
        
            .. warning:: 

                The '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

        Example::
        
            proj.save('pop1.txt')

        """
        from ANNarchy.core.IO import _save_data
        _save_data(filename, self._data())


    def load(self, filename):
        """
        Loads the saved state of the projection.

        Warning: Matlab data can not be loaded.
        
        *Parameters*:
        
        * **filename**: the filename with relative or absolute path.
        
        Example::
        
            proj.load('pop1.txt')

        """
        from ANNarchy.core.IO import _load_data, _load_proj_data
        _load_proj_data(self, _load_data(filename))

