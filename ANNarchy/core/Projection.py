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
import copy, inspect

from ANNarchy.core import Global
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.Dendrite import Dendrite
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.parser.Report import _process_random

# Template for specific population bypassing code generation.
# The id of the population should be let free with %(id)s
proj_generator_template = {
    'omp': {
        # C++ struct to encapsulate all data
        # Example:
        # struct ProjStruct%(id_proj)s{
        #     // Number of dendrites
        #     int size;
        #     // Connectivity
        #     std::vector<int> post_rank ;
        #     std::vector< std::vector< int > > pre_rank ;
        #     std::vector< std::vector< int > > delay ;
        #    
        #     // Local parameter w
        #     std::vector< std::vector< double > > w ;
        # }; 
        'header_proj_struct' : None,

        # Initilaize the projection
        # Example:
        # 
        #    TODO:
        'body_proj_init': None,

        # Updates the random numbers
        # Example:
        #   TODO
        'body_random_dist_update': None,

        # Initializes the random numbers
        # Example:
        #   TODO
        'body_random_dist_init': None,

        # Updates the synapse variables
        # Example:
        # 
        #    TODO:
        'body_update_synapse': None,

        # compute the postsynaptic potential
        # Example:
        # 
        #    TODO:
        'body_compute_psp': None,
        
        # Export of the C++ struct to Cython (must have an indent of 4)
        # Example:
        # 
        #    TODO:
        'pyx_proj_struct': None,
        
        # Wrapper class in Cython (no indentation)
        # Example:
        # 
        #    TODO:
        'pyx_proj_class': None,
    },
    'cuda': {
        'header_proj_struct' : None,
        'body_proj_init': None,
        'body_random_dist_update': None,
        'body_random_dist_init': None,
        'body_update_synapse': None,
        'body_compute_psp': None,
        'pyx_proj_struct': None,
        'pyx_proj_class': None,
    }
}

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
            if self.pre.neuron_type.type == 'rate':
                self.synapse = Synapse(psp = "w*pre.r")
            else:
                self.synapse = Synapse(equations = "", pre_spike="g_target += w", post_spike="")
        elif inspect.isclass(synapse):
            self.synapse = synapse()
        else:
            self.synapse = copy.deepcopy(synapse)

        self.synapse._analyse()
        self.generator = copy.deepcopy(proj_generator_template)

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
        
        # Add the population to the global variable
        Global._projections.append(self)

        # Finalize initialization
        self.initialized = False

        # Cython instance
        self.cyInstance = None

        # CSR object
        self._synapses = None

        # Recorded variables
        self.recorded_variables = {}

        # Reporting
        self.connector_name = "Specific"
        self.connector_description = "Specific"

    def _instantiate(self, module):

        self._connect(module)

        self.initialized = True  

        #self._init_attributes()

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
        "Number of post-synaptic neurons receiving synapses."
        if self.cyInstance:
            return len(self.post_ranks)
        else:
            return 0

    def __len__(self):
        " Number of postsynaptic neurons receiving synapses in this projection."
        return self.size

    @property
    def nb_synapses(self):
        "Total number of synapses in the projection."
        return sum([self.cyInstance.nb_synapses(n) for n in range(self.size)])
    

    @property
    def dendrites(self):
        """
        Iteratively returns the dendrites corresponding to this projection.
        """
        for idx, n in enumerate(self.post_ranks):
            yield Dendrite(self, n, idx)
    
    def dendrite(self, pos):
        """
        Returns the dendrite of a postsynaptic neuron according to its rank.

        *Parameters*:

            * **pos**: can be either the rank or the coordinates of the postsynaptic neuron
        """
        if not self.initialized:
            Global._error('dendrites can only be accessed after compilation.')
            exit(0)
        if isinstance(pos, int):
            rank = pos
        else:
            rank = self.post.rank_from_coordinates(pos)

        if rank in self.post_ranks:
            return Dendrite(self, rank, self.post_ranks.index(rank))
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
        for idx, n in enumerate(self.post_ranks):
            yield Dendrite(self, n, idx)

    ################################
    ## Access to attributes
    ################################

    @property
    def delay(self):
        if not hasattr(self.cyInstance, 'get_delay'):
            if self.max_delay <= 1 :
                return Global.config['dt']
            elif self.uniform_delay != -1:
                return self.uniform_delay * Global.config['dt']
        else:
            return [[pre * Global.config['dt'] for pre in post] for post in self.cyInstance.get_delay()]
    

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
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif hasattr(self, 'attributes'):
            if name in self.attributes:
                if not self.initialized:
                    return self.init[name]
                else:
                    return self._get_cython_attribute( name )
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)
        
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
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
        as a list of lists having the same geometry as the population if it is local.
        
        Parameter:
        
        * *attribute*: should be a string representing the variables's name.
        
        """
        return getattr(self.cyInstance, 'get_'+attribute)()
        
    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all post-synaptic neurons in the projection, 
        as a NumPy array having the same geometry as the population if it is local.
        
        Parameter:
        
        * *attribute*: should be a string representing the variables's name.
        
        """
        if isinstance(value, np.ndarray):
            value = list(value)
        if isinstance(value, list):
            if len(value) == len(self.post_ranks):
                for idx, n in enumerate(self.post_ranks):
                    if not len(value[idx]) == self.cyInstance.nb_synapses(idx):
                        Global._error('The postynaptic neuron ' + str(n) + ' receives '+ str(self.cyInstance.nb_synapses(idx))+ ' synapses.')
                        exit(0)
                    getattr(self.cyInstance, 'set_dendrite_'+attribute)(idx, value[idx])
            else:
                Global._error('The projection has ' + self.size + ' post-synaptic neurons.')
        elif isinstance(value, RandomDistribution):
            for idx, n in enumerate(self.post_ranks):
                getattr(self.cyInstance, 'set_dendrite_'+attribute)(idx, value.get_values(self.cyInstance.nb_synapses(idx)))
        else: # a single value
            if attribute in self.synapse.description['local']:
                for idx, n in enumerate(self.post_ranks):
                    getattr(self.cyInstance, 'set_dendrite_'+attribute)(idx, value*np.ones(self.cyInstance.nb_synapses(idx)))
            else:
                getattr(self.cyInstance, 'set_'+attribute)(value*np.ones(len(self.post_ranks)))

 
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
        Enables learning for all the synapses of this projection.
        
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
        # TODO
        # self.cyInstance._set_learn_frequency(params['freq'])
        # self.cyInstance._set_learn_offset(params['offset'])
            
    def disable_learning(self):
        """
        Disables learning for all synapses of this projection.

        When this method is called, synaptic plasticity is disabled (i.e the updating of any synaptic variable except g_target and psp) until the next call to ``enable_learning``.

        This method is useful when performing some tests on a learned network without messing with the learned weights.
        """
        self.cyInstance._set_learning(False)


    
    ################################
    ## Connector methods
    ################################
    
    def _store_csr(self, csr):
        self._synapses = csr
        self.max_delay = self._synapses.get_max_delay()
        self.uniform_delay = self._synapses.get_uniform_delay()
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.population.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)

    def connect_one_to_one(self, weights=1.0, delays=0.0, shift=None):
        """
        Builds a one-to-one connection pattern between the two populations.
        
        *Parameters*:
        
            * **weights**: initial synaptic values, either a single value (float) or a random distribution object.
            * **delays**: synaptic delays, either a single value or a random distribution object (default=dt).
            * **shift**: specifies if the ranks of the presynaptic population should be shifted to match the start of the post-synaptic population ranks. Particularly useful for PopulationViews. Does not work yet for populations with geometry. Default: if the two populations have the same number of neurons, it is set to True. If not, it is set to False (only the ranks count).
        """
        if not isinstance(self.pre, PopulationView) and not isinstance(self.post, PopulationView):
            shift=False # no need
        elif not shift:
            if self.pre.size == self.post.size:
                shift = True
            else:
                shift = False

        self.connector_name = "One-to-One"
        self.connector_description = "One-to-One, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.one_to_one(self.pre, self.post, weights, delays, shift))
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

        self.connector_name = "All-to-All"
        self.connector_description = "All-to-All, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.all_to_all(self.pre, self.post, weights, delays, allow_self_connections))
        return self

    def connect_gaussian(self, amp, sigma, delays=0.0, limit=0.01, allow_self_connections=False):
        """
        Builds a Gaussian connection pattern between the two populations.

        Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around 
        the neuron with the same normalized coordinates using a Gaussian profile.
        
        *Parameters*:
        
            * **amp**: amplitude of the Gaussian function
            * **sigma**: width of the Gaussian function
            * **delays**: synaptic delay, either a single value or a random distribution object (default=dt).
            * **limit**: proportion of *amp* below which synapses are not created (default: 0.01)
            * **allow_self_connections**: allows connections between a neuron and itself.
        """
        if self.pre!=self.post:
            allow_self_connections = True

        if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
            Global._error('gaussian connector is only possible on whole populations, not PopulationViews.')
            exit(0)


        self.connector_name = "Gaussian"
        self.connector_description = "Gaussian, $A$ %(A)s, $\sigma$ %(sigma)s, delays %(delay)s"% {'A': str(amp), 'sigma': str(sigma), 'delay': _process_random(delays)}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.gaussian(self.pre.geometry, self.post.geometry, amp, sigma, delays, limit, allow_self_connections))
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
            Global._error('DoG connector is only possible on whole populations, not PopulationViews.')
            exit(0)

        self.connector_name = "Difference-of-Gaussian"
        self.connector_description = "Difference-of-Gaussian, $A^+ %(Aplus)s, $\sigma^+$ %(sigmaplus)s, $A^- %(Aminus)s, $\sigma^-$ %(sigmaminus)s, delays %(delay)s"% {'Aplus': str(amp_pos), 'sigmaplus': str(sigma_pos), 'Aminus': str(amp_neg), 'sigmaminus': str(sigma_neg), 'delay': _process_random(delays)}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.dog(self.pre.geometry, self.post.geometry, amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections))
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


        self.connector_name = "Random"
        self.connector_description = "Random, sparseness %(proba)s, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays), 'proba': probability}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.fixed_probability(self.pre, self.post, probability, weights, delays, allow_self_connections))
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

        if number > self.pre.size:
            Global._error('connect_fixed_number_pre: the number of pre-synaptic neurons exceeds the size of the population.')
            exit(0)
        
        self.connector_name = "Random Convergent"
        self.connector_description = "Random Convergent %(number)s $\\rightarrow$ 1, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

        import ANNarchy.core.cython_ext.Connector as Connector
        self._store_csr(Connector.fixed_number_pre(self.pre, self.post, number, weights, delays, allow_self_connections))

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
        
        if number > self.pre.size:
            Global._error('connect_fixed_number_post: the number of post-synaptic neurons exceeds the size of the population.')
            exit(0)

        self.connector_name = "Random Divergent"
        self.connector_description = "Random Divergent 1 $\\rightarrow$ %(number)s, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

        import ANNarchy.core.cython_ext.Connector as Connector 
        self._store_csr(Connector.fixed_number_post(self.pre, self.post, number, weights, delays, allow_self_connections))
        return self
                
    def connect_with_func(self, method, **args):
        """
        Builds a connection pattern based on a user-defined method.

        *Parameters*:

        * **method**: method to call. The method **must** return a CSR object.
        * **args**: list of arguments needed by the function 
        """
        self._connector = method
        self._connector_params = args
        synapses = self._connector(self.pre, self.post, **args)
        synapses.validate()
        self._store_csr(synapses)

        self.connector_name = "User-defined"
        self.connector_description = "User-defined"

        return self

    def connect_from_matrix(self, weights, delays=0.0):
        """
        Builds a connection pattern according to a dense connectivity matrix.

        The matrix must be N*M, where N is the number of neurons in the post-synaptic population and M in the pre-synaptic one. Lists of lists must have the same size.

        If a synapse should not be created, the weight value should be None.

        *Parameters*:

        * **weights**: a matrix or list of lists representing the weights. If a value is None, the synapse will not be created.
        * **delays**: a matrix or list of lists representing the delays. Must represent the same synapses as weights. If the argument is omitted, delays are 0. 
        """
        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            Global._error('ANNarchy was not successfully installed.')
        csr = CSR()

        if isinstance(weights, list):
            try:
                weights= np.array(weights)
            except:
                Global._error('connect_from_matrix: You must provide a dense 2D matrix.')
                exit(0)

        uniform_delay = not isinstance(delays, (list, np.ndarray))
        if isinstance(delays, list):
            try:
                delays= np.array(delays)
            except:
                Global._error('connect_from_matrix: You must provide a dense 2D matrix.')
                exit(0)

        shape = weights.shape
        if shape != (self.post.size, self.pre.size):
            Global._error('connect_from_matrix: wrong size for for the matrix, should be', str((self.post.size, self.pre.size)))
            exit(0)

        for i in range(self.post.size):
            rk = []
            w = []
            d = []
            for idx, val in enumerate(list(weights[i, :])):
                if val != None:
                    rk.append(idx)
                    w.append(val) 
                    if not uniform_delay:
                        d.append(delays[i,idx])   
            if uniform_delay:        
                d.append(delays)
            if len(rk) > 0:
                csr.add(i, rk, w, d)

        # Store the synapses
        self.connector_name = "Connectivity matrix"
        self.connector_description = "Connectivity matrix"
        self._store_csr(csr)
        return self

    def connect_from_sparse(self, weights, delays=0.0):
        """
        Builds a connectivity pattern using a Scipy sparse matrix for the weights and (optionally) delays.

        *Parameters*:

        * **weights**: a sparse lil_matrix object created from scipy.
        * **delays**: the value of the homogenous delay (default: dt).
        """
        try:
            from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
        except:
            Global._error("scipy is not installed, sparse matrices can not be loaded.")
            exit(0)

        if not isinstance(weights, (lil_matrix, csr_matrix, csc_matrix)):
            Global._error("only lil, csr and csc matrices are allowed for now.")
            exit(0)

        # Create an empty CSR object
        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            Global._error('ANNarchy was not successfully installed.')
            exit(0)
        csr = CSR()

        # Find offsets
        if hasattr(self.pre, 'ranks'):
            offset_pre = self.pre.ranks[0]
        else:
            offset_pre = 0
        if hasattr(self.post, 'ranks'):
            offset_post = self.post.ranks[0]
        else:
            offset_post = 0
        
        # Process the sparse matrix and fill the csr
        W = csc_matrix(weights)
        W.sort_indices()
        (pre, post) = W.shape
        for idx_post in xrange(post):
            pre_rank = W.getcol(idx_post).indices
            w = W.getcol(idx_post).data
            csr.add(idx_post + offset_post, pre_rank + offset_pre, w, [float(delays)])
        
        # Store the synapses
        self.connector_name = "Sparse connectivity matrix"
        self.connector_description = "Sparse connectivity matrix"
        self._store_csr(csr)
        return self

    def connect_from_file(self, filename):
        """
        Builds a connection pattern using data saved using the Projection.save_connectivity() method.

        *Parameters*:

        * **filename**: file where the data was saved.

        .. note::

            Only the ranks, weights and delays are loaded, not the other variables.
        """
        # Create an empty CSR object
        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            Global._error('ANNarchy was not successfully installed.')
        csr = CSR()

        # Load the data        
        from ANNarchy.core.IO import _load_data
        try:
            data = _load_data(filename)
        except:
            Global._error('Unable to load the data', filename, 'into the projection.')
            exit(0)

        # Load the CSR object
        try:
            csr.post_rank = data['post_ranks']
            csr.pre_rank = data['pre_ranks']
            csr.w = data['w']
            csr.size = data['size']
            csr.nb_synapses = data['nb_synapses']
            if data['delay']:
                csr.delay = data['delay']
            csr.max_delay = data['max_delay']
            csr.uniform_delay = data['uniform_delay']
        except:
            Global._error('Unable to load the data', filename, 'into the projection.')
            exit(0)

        # Store the synapses
        self.connector_name = "From File"
        self.connector_description = "From File"
        self._store_csr(csr)
        return self

    def save_connectivity(self, filename):
        """
        Saves the projection pattern in a file.

        Only the connectivity matrix, the weights and delays are saved, not the other synaptic variables. 

        The generated data should be used to create a projection in another network::

            proj.connect_from_file(filename) 

        *Parameters*:

        * **filename**: file where the data will be saved.
        """
        if not self.initialized:
            data = {
                'post_ranks': self._synapses.post_rank,
                'pre_ranks': self._synapses.pre_rank,
                'w': self._synapses.w,
                'delay': self._synapses.delay,
                'max_delay': self._synapses.max_delay,
                'uniform_delay': self._synapses.uniform_delay,
                'size': self._synapses.size,
                'nb_synapses': self._synapses.nb_synapses,
            }
        else:
            data = {
                'post_ranks': self.post_ranks,
                'pre_ranks': [self.cyInstance.pre_rank(n) for n in range(self.size)],
                'w': self.cyInstance.get_w(),
                'delay': self.cyInstance.get_delay() if hasattr(self.cyInstance, 'get_delay') else None,
                'max_delay': self.max_delay,
                'uniform_delay': self.uniform_delay,
                'size': self.size,
                'nb_synapses': sum([self.cyInstance.nb_synapses(n) for n in range(self.size)])
            }

        import cPickle
        with open(filename, 'wb') as wfile:
            cPickle.dump(data, wfile, protocol=cPickle.HIGHEST_PROTOCOL)

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

    def connectivity_matrix(self, fill=0.0):
        """
        Returns a dense connectivity matrix (2D Numpy array) representing the connections between the pre- and post-populations.

        The first index of the matrix represents post-synaptic neurons, the second the pre-synaptic ones.

        If PopulationViews were used for creating the projection, the matrix is expanded to the whole populations by default. 

        *Parameters*:
        
        * **fill**: value to put in the matrix when there is no connection (default: 0.0).
        """
        if isinstance(self.pre, PopulationView):
            size_pre = self.pre.population.size
        else:
            size_pre = self.pre.size
        if isinstance(self.post, PopulationView):
            size_post = self.post.population.size
        else:
            size_post = self.post.size

        res = np.ones((size_post, size_pre)) * fill
        for rank in self.post_ranks:
            idx = self.post_ranks.index(rank)
            try:
                preranks = self.cyInstance.pre_rank(idx)
                w = self.cyInstance.get_dendrite_w(idx)
            except:
                Global._error('The connectivity matrix can only be accessed after compilation')
                return []
            res[rank, preranks] = w
        return res


    ################################
    ## Save/load methods
    ################################

    def _data(self):
        desc = {}
        desc['post_ranks'] = self.post_ranks
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
            # Number of synapses
            dendrite_desc['size'] = self.cyInstance.nb_synapses(d)
            # Attributes
            attributes = self.attributes
            if not 'w' in self.attributes:
                attributes.append('w')
            # Save all attributes           
            for var in attributes:
                try:
                    dendrite_desc[var] = getattr(self.cyInstance, 'get_dendrite_'+var)(d) 
                except Exception as e:
                    Global._error('Can not save the attribute ' + var + ' in the projection.')    
            # Add pre-synaptic ranks and delays
            dendrite_desc['rank'] = self.cyInstance.pre_rank(d)
            if hasattr(self.cyInstance, 'get_delay'):
                dendrite_desc['delay'] = self.cyInstance.get_delay()
            # Finish
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


    ################################
    ## Structural plasticity
    ################################
    def start_pruning(self, period=None):
        """
        Starts pruning the synapses in the projection if the synapse defines a 'pruning' argument.

        'structural_plasticity' must be set to True in setup().

        *Parameters*:

        * **period**: how often pruning should be evaluated (default: dt, i.e. each step)
        """
        if not period:
            period = Global.config['dt']
        if not Global._compiled:
            Global._error('Can not start pruning if the network is not compiled.')
            exit(0)
        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.start_pruning(int(period/Global.config['dt']), Global.get_current_step())
            except :
                Global._error("The synapse does not define a 'pruning' argument.")
                exit(0)
        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start pruning connections.")
            exit(0)

    def stop_pruning(self):
        """
        Stops pruning the synapses in the projection if the synapse defines a 'pruning' argument.

        'structural_plasticity' must be set to True in setup().
        """
        if not Global._compiled:
            Global._error('Can not stop pruning if the network is not compiled.')
            exit(0)
        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.stop_pruning()
            except:
                Global._error("The synapse does not define a 'pruning' argument.")
                exit(0)
        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start pruning connections.")
            exit(0)

    def start_creating(self, period=None):
        """
        Starts creating the synapses in the projection if the synapse defines a 'creating' argument.

        'structural_plasticity' must be set to True in setup().

        *Parameters*:

        * **period**: how often creating should be evaluated (default: dt, i.e. each step)
        """
        if not period:
            period = Global.config['dt']
        if not Global._compiled:
            Global._error('Can not start creating if the network is not compiled.')
            exit(0)
        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.start_creating(int(period/Global.config['dt']), Global.get_current_step())
            except:
                Global._error("The synapse does not define a 'creating' argument.")
                exit(0)
        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start creating connections.")
            exit(0)

    def stop_creating(self):
        """
        Stops creating the synapses in the projection if the synapse defines a 'creating' argument.

        'structural_plasticity' must be set to True in setup().
        """
        if not Global._compiled:
            Global._error('Can not stop creating if the network is not compiled.')
            exit(0)
        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.stop_creating()
            except:
                Global._error("The synapse does not define a 'creating' argument.")
                exit(0)
        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start creating connections.")
            exit(0)
