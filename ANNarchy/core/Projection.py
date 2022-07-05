#===============================================================================
#
#     Projection.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
import numpy as np
import math, os
import copy, inspect
import pickle

from ANNarchy.core import Global
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Dendrite import Dendrite
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core import ConnectorMethods

class Projection(object):
    """
    Container for all the synapses of the same type between two populations.
    """

    def __init__(self, pre, post, target, synapse=None, name=None, disable_omp=True, copied=False):
        """
        By default, the synapse only ensures linear synaptic transmission:

        * For rate-coded populations: ``psp = w * pre.r``
        * For spiking populations: ``g_target += w``

        to modify this behavior one need to provide a Synapse object.

        :param pre: pre-synaptic population (either its name or a ``Population`` object).
        :param post: post-synaptic population (either its name or a ``Population`` object).
        :param target: type of the connection.
        :param synapse: a ``Synapse`` instance.
        :param name: unique name of the projection (optional, it defaults to ``proj0``, ``proj1``, etc).
        :param disable_omp: especially for small- and mid-scale sparse spiking networks the parallelization of spike propagation is not scalable. But it can be enabled by setting this parameter to `False`.
        """
        # Check if the network has already been compiled
        if Global._network[0]['compiled'] and not copied:
            Global._error('you cannot add a projection after the network has been compiled.')

        # Store the pre and post synaptic populations
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object
        if isinstance(pre, str):
            for pop in Global._network[0]['populations']:
                if pop.name == pre:
                    self.pre = pop
        else:
            self.pre = pre

        if isinstance(post, str):
            for pop in Global._network[0]['populations']:
                if pop.name == post:
                    self.post = pop
        else:
            self.post = post

        # Store the arguments
        if isinstance(target, list) and len(target) == 1:
            self.target = target[0]
        else:
            self.target = target

        # Add the target(s) to the postsynaptic population
        if isinstance(self.target, list):
            for _target in self.target:
                self.post.targets.append(_target)
        else:
            self.post.targets.append(self.target)

        # check if a synapse description is attached
        if not synapse:
            # No synapse attached assume default synapse based on
            # presynaptic population.
            if self.pre.neuron_type.type == 'rate':
                from ANNarchy.models.Synapses import DefaultRateCodedSynapse
                self.synapse_type = DefaultRateCodedSynapse()
                self.synapse_type.type = 'rate'
            else:
                from ANNarchy.models.Synapses import DefaultSpikingSynapse
                self.synapse_type = DefaultSpikingSynapse()
                self.synapse_type.type = 'spike'

        elif inspect.isclass(synapse):
            self.synapse_type = synapse()
            self.synapse_type.type = self.pre.neuron_type.type
        else:
            self.synapse_type = copy.deepcopy(synapse)
            self.synapse_type.type = self.pre.neuron_type.type

        # Disable omp for spiking networks
        self.disable_omp = disable_omp

        # Analyse the parameters and variables
        self.synapse_type._analyse()

        # Create a default name
        self.id = len(Global._network[0]['projections'])
        if name:
            self.name = name
        else:
            self.name = 'proj'+str(self.id)

        # Get a list of parameters and variables
        self.parameters = []
        self.init = {}
        for param in self.synapse_type.description['parameters']:
            self.parameters.append(param['name'])
            self.init[param['name']] = param['init']

        self.variables = []
        for var in self.synapse_type.description['variables']:
            self.variables.append(var['name'])
            self.init[var['name']] = var['init']

        self.attributes = self.parameters + self.variables

        # Get a list of user-defined functions
        self.functions = [func['name'] for func in self.synapse_type.description['functions']]

        # Add the population to the global network
        Global._network[0]['projections'].append(self)

        # Finalize initialization
        self.initialized = False

        # Cython instance
        self.cyInstance = None

        # Connectivity
        self._synapses = None
        self._connection_method = None
        self._connection_args = None
        self._connection_delay = None
        self._connector = None
        self._lil_connectivity = None

        # Default configuration for connectivity
        self._storage_format = "lil"
        self._storage_order = "post_to_pre"

        # If a single weight value is used
        self._single_constant_weight = False

        # Are random distribution used for weights/delays
        self.connector_weight_dist = None
        self.connector_delay_dist = None

        # Reporting
        self.connector_name = "Specific"
        self.connector_description = "Specific"

        # Overwritten by derived classes, to add
        # additional code
        self._specific_template = {}

        # Set to False by derived classes to prevent saving of
        # data, e. g. in case of weight-sharing projections
        self._saveable = True

        # To allow case-specific adjustment of parallelization
        # parameters, e. g. openMP schedule, we introduce a
        # dictionary read by the ProjectionGenerator.
        #
        # Will be overwritten either by inherited classes or
        # by an omp_config provided to the compile() method.
        self._omp_config = {
            #'psp_schedule': 'schedule(dynamic)'
        }

        # If set to true, the code generator is not allowed to
        # split the matrix. This will be the case for many
        # SpecificProjections defined by the user or is disabled
        # globally.
        if self.synapse_type.type == "rate":
            # Normally, the split should not be used for rate-coded models
            # but maybe there are cases where we want to enable it ...
            self._no_split_matrix = Global.config["disable_split_matrix"]

            # If the number of elements is too small, the split
            # might not be efficient.
            if self.post.size < Global.OMP_MIN_NB_NEURONS:
                self._no_split_matrix = True

        else:
            # If the number of elements is too small, the split
            # might not be efficient.
            if self.post.size < Global.OMP_MIN_NB_NEURONS:
                self._no_split_matrix = True
            else:
                self._no_split_matrix = False

    # Add defined connectors
    connect_one_to_one = ConnectorMethods.connect_one_to_one
    connect_all_to_all = ConnectorMethods.connect_all_to_all
    connect_gaussian = ConnectorMethods.connect_gaussian
    connect_dog = ConnectorMethods.connect_dog
    connect_fixed_probability = ConnectorMethods.connect_fixed_probability
    connect_fixed_number_pre = ConnectorMethods.connect_fixed_number_pre
    connect_fixed_number_post = ConnectorMethods.connect_fixed_number_post
    connect_with_func = ConnectorMethods.connect_with_func
    connect_from_matrix = ConnectorMethods.connect_from_matrix
    connect_from_matrix_market = ConnectorMethods.connect_from_matrix_market
    _load_from_matrix = ConnectorMethods._load_from_matrix
    connect_from_sparse = ConnectorMethods.connect_from_sparse
    _load_from_sparse = ConnectorMethods._load_from_sparse
    connect_from_file = ConnectorMethods.connect_from_file
    _load_from_lil = ConnectorMethods._load_from_lil

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        copied_proj = Projection(pre=pre, post=post, target=self.target, synapse=self.synapse_type, name=self.name, disable_omp=self.disable_omp, copied=True)

        # these flags are modified during connect_XXX called before Network()
        copied_proj._single_constant_weight = self._single_constant_weight
        copied_proj.connector_weight_dist = self.connector_weight_dist
        copied_proj.connector_delay_dist = self.connector_delay_dist
        copied_proj.connector_name = self.connector_name

        # Control flags for code generation (maybe modified by connect_XXX())
        copied_proj._storage_format = self._storage_format
        copied_proj._storage_order = self._storage_order
        copied_proj._no_split_matrix = self._no_split_matrix

        # for some projection types saving is not allowed (e. g. Convolution, Pooling)
        copied_proj._saveable = self._saveable

        # optional flags
        if hasattr(self, "_bsr_size"):
            copied_proj._bsr_size = self._bsr_size

        return copied_proj

    def _generate(self):
        "Overriden by specific projections to generate the code"
        pass

    def _instantiate(self, module):
        """
        Instantiates the projection after compilation. The function should be
        called by Compiler._instantiate().

        :param:     module  cython module (ANNarchyCore instance)
        """
        self.initialized = self._connect(module)

    def _init_attributes(self):
        """
        Method used after compilation to initialize the attributes. The function
        should be called by Compiler._instantiate
        """
        for name, val in self.init.items():
            # the weights ('w') are already inited by the _connect() method.
            if not name in ['w']:
                self.__setattr__(name, val)

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().

        :param:     module  cython module (ANNarchyCore instance)
        :return:    True, if the connector was successfully instantiated. Potential errors are kept by 
                    Python exceptions. If the Cython - connector call fails (return False) the most likely
                    reason is that there was not enough memory available.
        """
        # Local import to prevent circular import (HD: 28th June 2021)
        from ANNarchy.generator.Utils import cpp_connector_available

        # Sanity check
        if not self._connection_method:
            Global._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Debug printout
        if Global.config["verbose"]:
            print("Connectivity parameter ("+self.name+"):", self._connection_args )

        # Instantiate the Cython wrapper
        if not self.cyInstance:
            cy_wrapper = getattr(module, 'proj'+str(self.id)+'_wrapper')
            self.cyInstance = cy_wrapper()

        # Check if there is a specialized CPP connector
        if not cpp_connector_available(self.connector_name, self._storage_format, self._storage_order):
            # No default connector -> initialize from LIL
            if self._lil_connectivity:
                return self.cyInstance.init_from_lil_connectivity(self._lil_connectivity)
            else:
                return self.cyInstance.init_from_lil_connectivity(self._connection_method(*((self.pre, self.post,) + self._connection_args)))

        else:
            # fixed probability pattern
            if self.connector_name == "Random":
                p = self._connection_args[0]
                allow_self_connections = self._connection_args[3]
                if isinstance(self._connection_args[1], RandomDistribution):
                    #some kind of distribution
                    w_dist_arg1, w_dist_arg2 = self._connection_args[1].get_cpp_args()
                else:
                    # constant
                    w_dist_arg1 = self._connection_args[1]
                    w_dist_arg2 = self._connection_args[1]

                if isinstance(self._connection_args[2], RandomDistribution):
                    #some kind of distribution
                    d_dist_arg1, d_dist_arg2 = self._connection_args[2].get_cpp_args()
                else:
                    # constant
                    d_dist_arg1 = self._connection_args[2]
                    d_dist_arg2 = self._connection_args[2]

                return self.cyInstance.fixed_probability(self.post.ranks, self.pre.ranks, p, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2, allow_self_connections)
        
            # fixed number pre prattern
            elif self.connector_name== "Random Convergent":
                number_nonzero = self._connection_args[0]
                if isinstance(self._connection_args[1], RandomDistribution):
                    #some kind of distribution
                    w_dist_arg1, w_dist_arg2 = self._connection_args[1].get_cpp_args()
                else:
                    # constant
                    w_dist_arg1 = self._connection_args[1]
                    w_dist_arg2 = self._connection_args[1]

                if isinstance(self._connection_args[2], RandomDistribution):
                    #some kind of distribution
                    d_dist_arg1, d_dist_arg2 = self._connection_args[2].get_cpp_args()
                else:
                    # constant
                    d_dist_arg1 = self._connection_args[2]
                    d_dist_arg2 = self._connection_args[2]

                return self.cyInstance.fixed_number_pre(self.post.ranks, self.pre.ranks, number_nonzero, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2)

            else:
                # This should never happen ...
                Global._error("No initialization for CPP-connector defined ...")

        # should be never reached ...
        return False

    def _store_connectivity(self, method, args, delay, storage_format="lil", storage_order="post_to_pre"):
        """
        Store connectivity data. This function is called from cython_ext.Connectors module.
        """
        if self._connection_method != None:
            Global._warning("Projection ", self.name, " was already connected ... data will be overwritten.")

        if storage_format == "auto" and self.synapse_type.type == "spike":
            Global._error("Automatic format selection is not supported for spiking models yet.")

        # Store connectivity pattern parameters
        self._connection_method = method
        self._connection_args = args
        self._connection_delay = delay
        self._storage_format = storage_format
        self._storage_order = storage_order

        # Local import to prevent circular import (HD: 15th March 2022)
        from ANNarchy.generator.Utils import cpp_connector_available

        # Automatic format selection using heuristics for rate-coded models
        if storage_format == "auto" and self.synapse_type.type == "rate":
            self._storage_format = self._automatic_format_selection(args)

        # Analyse the delay
        if isinstance(delay, (int, float)): # Uniform delay
            self.max_delay = round(delay/Global.config['dt'])
            self.uniform_delay = round(delay/Global.config['dt'])

        elif isinstance(delay, RandomDistribution): # Non-uniform delay
            self.uniform_delay = -1
            # Ensure no negative delays are generated
            if delay.min is None or delay.min < Global.config['dt']:
                delay.min = Global.config['dt']
            # The user needs to provide a max in order to compute max_delay
            if delay.max is None:
                Global._error('Projection.connect_xxx(): if you use a non-bounded random distribution for the delays (e.g. Normal), you need to set the max argument to limit the maximal delay.')

            self.max_delay = round(delay.max/Global.config['dt'])

        elif isinstance(delay, (list, np.ndarray)): # connect_from_matrix/sparse
            if len(delay) > 0:
                self.uniform_delay = -1
                self.max_delay = round(max([max(l) for l in delay])/Global.config['dt'])
            else: # list is empty, no delay
                self.max_delay = -1
                self.uniform_delay = -1

        else:
            Global._error('Projection.connect_xxx(): delays are not valid!')

        # Transmit the max delay to the pre pop
        if isinstance(self.pre, PopulationView):
            self.pre.population.max_delay = max(self.max_delay, self.pre.population.max_delay)
        else:
            self.pre.max_delay = max(self.max_delay, self.pre.max_delay)

    def _automatic_format_selection(self, args):
        """
        We check some heuristics to select a specific format (currently only on GPUs) implemented
        as decision tree:

            - If the filling degree is high enough a full matrix representation might be better
            - if the average row length is below a threshold the ELLPACK-R might be better
            - if the average row length is higher than a threshold the CSR might be better

        HD (17th Jan. 2022): Currently structural plasticity is only usable with LIL. But one could also
                             apply it for dense matrices in the future. For CSR and in particular the ELL-
                             like formats the potential memory-reallocations make the structural plasticity
                             a costly operation.
        """
        if Global.config["structural_plasticity"]:
            storage_format = "lil"

        elif self.connector_name == "All-to-All":
            storage_format = "dense"

        elif self.connector_name == "One-to-One":
            if Global._check_paradigm("cuda"):
                storage_format = "csr"
            else:
                storage_format = "lil"

        else:
            # we need to build up the matrix to analyze
            self._lil_connectivity = self._connection_method(*((self.pre, self.post,) + self._connection_args))

            # get the decision parameter
            density = float(self._lil_connectivity.nb_synapses) / float(self.pre.size * self.post.size)
            avg_nnz_per_row, _ = self._lil_connectivity.compute_average_row_length()

            # heuristic decision tree
            if density >= 0.6:
                storage_format = "dense"
            else:
                if Global._check_paradigm("cuda"):
                    if avg_nnz_per_row <= 128:
                        storage_format = "ellr"
                    else:
                        storage_format = "csr"
                else:
                    storage_format = "csr"

        Global._info("Automatic format selection for", self.name, ":", storage_format)
        return storage_format


    def _has_single_weight(self):
        "If a single weight should be generated instead of a LIL"
        is_cpu = Global.config['paradigm']=="openmp"
        has_constant_weight = self._single_constant_weight
        not_dense = not (self._storage_format == "dense")
        no_structural_plasticity = not Global.config['structural_plasticity']
        no_synaptic_plasticity = not self.synapse_type.description['plasticity']

        return has_constant_weight and no_structural_plasticity and no_synaptic_plasticity and is_cpu and not_dense

    def reset(self, attributes=-1, synapses=False):
        """
        Resets all parameters and variables of the projection to the value they had before the call to compile.

        **Note:** Only parameters and variables are reinitialized, not the connectivity structure (including the weights and delays).
        The parameter ``synapses`` will be used in a future release to also reinitialize the connectivity structure.

        :param attributes: list of attributes (parameter or variable) which should be reinitialized. Default: all attributes.
        """
        if attributes == -1:
            attributes = self.attributes

        if synapses:
            # destroy the previous C++ content
            self._clear()
            # call the init connectivity again
            self._connect(None)
            self.initialized = True

        for var in attributes:
            # Skip w
            if var=='w':
                continue
            # check it exists
            if not var in self.attributes:
                Global._warning("Projection.reset():", var, "is not an attribute of the population, won't reset.")
                continue
            # Set the value
            try:
                self.__setattr__(var, self.init[var])
            except Exception as e:
                Global._print(e)
                Global._warning("Projection.reset(): something went wrong while resetting", var)
        #Global._warning('Projection.reset(): only parameters and variables are reinitialized, not the connectivity structure (including the weights)...')

    ################################
    ## Dendrite access
    ################################
    @property
    def size(self):
        "Number of post-synaptic neurons receiving synapses."
        if self.cyInstance == None:
            Global._warning("Access 'size or len()' attribute of a Projection is only valid after compile()")
            return 0

        return len(self.cyInstance.post_rank())

    def __len__(self):
        # Number of postsynaptic neurons receiving synapses in this projection.
        return self.size

    @property
    def nb_synapses(self):
        "Total number of synapses in the projection."
        if self.cyInstance is None:
            Global._warning("Access 'nb_synapses' attribute of a Projection is only valid after compile()")
            return 0
        return self.cyInstance.nb_synapses()

    def nb_synapses_per_dendrite(self):
        "Total number of synapses for each dendrite as a list."
        if self.cyInstance is None:
            Global._warning("Access 'nb_synapses_per_dendrite' attribute of a Projection is only valid after compile()")
            return []
        return [self.cyInstance.dendrite_size(n) for n in range(self.size)]

    def nb_efferent_synapses(self):
        "Number of efferent connections. Intended only for spiking models."
        if self.synapse_type.type == "rate":
            Global._error("Projection.nb_efferent_synapses() is not available for rate-coded projections.")

        return self.cyInstance.nb_efferent_synapses()

    @property
    def post_ranks(self):
        if self.cyInstance:
            return self.cyInstance.post_rank()
        else:
             Global._warning("Access 'post_ranks' attribute of a Projection is only valid after compile()")
             return None

    @property
    def dendrites(self):
        """
        Iteratively returns the dendrites corresponding to this projection.
        """
        for idx, n in enumerate(self.post_ranks):
            yield Dendrite(self, n, idx)

    def dendrite(self, post):
        """
        Returns the dendrite of a postsynaptic neuron according to its rank.

        :param post: can be either the rank or the coordinates of the post-synaptic neuron.
        """
        if not self.initialized:
            Global._error('dendrites can only be accessed after compilation.')

        if isinstance(post, int):
            rank = post
        else:
            rank = self.post.rank_from_coordinates(post)

        if rank in self.post_ranks:
            return Dendrite(self, rank, self.post_ranks.index(rank))
        else:
            Global._error(" The neuron of rank "+ str(rank) + " has no dendrite in this projection.", exit=True)


    def synapse(self, pre, post):
        """
        Returns the synapse between a pre- and a post-synaptic neuron if it exists, None otherwise.

        :param pre: rank of the pre-synaptic neuron.
        :param post: rank of the post-synaptic neuron.
        """
        if not isinstance(pre, int) or not isinstance(post, int):
            Global._error('Projection.synapse() only accepts ranks for the pre and post neurons.')

        return self.dendrite(post).synapse(pre)


    # Iterators
    def __getitem__(self, *args, **kwds):
        # Returns dendrite of the given position in the postsynaptic population.
        # If only one argument is given, it is a rank. If it is a tuple, it is coordinates.

        if len(args) == 1:
            return self.dendrite(args[0])
        return self.dendrite(args)

    def __iter__(self):
        # Returns iteratively each dendrite in the population in ascending postsynaptic rank order.
        for idx, n in enumerate(self.post_ranks):
            yield Dendrite(self, n, idx)

    ################################
    ## Access to attributes
    ################################
    def get(self, name):
        """
        Returns a list of parameters/variables values for each dendrite in the projection.

        The list will have the same length as the number of actual dendrites (self.size), so it can be smaller than the size of the postsynaptic population. Use self.post_ranks to indice it.

        :param name: the name of the parameter or variable
        """
        return self.__getattr__(name)

    def set(self, value):
        """
        Sets the parameters/variables values for each dendrite in the projection.

        For parameters, you can provide:

        * a single value, which will be the same for all dendrites.

        * a list or 1D numpy array of the same length as the number of actual dendrites (self.size).

        For variables, you can provide:

        * a single value, which will be the same for all synapses of all dendrites.

        * a list or 1D numpy array of the same length as the number of actual dendrites (self.size). The synapses of each postsynaptic neuron will take the same value.

        **Warning:** it is not possible to set different values to each synapse using this method. One should iterate over the dendrites:

        ```python
        for dendrite in proj.dendrites:
            dendrite.w = np.ones(dendrite.size)
        ```

        :param value: a dictionary with the name of the parameter/variable as key.

        """

        for name, val in value.items():
            self.__setattr__(name, val)

    def __getattr__(self, name):
        # Method called when accessing an attribute.
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            return object.__getattribute__(self, name)
        elif hasattr(self, 'attributes'):
            if name in ['plasticity', 'transmission', 'update']:
                return self._get_flag(name)
            if name in ['delay']:
                return self._get_delay()
            if name in self.attributes:
                if not self.initialized:
                    return self.init[name]
                else:
                    return self._get_cython_attribute( name )
            elif name in self.functions:
                return self._function(name)
            else:
                return object.__getattribute__(self, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        # Method called when setting an attribute.
        if name == 'initialized' or not hasattr(self, 'initialized'): # Before the end of the constructor
            object.__setattr__(self, name, value)
        elif hasattr(self, 'attributes'):
            if name in ['plasticity', 'transmission', 'update']:
                self._set_flag(name, bool(value))
                return
            if name in ['delay']:
                self._set_delay(value)
                return
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

        :param attribute: a string representing the variables's name.

        """
        # Determine C++ data type
        ctype = None
        for var in self.synapse_type.description['variables']+self.synapse_type.description['parameters']:
            if var['name'] == attribute:
                ctype = var['ctype']

        if attribute == "w" and self._has_single_weight():
            return self.cyInstance.get_global_attribute(attribute, ctype)
        elif attribute in self.synapse_type.description['local']:
            return self.cyInstance.get_local_attribute_all(attribute, ctype)
        elif attribute in self.synapse_type.description['semiglobal']:
            return self.cyInstance.get_semiglobal_attribute_all(attribute, ctype)
        else:
            return self.cyInstance.get_global_attribute(attribute, ctype)

    def _set_cython_attribute(self, attribute, value):
        """
        Sets the value of the given attribute for all post-synaptic neurons in the projection,
        as a NumPy array having the same geometry as the population if it is local.

        :param attribute: a string representing the variables's name.
        :param value: the value it should take.

        """
        # Determine C++ data type
        ctype = None
        for var in self.synapse_type.description['variables']+self.synapse_type.description['parameters']:
            if var['name'] == attribute:
                ctype = var['ctype']

        # Convert np.arrays into lists/constants for better iteration
        if isinstance(value, np.ndarray):
            if np.ndim(value) == 0:
                value = float(value)
            else:
                value = list(value)

        # A list is given
        if isinstance(value, list):
            if len(value) == len(self.post_ranks):
                if attribute in self.synapse_type.description['local']:
                    for idx, n in enumerate(self.post_ranks):
                        if not len(value[idx]) == self.cyInstance.dendrite_size(idx):
                            Global._error('The postynaptic neuron ' + str(n) + ' receives '+ str(self.cyInstance.dendrite_size(idx))+ ' synapses.')
                        self.cyInstance.set_local_attribute_row(attribute, idx, value[idx], ctype)
                elif attribute in self.synapse_type.description['semiglobal']:
                    self.cyInstance.set_semiglobal_attribute_all(attribute, value, ctype)
                else:
                    Global._error('The parameter', attribute, 'is global to the population, cannot assign a list.')
            else:
                Global._error('The projection has', self.size, 'post-synaptic neurons, the list must have the same size.')
        # A Random Distribution is given
        elif isinstance(value, RandomDistribution):
            if attribute == "w" and self._has_single_weight():
                self.cyInstance.set_global_attribute(attribute, value.get_values(1), ctype)
            elif attribute in self.synapse_type.description['local']:
                for idx, n in enumerate(self.post_ranks):
                    self.cyInstance.set_local_attribute_row(attribute, idx, value.get_values(self.cyInstance.dendrite_size(idx)), ctype)
            elif attribute in self.synapse_type.description['semiglobal']:
                self.cyInstance.set_semiglobal_attribute_all(attribute, value.get_values(len(self.post_ranks)), ctype)
            elif attribute in self.synapse_type.description['global']:
                self.cyInstance.set_global_attribute(attribute, value.get_values(1), ctype)
        # A single value is given
        else:
            if attribute == "w" and self._has_single_weight():
                self.cyInstance.set_global_attribute(attribute, value, ctype)
            elif attribute in self.synapse_type.description['local']:
                for idx, n in enumerate(self.post_ranks):
                    self.cyInstance.set_local_attribute_row(attribute, idx, value*np.ones(self.cyInstance.dendrite_size(idx)), ctype)
            elif attribute in self.synapse_type.description['semiglobal']:
                self.cyInstance.set_semiglobal_attribute_all(attribute, value*np.ones(len(self.post_ranks)), ctype)
            else:
                self.cyInstance.set_global_attribute(attribute, value, ctype)

    def _get_attribute_cpp_type(self, attribute):
        """
        Determine C++ data type for a given attribute
        """
        ctype = None
        for var in self.synapse_type.description['variables']+self.synapse_type.description['parameters']:
            if var['name'] == attribute:
                ctype = var['ctype']

        return ctype

    def _get_flag(self, attribute):
        "flags such as learning, transmission"
        return getattr(self.cyInstance, '_get_'+attribute)()

    def _set_flag(self, attribute, value):
        "flags such as learning, transmission"
        getattr(self.cyInstance, '_set_'+attribute)(value)



    ################################
    ## Access to delays
    ################################
    def _get_delay(self):
        if not hasattr(self.cyInstance, 'get_delay'):
            if self.max_delay <= 1 :
                return Global.config['dt']
        elif self.uniform_delay != -1:
                return self.uniform_delay * Global.config['dt']
        else:
            return [[pre * Global.config['dt'] for pre in post] for post in self.cyInstance.get_delay()]

    def _set_delay(self, value):

        if self.cyInstance: # After compile()
            if not hasattr(self.cyInstance, 'get_delay'):
                if self.max_delay <= 1 and value != Global.config['dt']:
                    Global._error("set_delay: the projection was instantiated without delays, it is too late to create them...")

            elif self.uniform_delay != -1:
                if isinstance(value, np.ndarray):
                    if value.ndim > 0:
                        Global._error("set_delay: the projection was instantiated with uniform delays, it is too late to load non-uniform values...")
                    else:
                        value = max(1, round(float(value)/Global.config['dt']))
                elif isinstance(value, (float, int)):
                    value = max(1, round(float(value)/Global.config['dt']))
                else:
                    Global._error("set_delay: only float, int or np.array values are possible.")

                # The new max_delay is higher than before
                if value > self.max_delay:
                    self.max_delay = value
                    self.uniform_delay = value
                    self.cyInstance.set_delay(value)
                    if isinstance(self.pre, PopulationView):
                        self.pre.population.max_delay = max(self.max_delay, self.pre.population.max_delay)
                        self.pre.population.cyInstance.update_max_delay(self.pre.population.max_delay)
                    else:
                        self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
                        self.pre.cyInstance.update_max_delay(self.pre.max_delay)
                    return
                else:
                    self.uniform_delay = value
                    self.cyInstance.set_delay(value)

            else: # variable delays
                if not isinstance(value, (np.ndarray, list)):
                    Global._error("set_delay with variable delays: you must provide a list of lists of exactly the same size as before.")

                # Check the number of delays
                nb_values = sum([len(s) for s in value])
                if nb_values != self.nb_synapses:
                    Global._error("set_delay with variable delays: the sizes do not match. You have to provide one value for each existing synapse.")
                if len(value) != len(self.post_ranks):
                    Global._error("set_delay with variable delays: the sizes do not match. You have to provide one value for each existing synapse.")

                # Convert to steps
                if isinstance(value, np.ndarray):
                    delays = [[max(1, round(value[i, j]/Global.config['dt'])) for j in range(value.shape[1])] for i in range(value.shape[0])]
                else:
                    delays = [[max(1, round(v/Global.config['dt'])) for v in c] for c in value]

                # Max delay
                max_delay = max([max(l) for l in delays])

                if max_delay > self.max_delay:
                    self.max_delay = max_delay

                    # Send the max delay to the pre population
                    if isinstance(self.pre, PopulationView):
                        self.pre.population.max_delay = max(self.max_delay, self.pre.population.max_delay)
                        self.pre.population.cyInstance.update_max_delay(self.pre.population.max_delay)
                    else:
                        self.pre.max_delay = max(self.max_delay, self.pre.max_delay)
                        self.pre.cyInstance.update_max_delay(self.pre.max_delay)

                # Send the new values to the projection
                self.cyInstance.set_delay(delays)

                # Update ring buffers (if there exist)
                self.cyInstance.update_max_delay(self.max_delay)

        else: # before compile()
            Global._error("set_delay before compile(): not implemented yet.")


    ################################
    ## Access to functions
    ################################
    def _function(self, func):
        "Access a user defined function"
        if not self.initialized:
            Global._error('the network is not compiled yet, cannot access the function ' + func)

        return getattr(self.cyInstance, func)

    ################################
    ## Learning flags
    ################################
    def enable_learning(self, period=None, offset=None):
        """
        Enables learning for all the synapses of this projection.

        For example, providing the following parameters at time 10 ms:

        ```python
        enable_learning(period=10., offset=5.)
        ```

        would call the updating methods at times 15, 25, 35, etc...

        The default behaviour is that the synaptic variables are updated at each time step. The parameters must be multiple of ``dt``.

        :param period: determines how often the synaptic variables will be updated.
        :param offset: determines the offset at which the synaptic variables will be updated relative to the current time.

        """
        # Check arguments
        if not period is None and not offset is None:
            if offset >= period:
                Global._error('enable_learning(): the offset must be smaller than the period.')

        if period is None and not offset is None:
            Global._error('enable_learning(): if you define an offset, you have to define a period.')

        try:
            self.cyInstance._set_update(True)
            self.cyInstance._set_plasticity(True)
            if period != None:
                self.cyInstance._set_update_period(int(period/Global.config['dt']))
            else:
                self.cyInstance._set_update_period(int(1))
                period = Global.config['dt']
            if offset != None:
                relative_offset = Global.get_time() % period + offset
                self.cyInstance._set_update_offset(int(int(relative_offset%period)/Global.config['dt']))
            else:
                self.cyInstance._set_update_offset(int(0))
        except:
            Global._warning('Enable_learning() is only possible after compile()')

    def disable_learning(self, update=None):
        """
        Disables learning for all synapses of this projection.

        The effect depends on the rate-coded or spiking nature of the projection:

        * **Rate-coded**: the updating of all synaptic variables is disabled (including the weights ``w``). This is equivalent to ``proj.update = False``.

        * **Spiking**: the updating of the weights ``w`` is disabled, but all other variables are updated. This is equivalent to ``proj.plasticity = False``.

        This method is useful when performing some tests on a trained network without messing with the learned weights.
        """
        try:
            if self.synapse_type.type == 'rate':
                self.cyInstance._set_update(False)
            else:
                self.cyInstance._set_plasticity(False)
        except:
            Global._warning('disabling learning is only possible after compile().')


    ################################
    ## Methods on connectivity matrix
    ################################

    def save_connectivity(self, filename):
        """
        Saves the connectivity of the projection into a file.

        Only the connectivity matrix, the weights and delays are saved, not the other synaptic variables.

        The generated data can be used to create a projection in another network:

        ```python
        proj.connect_from_file(filename)
        ```

        * If the file name is '.npz', the data will be saved and compressed using `np.savez_compressed` (recommended).

        * If the file name ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * If the file name is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * Otherwise, the data will be pickled into a simple binary text file using pickle.

        :param filename: file name, may contain relative or absolute path.

        """
        # Check that the network is compiled
        if not self.initialized:
            Global._error('save_connectivity(): the network has not been compiled yet.')
            return

        # Check if the repertory exist
        (path, fname) = os.path.split(filename)

        if not path == '':
            if not os.path.isdir(path):
                Global._print('Creating folder', path)
                os.mkdir(path)

        extension = os.path.splitext(fname)[1]

        # Gathering the data
        data = {
            'name': self.name,
            'post_ranks': self.post_ranks,
            'pre_ranks': np.array(self.cyInstance.pre_rank_all(), dtype=object),
            'w': np.array(self.w, dtype=object),
            'delay': np.array(self.cyInstance.get_delay(), dtype=object) if hasattr(self.cyInstance, 'get_delay') else None,
            'max_delay': self.max_delay,
            'uniform_delay': self.uniform_delay,
            'size': self.size,
            'nb_synapses': self.cyInstance.nb_synapses()
        }

        # Save the data
        if extension == '.gz':
            Global._print("Saving connectivity in gunzipped binary format...")
            try:
                import gzip
            except:
                Global._error('gzip is not installed.')
                return
            with gzip.open(filename, mode = 'wb') as w_file:
                try:
                    pickle.dump(data, w_file, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    Global._print('Error while saving in gzipped binary format.')
                    Global._print(e)
                    return

        elif extension == '.npz':
            Global._print("Saving connectivity in Numpy format...")
            np.savez_compressed(filename, **data )

        elif extension == '.mat':
            Global._print("Saving connectivity in Matlab format...")
            if data['delay'] is None:
                data['delay'] = 0
            try:
                import scipy.io as sio
                sio.savemat(filename, data)
            except Exception as e:
                Global._error('Error while saving in Matlab format.')
                Global._print(e)
                return

        else:
            Global._print("Saving connectivity in text format...")
            # save in Pythons pickle format
            with open(filename, mode = 'wb') as w_file:
                try:
                    pickle.dump(data, w_file, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    Global._print('Error while saving in text format.')
                    Global._print(e)
                    return
            return

    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        """
        Gathers all receptive fields within this projection.

        :param variable: name of the variable
        :param in_post_geometry: if False, the data will be plotted as square grid. (default = True)
        """
        if in_post_geometry:
            x_size = self.post.geometry[1]
            y_size = self.post.geometry[0]
        else:
            x_size = int( math.floor(math.sqrt(self.post.size)) )
            y_size = int( math.ceil(math.sqrt(self.post.size)) )


        def get_rf(rank): # TODO: IMPROVE
            res = np.zeros( self.pre.size )
            for n in range(len(self.post_ranks)):
                if self.post_ranks[n] == n:
                    pre_ranks = self.cyInstance.pre_rank(n)
                    data = self.cyInstance.get_local_attribute_row(variable, rank, Global.config["precision"])
                    for j in range(len(pre_ranks)):
                        res[pre_ranks[j]] = data[j]
            return res.reshape(self.pre.geometry)

        res = np.zeros((1, x_size*self.pre.geometry[1]))
        for y in range ( y_size ):
            row = np.concatenate(  [ get_rf(self.post.rank_from_coordinates( (y, x) ) ) for x in range ( x_size ) ], axis = 1)
            res = np.concatenate((res, row))

        return res

    def connectivity_matrix(self, fill=0.0):
        """
        Returns a dense connectivity matrix (2D Numpy array) representing the connections between the pre- and post-populations.

        The first index of the matrix represents post-synaptic neurons, the second the pre-synaptic ones.

        If PopulationViews were used for creating the projection, the matrix is expanded to the whole populations by default.

        :param fill: value to put in the matrix when there is no connection (default: 0.0).
        """
        if not self.initialized:
            Global._error('The connectivity matrix can only be accessed after compilation')

        # get correct dimensions for dense matrix
        if isinstance(self.pre, PopulationView):
            size_pre = self.pre.population.size
        else:
            size_pre = self.pre.size
        if isinstance(self.post, PopulationView):
            size_post = self.post.population.size
        else:
            size_post = self.post.size

        # create empty dense matrix with default values
        res = np.ones((size_post, size_pre)) * fill

        # fill row-by-row with real values
        for rank in self.post_ranks:
            idx = self.post_ranks.index(rank)
            preranks = self.cyInstance.pre_rank(idx)
            if "w" in self.synapse_type.description['local'] and (not self._has_single_weight()):
                w = self.cyInstance.get_local_attribute_row("w", idx, Global.config["precision"])
            elif "w" in self.synapse_type.description['semiglobal']:
                w = self.cyInstance.get_semiglobal_attribute("w", idx, Global.config["precision"])*np.ones(self.cyInstance.dendrite_size(idx))
            else:
                w = self.cyInstance.get_global_attribute("w", Global.config["precision"])*np.ones(self.cyInstance.dendrite_size(idx))
            res[rank, preranks] = w
        return res


    ################################
    ## Save/load methods
    ################################

    def _data(self):
        "Method gathering all info about the projection when calling save()"

        if not self.initialized:
            Global._error('save_connectivity(): the network has not been compiled yet.')

        desc = {}
        desc['name'] = self.name
        desc['pre'] = self.pre.name
        desc['post'] = self.post.name
        desc['target'] = self.target
        desc['post_ranks'] = self.post_ranks
        desc['attributes'] = self.attributes
        desc['parameters'] = self.parameters
        desc['variables'] = self.variables
        desc['delays'] = self._get_delay()

        # Determine if we have varying number of elements per row
        # based on the pre-synaptic ranks
        pre_ranks = self.cyInstance.pre_rank_all()
        dend_size = len(pre_ranks[0])
        ragged_list = False
        for i in range(1, len(pre_ranks)):
            if len(pre_ranks[i]) != dend_size:
                ragged_list = True
                break

        # Save pre_ranks
        if ragged_list:
            desc['pre_ranks'] = np.array(self.cyInstance.pre_rank_all(), dtype=object)
        else:
            desc['pre_ranks'] = np.array(self.cyInstance.pre_rank_all())

        # Attributes to save
        attributes = self.attributes
        if not 'w' in self.attributes:
            attributes.append('w')

        # Save all attributes
        for var in attributes:
            try:
                ctype = self._get_attribute_cpp_type(var)
                if var == "w" and self._has_single_weight():
                    desc[var] = self.cyInstance.get_global_attribute("w", ctype)

                elif var in self.synapse_type.description['local']:
                    if ragged_list:
                        desc[var] = np.array(self.cyInstance.get_local_attribute_all(var, ctype), dtype=object)
                    else:
                        desc[var] = self.cyInstance.get_local_attribute_all(var, ctype)
                elif var in self.synapse_type.description['semiglobal']:
                    desc[var] = self.cyInstance.get_semiglobal_attribute_all(var, ctype)
                else:
                    desc[var] = self.cyInstance.get_global_attribute(var, ctype) # linear array or single constant
            except:
                Global._warning('Can not save the attribute ' + var + ' in the projection.')

        return desc

    def save(self, filename):
        """
        Saves all information about the projection (connectivity, current value of parameters and variables) into a file.

        * If the file name is '.npz', the data will be saved and compressed using `np.savez_compressed` (recommended).

        * If the file name ends with '.gz', the data will be pickled into a binary file and compressed using gzip.

        * If the file name is '.mat', the data will be saved as a Matlab 7.2 file. Scipy must be installed.

        * Otherwise, the data will be pickled into a simple binary text file using pickle.

        :param filename: file name, may contain relative or absolute path.

        **Warning:** the '.mat' data will not be loadable by ANNarchy, it is only for external analysis purpose.

        Example:

        ```python
        proj.save('proj1.npz')
        proj.save('proj1.txt')
        proj.save('proj1.txt.gz')
        proj.save('proj1.mat')
        ```
        """
        from ANNarchy.core.IO import _save_data
        _save_data(filename, self._data())


    def load(self, filename):
        """
        Loads the saved state of the projection by `Projection.save()`.

        Warning: Matlab data can not be loaded.

        Example:

        ```python
        proj.load('proj1.npz')
        proj.load('proj1.txt')
        proj.load('proj1.txt.gz')
        ```

        :param filename: the file name with relative or absolute path.
        """
        from ANNarchy.core.IO import _load_connectivity_data
        self._load_proj_data(_load_connectivity_data(filename))


    def _load_proj_data(self, desc):
        """
        Updates the projection with the stored data set.
        """
        # Sanity check
        if desc == None:
            # _load_proj should have printed an error message
            return

        # If it's not saveable there is nothing to load
        if not self._saveable:
            return

        # Check deprecation
        if not 'attributes' in desc.keys():
            Global._error('The file was saved using a deprecated version of ANNarchy.')
            return
        if 'dendrites' in desc: # Saved before 4.5.3
            Global._error("The file was saved using a deprecated version of ANNarchy.")
            return

        # If the post ranks and/or pre-ranks have changed, overwrite
        connectivity_changed=False
        if 'post_ranks' in desc and not np.all((desc['post_ranks']) == self.post_ranks):
            connectivity_changed=True
        if 'pre_ranks' in desc and not np.all((desc['pre_ranks']) == np.array(self.cyInstance.pre_rank_all(), dtype=object)):
            connectivity_changed=True

        # synaptic weights
        weights = desc["w"]

        # Delays can be either uniform (int, float) or non-uniform (np.ndarray).
        # HD (30th May 2022):
        #   Unfortunately, the storage of constants changed over the time. At the
        #   end of this code block, we should have either a single constant or a
        #   numpy nd-array
        delays = 0
        if 'delays' in desc:
            delays = desc['delays']

            if isinstance(delays, (float, int)):
                # will be handled below
                pass

            elif isinstance(delays, np.ndarray):
                # constants are stored as 0-darray
                if delays.ndim == 0:
                    # transform into single float
                    delays = float(delays)
                else:
                    # nothing to do as it is numpy nd-array
                    pass

            else:
                # ragged list to nd-array
                delays = np.array(delays, dtype=object)

        # Some patterns like fixed_number_pre/post or fixed_probability change the
        # connectivity. If this is not the case, we can simply set the values.
        if connectivity_changed:
            # (re-)initialize connectivity
            if isinstance(delays, (float, int)):
                delays = [[delays]] # wrapper expects list from list

            self.cyInstance.init_from_lil(desc['post_ranks'], desc['pre_ranks'], weights, delays)
        else:
            # set weights
            self._set_cython_attribute("w", weights)

            # set delays if there were some
            self._set_delay(delays)

        # Other variables
        for var in desc['attributes']:
            if var == "w":
                continue # already done

            try:
                self._set_cython_attribute(var, desc[var])
            except Exception as e:
                Global._print(e)
                Global._warning('load(): the variable', var, 'does not exist in the current version of the network, skipping it.')
                continue

    ################################
    ## Structural plasticity
    ################################
    def start_pruning(self, period=None):
        """
        Starts pruning the synapses in the projection if the synapse defines a 'pruning' argument.

        'structural_plasticity' must be set to True in setup().

        :param period: how often pruning should be evaluated (default: dt, i.e. each step)
        """
        if not period:
            period = Global.config['dt']
        if not self.cyInstance:
            Global._error('Can not start pruning if the network is not compiled.')

        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.start_pruning(int(period/Global.config['dt']), Global.get_current_step())
            except :
                Global._error("The synapse does not define a 'pruning' argument.")

        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start pruning connections.")


    def stop_pruning(self):
        """
        Stops pruning the synapses in the projection if the synapse defines a 'pruning' argument.

        'structural_plasticity' must be set to True in setup().
        """
        if not self.cyInstance:
            Global._error('Can not stop pruning if the network is not compiled.')

        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.stop_pruning()
            except:
                Global._error("The synapse does not define a 'pruning' argument.")

        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start pruning connections.")

    def start_creating(self, period=None):
        """
        Starts creating the synapses in the projection if the synapse defines a 'creating' argument.

        'structural_plasticity' must be set to True in setup().

        :param period: how often creating should be evaluated (default: dt, i.e. each step)
        """
        if not period:
            period = Global.config['dt']
        if not self.cyInstance:
            Global._error('Can not start creating if the network is not compiled.')

        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.start_creating(int(period/Global.config['dt']), Global.get_current_step())
            except:
                Global._error("The synapse does not define a 'creating' argument.")

        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start creating connections.")

    def stop_creating(self):
        """
        Stops creating the synapses in the projection if the synapse defines a 'creating' argument.

        'structural_plasticity' must be set to True in setup().
        """
        if not self.cyInstance:
            Global._error('Can not stop creating if the network is not compiled.')

        if Global.config['structural_plasticity']:
            try:
                self.cyInstance.stop_creating()
            except:
                Global._error("The synapse does not define a 'creating' argument.")

        else:
            Global._error("You must set 'structural_plasticity' to True in setup() to start creating connections.")

    ################################
    ## Memory Management
    ################################
    def size_in_bytes(self):
        """
        Returns the size in bytes of the allocated memory on C++ side. Note that this does not reflect monitored data and that it only works after compile() was invoked.
        """
        if self.initialized:
            return self.cyInstance.size_in_bytes()
        else:
            return 0

    def _clear(self):
        """
        Deallocates the container within the C++ instance. The population object is not usable anymore after calling this function.

        Warning: should be only called by the net deconstructor (in the context of parallel_run).
        """
        if self.initialized:
            self.cyInstance.clear()
            self.initialized = False
