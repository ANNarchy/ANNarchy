"""
Implements the built-in connectivity patterns available in ANNarchy. See the documentaton
for more details: https://annarchy.readthedocs.io/en/latest/manual/Connector.html

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import numpy as np

from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.parser.report.LatexParser import _process_random
from ANNarchy.intern import Messages
from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import ConfigManager

try:
    from ANNarchy.cython_ext import *
except Exception as e:
    Messages._print(e)

################################
## Connector methods
################################
def connect_one_to_one(self: "Projection", 
                       weights: float | RandomDistribution = 1.0, 
                       delays: float | RandomDistribution = 0.0, 
                       force_multiple_weights:bool=False, 
                       storage_format:str=None, storage_order:str=None) -> "Projection":
    """
    one-to-one connection pattern.

    :param weights: Initial synaptic values, either a single value (float) or a random distribution object.
    :param delays: Synaptic delays, either a single value or a random distribution object (default=dt).
    :param force_multiple_weights: If a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre.size != self.post.size:
        Messages._warning("connect_one_to_one() between", self.pre.name, 'and', self.post.name, 'with target', self.target)
        Messages._print("\t the two populations have different sizes, please check the connection pattern is what you expect.")

    self.connector_name = "One-to-One"
    self.connector_description = "One-to-One, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    if storage_format == "dense":
        Messages._error("The usage of 'dense' storage format on one-to-one pattern is not allowed.")

    # if weights or delays are from random distribution I need to know this in code generator
    self.connector_weight_dist = weights if isinstance(weights, RandomDistribution) else None
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(one_to_one, (weights, delays, storage_format, storage_order), delays, storage_format, storage_order)

    return self

def connect_all_to_all(self: "Projection",
                       weights: float | RandomDistribution,
                       delays: float | RandomDistribution =0.0,
                       allow_self_connections:bool=False,
                       force_multiple_weights:bool=False,
                       storage_format:str=None,
                       storage_order:str=None)  -> "Projection":
    """
    all-to-all (fully-connected) connection pattern.

    :param weights: Synaptic values, either a single value or a random distribution object.
    :param delays: Synaptic delays, either a single value or a random distribution object (default=dt).
    :param allow_self_connections: If True, self-connections between a neuron and itself are allowed (default = False if the pre- and post-populations are identical, True otherwise).
    :param force_multiple_weights: If a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    pre_pop = self.pre if not isinstance(self.pre, PopulationView) else self.pre.population
    post_pop = self.post if not isinstance(self.post, PopulationView) else self.post.population
    if pre_pop != post_pop:
        allow_self_connections = True

    self.connector_name = "All-to-All"
    self.connector_description = "All-to-All, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

    # Does the projection define a single non-plastic weight?
    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    # if weights or delays are from random distribution I need to know this in code generator
    self.connector_weight_dist = weights if isinstance(weights, RandomDistribution) else None
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    # Store the connectivity
    self._store_connectivity(all_to_all, (weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)

    return self

def connect_gaussian(self: "Projection", amp:float, sigma:float, delays: float | RandomDistribution=0.0, limit:float=0.01, allow_self_connections:bool=False, storage_format:str=None)  -> "Projection":
    """
    Gaussian connection pattern.

    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around the neuron with the same normalized coordinates using a Gaussian profile.

    :param amp: Amplitude of the Gaussian function
    :param sigma: Width of the Gaussian function
    :param delays: Synaptic delay, either a single value or a random distribution object (default=dt).
    :param limit: Proportion of `amp` below which synapses are not created
    :param allow_self_connections: Allows connections between a neuron and itself.
    """
    if self.pre != self.post:
        allow_self_connections = True

    if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
        Messages._error('Gaussian connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Gaussian"
    self.connector_description = f"Gaussian, amplitude {amp}, sigma {sigma}, delays {_process_random(delays)}"

    # weights are not drawn, delays possibly
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(gaussian, (amp, sigma, delays, limit, allow_self_connections, storage_format, "post_to_pre"), delays, storage_format, "post_to_pre")
    
    return self

def connect_dog(self: "Projection", amp_pos:float, sigma_pos:float, amp_neg:float, sigma_neg:float, delays:float | RandomDistribution=0.0, limit:float=0.01, allow_self_connections:bool=False, storage_format:str=None)  -> "Projection":
    """
    Difference-Of-Gaussians connection pattern.

    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around the neuron with the same normalized coordinates using a Difference-Of-Gaussians profile.

    :param amp_pos: Amplitude of the positive Gaussian function
    :param sigma_pos: Width of the positive Gaussian function
    :param amp_neg: Amplitude of the negative Gaussian function
    :param sigma_neg: Width of the negative Gaussian function
    :param delays: Synaptic delay, either a single value or a random distribution object (default=dt).
    :param limit: Proportion of *amp* below which synapses are not created (default: 0.01)
    :param allow_self_connections: Allows connections between a neuron and itself.
    """
    if self.pre != self.post:
        allow_self_connections = True

    if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
        Messages._error('DoG connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Difference-of-Gaussian"
    self.connector_description = f"Difference-of-Gaussian, A+ {amp_pos}, sigma+ {sigma_pos}, A- {amp_neg}, sigma- {sigma_neg}, delays {_process_random(delays)}"

    # delays are possibly drawn from distribution, weights not
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(dog, (amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections, storage_format, "post_to_pre"), delays, storage_format, "post_to_pre")

    return self

def connect_fixed_probability(self: "Projection", probability:float, weights:float | RandomDistribution, delays:float | RandomDistribution=0.0, allow_self_connections:bool=False, force_multiple_weights:bool=False, storage_format:str=None, storage_order:str=None)  -> "Projection":
    """
    Probabilistic sparse connection pattern.

    Each neuron in the postsynaptic population is connected to neurons of the presynaptic population with the given probability. Self-connections are avoided by default.

    :param probability: Probability that a synapse is created.
    :param weights: Either a single value for all synapses or a RandomDistribution object.
    :param delays: Either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: Defines if self-connections are allowed (default=False).
    :param force_multiple_weights: If a single value is provided for `weights` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting `force_multiple_weights` to True ensures that a value per synapse will be used.
    """
    if self.pre != self.post:
        allow_self_connections = True

    self.connector_name = "Random"
    self.connector_description = "Random, sparseness %(proba)s, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays), 'proba': probability}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    # if weights or delays are from random distribution I need to know this in code generator
    self.connector_weight_dist = weights if isinstance(weights, RandomDistribution) else None
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(fixed_probability, (probability, weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)

    return self

def connect_fixed_number_pre(self: "Projection", number:int, weights: float | RandomDistribution, delays: float | RandomDistribution=0.0, allow_self_connections:bool=False, force_multiple_weights:bool=False, storage_format:str=None, storage_order:str=None)  -> "Projection":
    """
    Connection pattern where each post-synaptic neuron receives a fixed number of pre-synaptic neurons.

    :param number: Number of synapses per postsynaptic neuron.
    :param weights: Either a single value for all synapses or a RandomDistribution object.
    :param delays: Either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: Defines if self-connections are allowed (default=False).
    :param force_multiple_weights: If a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre != self.post:
        allow_self_connections = True

    if number > self.pre.size:
        Messages._error('fixed_number_pre: the number of pre-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Convergent"
    self.connector_description = "Random Convergent %(number)s $\\rightarrow$ 1, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    # if weights or delays are from random distribution I need to know this in code generator
    self.connector_weight_dist = weights if isinstance(weights, RandomDistribution) else None
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(fixed_number_pre, (number, weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)

    return self

def connect_fixed_number_post(self: "Projection", number:int, weights:float | RandomDistribution=1.0, delays:float | RandomDistribution=0.0, allow_self_connections:bool=False, force_multiple_weights:bool=False, storage_format:str=None, storage_order:str=None)  -> "Projection":
    """
    Each pre-synaptic neuron randomly sends a fixed number of connections to the post-synaptic neurons.

    :param number: Number of synapses per pre-synaptic neuron.
    :param weights: Either a single value for all synapses or a RandomDistribution object.
    :param delays: Either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: Defines if self-connections are allowed (default=False)
    :param force_multiple_weights: If a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre != self.post:
        allow_self_connections = True

    if number > self.post.size:
        Messages._error('fixed_number_post: the number of post-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Divergent"
    self.connector_description = "Random Divergent 1 $\\rightarrow$ %(number)s, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    # if weights or delays are from random distribution I need to know this in code generator
    self.connector_weight_dist = weights if isinstance(weights, RandomDistribution) else None
    self.connector_delay_dist = delays if isinstance(delays, RandomDistribution) else None

    self._store_connectivity(fixed_number_post, (number, weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)
    return self


def connect_with_func(self: "Projection", method, storage_format:str=None, storage_order:str=None, **args) -> "Projection":
    """
    Connection pattern based on a user-defined function.

    The two first arguments of the function must be the pre and post populations. The third argument is the step size *dt*.
    Additional arguments can be passed at creation time.

    The function must return a `ann.LILConnectivity` object.

    Example:

    ```python
    def probabilistic_pattern(pre, post, dt, weight, probability):
        # Create a LIL structure for the connectivity matrix
        synapses = ann.LILConnectivity(dt=dt)
        # For all neurons in the post-synaptic population
        for post_rank in range(post.size):
            # Decide which pre-synaptic neurons should form synapses
            ranks = []
            for pre_rank in range(pre.size):
                if numpy.random.random() < probability:
                    ranks.append(pre_rank)
            # Create weights and delays arrays of the same size
            values = [weight for i in range(len(ranks)) ]
            delays = [0 for i in range(len(ranks)) ]
            # Add this information to the LIL matrix
            synapses.add(post_rank, ranks, values, delays)

        return synapses

    proj = net.connect(pop1, pop2, target = 'inh')
    proj.from_function(
        method=probabilistic_pattern, 
        weight=1.0, 
        probability=0.3
    )
    ```

    :param method: Method to call. The method **must** return a LILConnectivity object.
    :param args: List of additional arguments needed by the function.
    """
    # Construct the pattern
    synapses = method(self.pre, self.post, ConfigManager().get('dt', net_id=self.pre.net_id), **args)

    # Sanity check: doublons, ascending order of ranks
    synapses.validate()

    # No delay or uniform delay: single value / non-uniform delay: LIL
    delays = synapses.max_delay if synapses.uniform_delay != -1 else synapses.delay

    # Store for later initialization
    self._store_connectivity(self._load_from_lil, (synapses, ), delays, storage_format=storage_format, storage_order=storage_order)

    # Report
    self.connector_name = "User-defined"
    self.connector_description = "Created by the method " + method.__name__
    return self

def connect_from_matrix_market(self: "Projection", filename:str, storage_format:str=None, storage_order:str=None) -> "Projection":
    """
    Loads a weight matrix encoded in the Matrix Market format. This connector is intended for benchmarking purposes.

    :param filename: Filename of the Matrix Market (.mtx) file.
    """
    from scipy.io import mmread
    from scipy.sparse import coo_matrix

    from ANNarchy.cython_ext import LILConnectivity
    if not filename.endswith(".mtx"):
        raise ValueError("from_matrix_market(): expected .mtx file.")

    # read with SciPy
    tmp = mmread(filename)

    # scipy should return a coo_matrix in case of sparse matrices
    if isinstance(tmp, coo_matrix):
        # transform into LIL (in place)
        tmp = tmp.tolil(copy=True)

        # build up ANNarchy LIL
        synapses = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))
        row_idx = 0
        for col_idx, val in zip(tmp.rows, tmp.data):
            synapses.push_back(row_idx, col_idx, val, [0])
            row_idx+=1

    elif isinstance(tmp, np.ndarray):
        # build up ANNarchy LIL
        synapses = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))

        col_idx = np.arange(tmp.shape[1])
        for row_idx in range(tmp.shape[0]):
            synapses.push_back(row_idx, col_idx, tmp[row_idx,:], [0])

    else:
        raise ValueError("Error on read-out of matrix market file.")

    # not needed anymore
    del tmp

    delays = 0

    self._store_connectivity(self._load_from_lil, (synapses, ), delays, storage_format=storage_format, storage_order=storage_order)

    self.connector_name = "MatrixMarket"
    self.connector_description = "A weight matrix load from .mtx file"
    return self

def _load_from_lil(self: "Projection", pre: "Population", post: "Population", synapses: LILConnectivity):
    """
    Load from LILConnectivity instance.
    """
    if not isinstance(synapses, LILConnectivity):
        Messages._error(f"_load_from_lil(): expected a LILConnectivty instance (Projection: pre={pre.name}, post={post.name}, name='{self.name}').")

    return synapses

def connect_from_matrix(self, weights: np.array, delays=0.0, pre_post=False, storage_format=None, storage_order=None) -> "Projection":
    """
    Builds a connection pattern according to a dense connectivity matrix.

    The matrix must be N*M, where N is the number of neurons in the post-synaptic population and M in the pre-synaptic one. Lists of lists must have the same size.

    If a synapse should not be created, the weight value should be None.

    :param weights: Numpy array (or list of lists of equal size) representing the weights. If a value is None, the corresponding synapse will not be created.
    :param delays: Numpy array representing the delays. Must represent the same synapses as the `weights` argument. If omitted, the delays are considered 0.
    :param pre_post: States which index is first. By default, the first dimension is related to the post-synaptic population. If ``pre_post`` is True, the first dimension is the pre-synaptic population.
    """

    # Store the synapses
    self.connector_name = "Connectivity matrix"
    self.connector_description = "Connectivity matrix"

    if isinstance(weights, list):
        try:
            weights = np.array(weights)
        except:
            Messages._error('from_matrix(): You must provide a dense 2D matrix.')

    self._store_connectivity(self._load_from_matrix, (weights, delays, pre_post), delays, storage_format, storage_order)

    return self

def _load_from_matrix(self, pre, post, weights, delays, pre_post):
    """
    Initializes a connectivity matrix between two populations based on a provided matrix.
    This connector method always return a LIL-structure.

    :param pre: pre-synaptic Population instance
    :param post: post-synaptic Population instance
    :param weights: matrix / list-in-list which contains synaptic weights
    :param delays: matrix / list-in-list which contains synaptic delays
    :param pre_post: needs to be set to True if the weights are not a post times pre matrix.
    """
    lil = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))

    uniform_delay = not isinstance(delays, (list, np.ndarray))
    if isinstance(delays, list):
        try:
            delays = np.array(delays)
        except:
            Messages._error('from_matrix(): You must provide a dense 2D matrix.')

    if pre_post: # if the user prefers pre as the first index...
        weights = weights.T
        if isinstance(delays, np.ndarray):
            delays = delays.T

    shape = weights.shape
    if shape != (self.post.size, self.pre.size):
        if not pre_post:
            Messages._print("ERROR: from_matrix(): the matrix does not have the correct dimensions.")
            Messages._print('Expected:', (self.post.size, self.pre.size))
            Messages._print('Received:', shape)

        else:
            Messages._print("ERROR: from_matrix(): the matrix does not have the correct dimensions.")
            Messages._print('Expected:', (self.pre.size, self.post.size))
            Messages._print('Received:', shape)
        Messages._error('Quitting...')

    for i in range(self.post.size):
        if isinstance(self.post, PopulationView):
            rk_post = self.post.ranks[i]
        else:
            rk_post = i
        r = []
        w = []
        d = []
        for j in range(self.pre.size):
            val = weights[i, j]
            if val != None:
                if isinstance(self.pre, PopulationView):
                    rk_pre = self.pre.ranks[j]
                else:
                    rk_pre = j
                r.append(rk_pre)
                w.append(val)
                if not uniform_delay:
                    d.append(delays[i, j])
        if uniform_delay:
            d.append(delays)
        if len(r) > 0:
            lil.add(rk_post, r, w, d)

    return lil

def connect_from_sparse(self, weights:"scipy.sparse.lil_matrix", delays: int | float=0.0, storage_format:str=None, storage_order:str=None) -> "Projection":
    """
    Builds a connectivity pattern using a Scipy sparse matrix for the weights and (optionally) delays.

    Warning: a sparse matrix has pre-synaptic ranks as first dimension.

    :param weights: a sparse lil_matrix object created from scipy.
    :param delays: the value of the constant delay (default: dt). Variable delays are not allowed.
    """
    try:
        from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
    except:
        Messages._error("from_sparse(): scipy is not installed, sparse matrices can not be loaded.")

    if not isinstance(weights, (lil_matrix, csr_matrix, csc_matrix)):
        Messages._error("from_sparse(): only lil, csr and csc matrices are allowed for now.")

    if not isinstance(delays, (int, float)):
        Messages._error("from_sparse(): only constant delays are allowed for sparse matrices.")

    weights = csc_matrix(weights)

    # if weights[weights.nonzero()].max() == weights[weights.nonzero()].min() :
    #     self._single_constant_weight = True

    # Store the synapses
    self.connector_name = "Sparse connectivity matrix"
    self.connector_description = "Sparse connectivity matrix"
    self._store_connectivity(self._load_from_sparse, (weights, delays), delays, storage_format, storage_order)

    return self

def _load_from_sparse(self, pre, post, weights, delays):
    # Create an empty LIL object
    lil = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))

    # Find offsets
    if isinstance(self.pre, PopulationView):
        pre_ranks = self.pre.ranks
    else:
        pre_ranks = [i for i in range(self.pre.size)]

    if isinstance(self.post, PopulationView):
        post_ranks = self.post.ranks
    else:
        post_ranks = [i for i in range(self.post.size)]

    # Process the sparse matrix and fill the lil
    weights.sort_indices()
    (pre, post) = weights.shape

    if (pre, post) != (len(pre_ranks), len(post_ranks)):
        Messages._print("ERROR: from_sparse(): the sparse matrix does not have the correct dimensions.")
        Messages._print('Expected:', (len(pre_ranks), len(post_ranks)))
        Messages._print('Received:', (pre, post))
        Messages._error('Quitting...')

    for idx_post in range(post):
        idx_pre = weights.getcol(idx_post).indices
        w = weights.getcol(idx_post).data
        pr = [pre_ranks[i] for i in idx_pre]
        lil.add(post_ranks[idx_post], pr, w, [float(delays)])

    return lil

def connect_from_file(self, filename:str, pickle_encoding:str=None, storage_format:str=None, storage_order:str=None)  -> "Projection":
    """
    Builds the connectivity matrix using data saved using `Projection.save_connectivity()` (not `save()`!).

    Admissible file formats are compressed Numpy files (.npz), gunzipped binary text files (.gz) or binary text files.

    Note: Only the ranks, weights and delays are loaded, not the other variables.

    :param filename: file where the connections were saved.
    """
    # Create an empty LIL object
    lil = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))

    # Load the data
    from ANNarchy.core.IO import _load_connectivity_data
    try:
        data = _load_connectivity_data(filename, pickle_encoding)
    except Exception as e:
        Messages._print(e)
        Messages._error('from_file(): Unable to load the data', filename, 'into the projection.')

    # Load the LIL object
    try:
        # Size
        lil.size = data['size']
        lil.nb_synapses = data['nb_synapses']

        # Ranks
        lil.post_rank = list(data['post_ranks'])
        lil.pre_rank = list(data['pre_ranks'])

        # Weights
        single_w = False
        if isinstance(data['w'], (int, float)):
            single_w = True
            lil.w = [[float(data['w'])]]
        elif isinstance(data['w'], (np.ndarray,)) and data['w'].size == 1:
            single_w = True
            lil.w = [[float(data['w'])]]
        else:
            lil.w = data['w']

        if NetworkManager().get_network(net_id = self.net_id).compiled:
            # We have already compiled the network, so changing the flag will result in crashes when accessing 'w'
            if single_w != self._single_constant_weight:
                Messages._print("Projection (name="+self.name+"): potential mismatch between weight vector in the projection and the save file.")
                Messages._print("    single weight in file:", single_w)
                Messages._print("    single weight in network:", self._single_constant_weight)
        else:
            # we have not yet compiled, so simply adjust the flag
            self._single_constant_weight = single_w

        # Delays
        lil.max_delay = data['max_delay']
        lil.uniform_delay = data['uniform_delay']
        if data['delay'] is not None:
            if lil.uniform_delay == -1:
                lil.delay = [list(np.array(tmp) / ConfigManager().get("dt", self.net_id)) for tmp in data['delay']]
            else:
                lil.delay = [[lil.max_delay]]

    except Exception as e:
        Messages._print(e)
        Messages._error('Unable to load the data', filename, 'into the projection.')

    # Store the synapses
    self.connector_name = "From File"
    self.connector_description = "From File"
    self._store_connectivity(self._load_from_lil, (lil,), lil.max_delay if lil.uniform_delay > 0 else lil.delay, storage_format=storage_format, storage_order=storage_order)

    return self
