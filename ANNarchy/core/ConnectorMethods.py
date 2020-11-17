#===============================================================================
#
#     ConnectorMethods.py
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

from ANNarchy.core import Global
from ANNarchy.core.Random import Uniform
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.parser.report.LatexParser import _process_random

try:
    from ANNarchy.core.cython_ext import *
except Exception as e:
    Global._print(e)

################################
## Connector methods
################################
def connect_one_to_one(self, weights=1.0, delays=0.0, force_multiple_weights=False):
    """
    Builds a one-to-one connection pattern between the two populations.

    :param weights: initial synaptic values, either a single value (float) or a random distribution object.
    :param delays: synaptic delays, either a single value or a random distribution object (default=dt).
    :param force_multiple_weights: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre.size != self.post.size:
        Global._warning("connect_one_to_one() between", self.pre.name, 'and', self.post.name, 'with target', self.target)
        Global._print("\t the two populations have different sizes, please check the connection pattern is what you expect.")

    self.connector_name = "One-to-One"
    self.connector_description = "One-to-One, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity( one_to_one, (weights, delays, "lil", "post_to_pre"), delays )
    return self

def connect_all_to_all(self, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False, storage_format="lil", storage_order="post_to_pre"):
    """
    Builds an all-to-all connection pattern between the two populations.

    :param weights: synaptic values, either a single value or a random distribution object.
    :param delays: synaptic delays, either a single value or a random distribution object (default=dt).
    :param allow_self_connections: if True, self-connections between a neuron and itself are allowed (default = False if the pre- and post-populations are identical, True otherwise).
    :param force_multiple_weights: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    :param storage_format: for some of the default connection patterns, ANNarchy provide different storage formats. For all-to-all we support list-of-list ("lil") or compressed sparse row ("csr"), by default lil is chosen.
    :param storage_order: for some of the available storage formats, ANNarchy provides different storage orderings. For all-to-all we support pre_to_post and post_to_pre, by default post_to_pre is chosen.

    Please note, the last two arguments should be changed carefully, as they can have large impact on the computational performance of ANNarchy.
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

    # Is it a dense connectivity matrix?
    if allow_self_connections and not isinstance(self.pre, PopulationView) and not isinstance(self.post, PopulationView):
        # TODO: for the moment disabled as it is not implemented
        # correctly (HD (15. Feb. 2019))
        #self._dense_matrix = True
        self._dense_matrix = False

    # Store the connectivity
    self._store_connectivity( all_to_all, (weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order )
    return self

def connect_gaussian(self, amp, sigma, delays=0.0, limit=0.01, allow_self_connections=False):
    """
    Builds a Gaussian connection pattern between the two populations.

    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around
    the neuron with the same normalized coordinates using a Gaussian profile.

    :param amp: amplitude of the Gaussian function
    :param sigma: width of the Gaussian function
    :param delays: synaptic delay, either a single value or a random distribution object (default=dt).
    :param limit: proportion of *amp* below which synapses are not created (default: 0.01)
    :param allow_self_connections: allows connections between a neuron and itself.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
        Global._error('Gaussian connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Gaussian"
    self.connector_description = "Gaussian, $A$ %(A)s, $\sigma$ %(sigma)s, delays %(delay)s"% {'A': str(amp), 'sigma': str(sigma), 'delay': _process_random(delays)}

    self._store_connectivity( gaussian, (amp, sigma, delays, limit, allow_self_connections, "lil", "post_to_pre"), delays)
    return self

def connect_dog(self, amp_pos, sigma_pos, amp_neg, sigma_neg, delays=0.0, limit=0.01, allow_self_connections=False):
    """
    Builds a Difference-Of-Gaussians connection pattern between the two populations.

    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around
    the neuron with the same normalized coordinates using a Difference-Of-Gaussians profile.

    :param amp_pos: amplitude of the positive Gaussian function
    :param sigma_pos: width of the positive Gaussian function
    :param amp_neg: amplitude of the negative Gaussian function
    :param sigma_neg: width of the negative Gaussian function
    :param delays: synaptic delay, either a single value or a random distribution object (default=dt).
    :param limit: proportion of *amp* below which synapses are not created (default: 0.01)
    :param allow_self_connections: allows connections between a neuron and itself.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
        Global._error('DoG connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Difference-of-Gaussian"
    self.connector_description = "Difference-of-Gaussian, $A^+ %(Aplus)s, $\sigma^+$ %(sigmaplus)s, $A^- %(Aminus)s, $\sigma^-$ %(sigmaminus)s, delays %(delay)s"% {'Aplus': str(amp_pos), 'sigmaplus': str(sigma_pos), 'Aminus': str(amp_neg), 'sigmaminus': str(sigma_neg), 'delay': _process_random(delays)}

    self._store_connectivity( dog, (amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections,  "lil", "post_to_pre"), delays,  "lil", "post_to_pre")
    return self

def connect_fixed_probability(self, probability, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False, storage_format="lil", storage_order="post_to_pre"):
    """
    Builds a probabilistic connection pattern between the two populations.

    Each neuron in the postsynaptic population is connected to neurons of the presynaptic population with the given probability. Self-connections are avoided by default.

    :param probability: probability that a synapse is created.
    :param weights: either a single value for all synapses or a RandomDistribution object.
    :param delays: either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: defines if self-connections are allowed (default=False).
    :param force_multiple_weights: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    :param storage_format: for some of the default connection patterns ANNarchy provide different storage formats. For all-to-all we support list-of-list ("lil") or compressed sparse row ("csr"), by default lil is chosen.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    self.connector_name = "Random"
    self.connector_description = "Random, sparseness %(proba)s, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays), 'proba': probability}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity( fixed_probability, (probability, weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)
    return self

def connect_fixed_number_pre(self, number, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False, storage_format="lil", storage_order="post_to_pre"):
    """
    Builds a connection pattern between the two populations with a fixed number of pre-synaptic neurons.

    Each neuron in the postsynaptic population receives connections from a fixed number of neurons of the presynaptic population chosen randomly.

    :param number: number of synapses per postsynaptic neuron.
    :param weights: either a single value for all synapses or a RandomDistribution object.
    :param delays: either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: defines if self-connections are allowed (default=False).
    :param force_multiple_weights: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if number > self.pre.size:
        Global._error('connect_fixed_number_pre: the number of pre-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Convergent"
    self.connector_description = "Random Convergent %(number)s $\\rightarrow$ 1, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity( fixed_number_pre, (number, weights, delays, allow_self_connections, storage_format, storage_order), delays, storage_format, storage_order)
    return self

def connect_fixed_number_post(self, number, weights=1.0, delays=0.0, allow_self_connections=False, force_multiple_weights=False):
    """
    Builds a connection pattern between the two populations with a fixed number of post-synaptic neurons.

    Each neuron in the pre-synaptic population sends connections to a fixed number of neurons of the post-synaptic population chosen randomly.

    :param number: number of synapses per pre-synaptic neuron.
    :param weights: either a single value for all synapses or a RandomDistribution object.
    :param delays: either a single value for all synapses or a RandomDistribution object (default = dt)
    :param allow_self_connections: defines if self-connections are allowed (default=False)
    :param force_multiple_weights: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if number > self.post.size:
        Global._error('connect_fixed_number_post: the number of post-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Divergent"
    self.connector_description = "Random Divergent 1 $\\rightarrow$ %(number)s, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity( fixed_number_post, (number, weights, delays, allow_self_connections, "lil", "post_to_pre"), delays, "lil", "post_to_pre")
    return self

def connect_with_func(self, method, **args):
    """
    Builds a connection pattern based on a user-defined method.

    :param method: method to call. The method **must** return a CSR object.
    :param args: list of arguments needed by the function
    """
    # Invoke the method directly, we need the delays already....
    synapses = method(self.pre, self.post, **args)
    synapses.validate()

    # Treat delays
    if synapses.uniform_delay != -1: # uniform delay
        d = synapses.max_delay * Global.config['dt']
    else:
        d = Uniform(0., synapses.max_delay * Global.config['dt']) # Just to trick _store_connectivity(), the real delays are in the CSR

    self._store_connectivity(self._load_from_lil, (synapses, ), d)

    self.connector_name = "User-defined"
    self.connector_description = "Created by the method " + method.__name__
    return self

def _load_from_lil(self, pre, post, synapses):
    return synapses

def connect_from_matrix(self, weights, delays=0.0, pre_post=False):
    """
    Builds a connection pattern according to a dense connectivity matrix.

    The matrix must be N*M, where N is the number of neurons in the post-synaptic population and M in the pre-synaptic one. Lists of lists must have the same size.

    If a synapse should not be created, the weight value should be None.

    :param weights: a matrix or list of lists representing the weights. If a value is None, the synapse will not be created.
    :param delays: a matrix or list of lists representing the delays. Must represent the same synapses as weights. If the argument is omitted, delays are 0.
    :param pre_post: states which index is first. By default, the first dimension is related to the post-synaptic population. If ``pre_post`` is True, the first dimension is the pre-synaptic population.
    """

    # Store the synapses
    self.connector_name = "Connectivity matrix"
    self.connector_description = "Connectivity matrix"

    if isinstance(weights, list):
        try:
            weights= np.array(weights)
        except:
            Global._error('connect_from_matrix(): You must provide a dense 2D matrix.')

    self._store_connectivity(self._load_from_matrix, (weights, delays, pre_post), delays)

    return self

def _load_from_matrix(self, pre, post, weights, delays, pre_post):
    lil = LILConnectivity()

    uniform_delay = not isinstance(delays, (list, np.ndarray))
    if isinstance(delays, list):
        try:
            delays= np.array(delays)
        except:
            Global._error('connect_from_matrix(): You must provide a dense 2D matrix.')

    if pre_post: # if the user prefers pre as the first index...
        weights = weights.T
        if isinstance(delays, np.ndarray):
            delays = delays.T

    shape = weights.shape
    if shape != (self.post.size, self.pre.size):
        if not pre_post:
            Global._print("ERROR: connect_from_matrix(): the matrix does not have the correct dimensions.")
            Global._print('Expected:', (self.post.size, self.pre.size))
            Global._print('Received:', shape)

        else:
            Global._print("ERROR: connect_from_matrix(): the matrix does not have the correct dimensions.")
            Global._print('Expected:', (self.pre.size, self.post.size))
            Global._print('Received:', shape)
        Global._error('Quitting...')

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
                    d.append(delays[i,j])
        if uniform_delay:
            d.append(delays)
        if len(r) > 0:
            lil.add(rk_post, r, w, d)

    return lil

def connect_from_sparse(self, weights, delays=0.0):
    """
    Builds a connectivity pattern using a Scipy sparse matrix for the weights and (optionally) delays.

    Warning: a sparse matrix has pre-synaptic ranks as first dimension.

    :param weights: a sparse lil_matrix object created from scipy.
    :param delays: the value of the constant delay (default: dt).
    """
    try:
        from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
    except:
        Global._error("connect_from_sparse(): scipy is not installed, sparse matrices can not be loaded.")

    if not isinstance(weights, (lil_matrix, csr_matrix, csc_matrix)):
        Global._error("connect_from_sparse(): only lil, csr and csc matrices are allowed for now.")

    if not isinstance(delays, (int, float)):
        Global._error("connect_from_sparse(): only constant delays are allowed for sparse matrices.")

    weights = csc_matrix(weights)

    # if weights[weights.nonzero()].max() == weights[weights.nonzero()].min() :
    #     self._single_constant_weight = True

    # Store the synapses
    self.connector_name = "Sparse connectivity matrix"
    self.connector_description = "Sparse connectivity matrix"
    self._store_connectivity(self._load_from_sparse, (weights, delays), delays)

    return self

def _load_from_sparse(self, pre, post, weights, delays):
    # Create an empty LIL object
    lil = LILConnectivity()

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
        Global._print("ERROR: connect_from_sparse(): the sparse matrix does not have the correct dimensions.")
        Global._print('Expected:', (len(pre_ranks), len(post_ranks)))
        Global._print('Received:', (pre, post))
        Global._error('Quitting...')


    for idx_post in range(post):
        idx_pre = weights.getcol(idx_post).indices
        w = weights.getcol(idx_post).data
        pr = [pre_ranks[i] for i in idx_pre]
        lil.add(post_ranks[idx_post], pr, w, [float(delays)])

    return lil

def connect_from_file(self, filename):
    """
    Builds the connectivity matrix using data saved using the Projection.save_connectivity() method (not save()!).

    Admissible file formats are compressed Numpy files (.npz), gunzipped binary text files (.gz) or binary text files.

    :param filename: file where the connections were saved.

    .. note::

        Only the ranks, weights and delays are loaded, not the other variables.
    """
    # Create an empty LIL object
    lil = LILConnectivity()

    # Load the data
    from ANNarchy.core.IO import _load_connectivity_data
    try:
        data = _load_connectivity_data(filename)
    except Exception as e:
        Global._print(e)
        Global._error('connect_from_file(): Unable to load the data', filename, 'into the projection.')

    # Load the LIL object
    try:
        # Size
        lil.size = data['size']
        lil.nb_synapses = data['nb_synapses']

        # Ranks
        lil.post_rank = list(data['post_ranks'])
        lil.pre_rank = list(data['pre_ranks'])

        # Weights
        if isinstance(data['w'], (int, float)):
            self._single_constant_weight = True
            lil.w = [[float(data['w'])]]
        elif isinstance(data['w'], (np.ndarray,)) and data['w'].size == 1:
            self._single_constant_weight = True
            lil.w = [[float(data['w'])]]
        else:
            lil.w = data['w']

        # Delays
        if data['delay']:
            lil.delay = data['delay']
        lil.max_delay = data['max_delay']
        lil.uniform_delay = data['uniform_delay']

    except Exception as e:
        Global._print(e)
        Global._error('Unable to load the data', filename, 'into the projection.')

    # Store the synapses
    self.connector_name = "From File"
    self.connector_description = "From File"
    self._store_connectivity(self._load_from_lil, (lil,), lil.max_delay if lil.uniform_delay > 0 else lil.delay)

    return self
