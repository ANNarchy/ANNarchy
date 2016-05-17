"""

    Connectors.py

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
from ANNarchy.core.Random import RandomDistribution, Uniform
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.Dendrite import Dendrite
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.parser.Report import _process_random

try:
    import ANNarchy.core.cython_ext.Connector as Connector
except Exception as e:
    Global._print(e)


################################
## Connector methods
################################

def connect_one_to_one(self, weights=1.0, delays=0.0, shift=None, force_multiple_weights=False):
    """
    Builds a one-to-one connection pattern between the two populations.

    *Parameters*:

        * **weights**: initial synaptic values, either a single value (float) or a random distribution object.
        * **delays**: synaptic delays, either a single value or a random distribution object (default=dt).
        * **shift**: specifies if the ranks of the presynaptic population should be shifted to match the start of the post-synaptic population ranks. Particularly useful for PopulationViews. Does not work yet for populations with geometry. Default: if the two populations have the same number of neurons, it is set to True. If not, it is set to False (only the ranks count).
        * **force_multiple_weights**: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
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

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity(Connector.one_to_one, (weights, delays, shift), delays)
    return self

def connect_all_to_all(self, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False):
    """
    Builds an all-to-all connection pattern between the two populations.

    *Parameters*:

        * **weights**: synaptic values, either a single value or a random distribution object.
        * **delays**: synaptic delays, either a single value or a random distribution object (default=dt).
        * **allow_self_connections**: if True, self-connections between a neuron and itself are allowed (default = False if the pre- and post-populations are identical, True otherwise).
        * **force_multiple_weights**: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    self.connector_name = "All-to-All"
    self.connector_description = "All-to-All, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays)}

    # Does the projection define a single non-plastic weight?
    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    # Is it a dense connectivity matrix?
    if allow_self_connections and not isinstance(self.pre, PopulationView) and not isinstance(self.post, PopulationView):
        self._dense_matrix = True

    # Store the connectivity
    self._store_connectivity(Connector.all_to_all, (weights, delays, allow_self_connections), delays)
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
        Global._error('Gaussian connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Gaussian"
    self.connector_description = "Gaussian, $A$ %(A)s, $\sigma$ %(sigma)s, delays %(delay)s"% {'A': str(amp), 'sigma': str(sigma), 'delay': _process_random(delays)}

    self._store_connectivity(Connector.gaussian, (amp, sigma, delays, limit, allow_self_connections), delays)
    return self

def connect_dog(self, amp_pos, sigma_pos, amp_neg, sigma_neg, delays=0.0, limit=0.01, allow_self_connections=False):
    """
    Builds a Difference-Of-Gaussians connection pattern between the two populations.

    Each neuron in the postsynaptic population is connected to a region of the presynaptic population centered around
    the neuron with the same normalized coordinates using a Difference-Of-Gaussians profile.

    *Parameters*:

        * **amp_pos**: amplitude of the positive Gaussian function
        * **sigma_pos**: width of the positive Gaussian function
        * **amp_neg**: amplitude of the negative Gaussian function
        * **sigma_neg**: width of the negative Gaussian function
        * **delays**: synaptic delay, either a single value or a random distribution object (default=dt).
        * **limit**: proportion of *amp* below which synapses are not created (default: 0.01)
        * **allow_self_connections**: allows connections between a neuron and itself.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if isinstance(self.pre, PopulationView) or isinstance(self.post, PopulationView):
        Global._error('DoG connector is only possible on whole populations, not PopulationViews.')

    self.connector_name = "Difference-of-Gaussian"
    self.connector_description = "Difference-of-Gaussian, $A^+ %(Aplus)s, $\sigma^+$ %(sigmaplus)s, $A^- %(Aminus)s, $\sigma^-$ %(sigmaminus)s, delays %(delay)s"% {'Aplus': str(amp_pos), 'sigmaplus': str(sigma_pos), 'Aminus': str(amp_neg), 'sigmaminus': str(sigma_neg), 'delay': _process_random(delays)}

    self._store_connectivity(Connector.dog, (amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections), delays)
    return self

def connect_fixed_probability(self, probability, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False):
    """
    Builds a probabilistic connection pattern between the two populations.

    Each neuron in the postsynaptic population is connected to neurons of the presynaptic population with the given probability. Self-connections are avoided by default.

    *Parameters*:

        * **probability**: probability that a synapse is created.
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        * **allow_self_connections** : defines if self-connections are allowed (default=False).
        * **force_multiple_weights**: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    self.connector_name = "Random"
    self.connector_description = "Random, sparseness %(proba)s, weights %(weight)s, delays %(delay)s" % {'weight': _process_random(weights), 'delay': _process_random(delays), 'proba': probability}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity(Connector.fixed_probability, (probability, weights, delays, allow_self_connections), delays)
    return self

def connect_fixed_number_pre(self, number, weights, delays=0.0, allow_self_connections=False, force_multiple_weights=False):
    """
    Builds a connection pattern between the two populations with a fixed number of pre-synaptic neurons.

    Each neuron in the postsynaptic population receives connections from a fixed number of neurons of the presynaptic population chosen randomly.

    *Parameters*:

        * **number**: number of synapses per postsynaptic neuron.
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        * **allow_self_connections** : defines if self-connections are allowed (default=False).
        * **force_multiple_weights**: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if number > self.pre.size:
        Global._error('connect_fixed_number_pre: the number of pre-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Convergent"
    self.connector_description = "Random Convergent %(number)s $\\rightarrow$ 1, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity(Connector.fixed_number_pre, (number, weights, delays, allow_self_connections), delays)

    return self

def connect_fixed_number_post(self, number, weights=1.0, delays=0.0, allow_self_connections=False, force_multiple_weights=False):
    """
    Builds a connection pattern between the two populations with a fixed number of post-synaptic neurons.

    Each neuron in the pre-synaptic population sends connections to a fixed number of neurons of the post-synaptic population chosen randomly.

    *Parameters*:

        * **number**: number of synapses per pre-synaptic neuron.
        * **weights**: either a single value for all synapses or a RandomDistribution object.
        * **delays**: either a single value for all synapses or a RandomDistribution object (default = dt)
        * **allow_self_connections** : defines if self-connections are allowed (default=False)
        * **force_multiple_weights**: if a single value is provided for ``weights`` and there is no learning, a single weight value will be used for the whole projection instead of one per synapse. Setting ``force_multiple_weights`` to True ensures that a value per synapse will be used.
    """
    if self.pre!=self.post:
        allow_self_connections = True

    if number > self.pre.size:
        Global._error('connect_fixed_number_post: the number of post-synaptic neurons exceeds the size of the population.')

    self.connector_name = "Random Divergent"
    self.connector_description = "Random Divergent 1 $\\rightarrow$ %(number)s, weights %(weight)s, delays %(delay)s"% {'weight': _process_random(weights), 'delay': _process_random(delays), 'number': number}

    if isinstance(weights, (int, float)) and not force_multiple_weights:
        self._single_constant_weight = True

    self._store_connectivity(Connector.fixed_number_post, (number, weights, delays, allow_self_connections), delays)
    return self

def connect_with_func(self, method, **args):
    """
    Builds a connection pattern based on a user-defined method.

    *Parameters*:

    * **method**: method to call. The method **must** return a CSR object.
    * **args**: list of arguments needed by the function
    """
    # Invoke the method directly, we need the delays already....
    synapses = method(self.pre, self.post, **args)
    synapses.validate()

    # Treat delays
    if synapses.uniform_delay != -1: # uniform delay
        d = synapses.max_delay * Global.config['dt']
    else:
        d = Uniform(0., synapses.max_delay * Global.config['dt']) # Just to trick _store_connectivity(), the real delays are in the CSR

    self._store_connectivity(self._load_from_csr, (synapses, ), d)

    self.connector_name = "User-defined"
    self.connector_description = "Created by the method " + method.__name__

    return self

def _load_from_csr(self, pre, post, synapses):
    return synapses

def connect_from_matrix(self, weights, delays=0.0, pre_post=False):
    """
    Builds a connection pattern according to a dense connectivity matrix.

    The matrix must be N*M, where N is the number of neurons in the post-synaptic population and M in the pre-synaptic one. Lists of lists must have the same size.

    If a synapse should not be created, the weight value should be None.

    *Parameters*:

    * **weights**: a matrix or list of lists representing the weights. If a value is None, the synapse will not be created.
    * **delays**: a matrix or list of lists representing the delays. Must represent the same synapses as weights. If the argument is omitted, delays are 0.
    * **pre_post**: states which index is first. By default, the first dimension is related to the post-synaptic population. If ``pre_post`` is True, the first dimension is the pre-synaptic population.
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
    csr = Connector.CSR()

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
            csr.add(rk_post, r, w, d)

    return csr

def connect_from_sparse(self, weights, delays=0.0):
    """
    Builds a connectivity pattern using a Scipy sparse matrix for the weights and (optionally) delays.

    Warning: a sparse matrix has pre-synaptic ranks as first dimension.

    *Parameters*:

    * **weights**: a sparse lil_matrix object created from scipy.
    * **delays**: the value of the constant delay (default: dt).
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
    from scipy.sparse import csc_matrix

    # Create an empty CSR object
    csr = Connector.CSR()

    # Find offsets
    if isinstance(self.pre, PopulationView):
        pre_ranks = self.pre.ranks
    else:
        pre_ranks = [i for i in range(self.pre.size)]

    if isinstance(self.post, PopulationView):
        post_ranks = self.post.ranks
    else:
        post_ranks = [i for i in range(self.post.size)]

    # Process the sparse matrix and fill the csr
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
        csr.add(post_ranks[idx_post], pr, w, [float(delays)])

    return csr

def connect_from_file(self, filename):
    """
    Builds a connection pattern using data saved using the Projection.save_connectivity() method (not save()!).

    *Parameters*:

    * **filename**: file where the data was saved.

    .. note::

        Only the ranks, weights and delays are loaded, not the other variables.
    """
    # Create an empty CSR object
    csr = Connector.CSR()

    # Load the data
    from ANNarchy.core.IO import _load_data
    try:
        data = _load_data(filename)
    except Exception as e:
        Global._print(e)
        Global._error('connect_from_file(): Unable to load the data', filename, 'into the projection.')

    # Load the CSR object
    try:
        csr.post_rank = data['post_ranks']
        csr.pre_rank = data['pre_ranks']
        if isinstance(data['w'], (int, float)):
            self._single_constant_weight = True
            csr.w = [[data['w']]]
        else:
            csr.w = data['w']
        csr.size = data['size']
        csr.nb_synapses = data['nb_synapses']
        if data['delay']:
            csr.delay = data['delay']
        csr.max_delay = data['max_delay']
        csr.uniform_delay = data['uniform_delay']
    except Exception as e:
        Global._print(e)
        Global._error('Unable to load the data', filename, 'into the projection.')

    # Store the synapses
    self.connector_name = "From File"
    self.connector_description = "From File"
    self._store_connectivity(self._load_from_csr, (csr,), csr.max_delay if csr.uniform_delay > 0 else csr.delay)
    return self
