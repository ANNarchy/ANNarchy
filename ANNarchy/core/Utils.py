"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.intern import Messages


from functools import wraps
import time


######################
# Time measurement
######################
def timeit(func):
    """
    Decorator to measure the execution time of a method.

    ```python
    @ann.timeit
    def run(net, T):
        net.simulate(T)
        return net.m.get()

    net = ann.Network()
    data = run(net, 1000)
    ```
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

######################
# Sparse matrices
######################

def sparse_random_matrix(pre: "Population", post: "Population", proba:float, weights:float|RandomDistribution) -> "scipy.sparse.lil_matrix":
    """
    Returns a sparse lil-matrix for use in `Projection.from_sparse()`.
    
    Creates a `scipy.sparse.lil_matrix` connectivity matrix that connects the `pre` and `post` populations with the probability `p` and the value `weight`, which can be either a constant or an ANNarchy random distribution object.

    ```python
    pop1 = net.create(100, neuron)
    pop2 = net.create(100, neuron)
    proj = net.connect(pre=pop1, post=pop2, target='exc')

    matrix = sparse_random_matrix(pre=pop1, post=pop2, p=0.1, w=ann.Uniform(0.0, 1.0))
    proj.from_sparse(matrix)
    ```

    :param pre: pre-synaptic population.
    :param post: post-synaptic population.
    :param proba: connection probability.
    :param weights: weight values (constant or random).
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Messages._warning("scipy is not installed, sparse matrices won't work")
        return None
    
    from random import sample
    W = lil_matrix((pre.size, post.size))
    
    for i in range(pre.size):
        k=np.random.binomial(post.size, proba,1)[0]
        tmp = sample(range(post.size),k)
        W.rows[i]=list(np.sort(tmp))
        
        if isinstance(weights, (int, float)):
            W.data[i] = [weights]*k
        
        elif isinstance(weights, RandomDistribution):
            W.data[i] = weights.get_list_values(k)
        
        else:
            raise ValueError("sparse_random_matrix expects either a float or RandomDistribution object.")

    return W

def sparse_delays_from_weights(weights: "scipy.sparse.lil_matrix", delays: float | RandomDistribution) -> "scipy.sparse.lil_matrix":
    """
    Returns a sparse delay matrix with the same connectivity as the sparse matrix `weight_matrix`.

    ```python
    matrix = sparse_random_matrix(pre=pop1, post=pop2, p=0.1, w=ann.Uniform(0.0, 1.0))
    delays = sparse_delays_from_weights(matrix, ann.Uniform(5.0, 10.0))
    proj.from_sparse(matrix, delays)
    ```

    :param weights: scipy sparse matrix to use for the connectivity.
    :param delays: delay value (constant or random).
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Messages._warning("scipy is not installed, sparse matrices won't work")
        return None

    delay_matrix = lil_matrix(weights.get_shape())

    (rows, cols) = weights.nonzero()

    for r, c in zip(rows, cols):
        if isinstance(delays, (int, float)):
            delay_matrix[r,c] = delays
        elif isinstance(delays, RandomDistribution):
            delay_matrix[r,c] = delays.get_value()
        else:
            raise ValueError("sparse_random_matrix expects either a float or RandomDistribution object.")

    return delay_matrix

################################
## Performance Measurement
################################

def compute_delivered_spikes(proj, pre_spike_events, post_spike_events):
    """
    This function counts the number of delivered spikes for a given Projection
    considering both pre- and post-synaptic events.

    :param proj: the Projection.
    :param pre_spike_events: the pre-synaptic spike events per time step (the result of a spike recording)
    :param post_spike_events: the post-synaptic spike events per time step (the result of a spike recording)
    """
    pre_throughput = compute_delivered_efferent_spikes(proj, pre_spike_events)
    post_throughput = compute_delivered_afferent_spikes(proj, post_spike_events)
    return pre_throughput + post_throughput

def compute_delivered_efferent_spikes(proj, spike_events):
    """
    This function counts the number of delivered spikes for a given Projection and
    spike sequence.

    :param proj: the Projection.
    :param spike_events: the spike events per time step (the result of a spike recording)
    """
    nb_efferent_synapses = proj._nb_efferent_synapses()
    delivered_events = 0

    for neur_rank, time_steps in spike_events.items():
        if neur_rank in proj.pre.ranks and neur_rank in nb_efferent_synapses.keys():
            delivered_events += nb_efferent_synapses[neur_rank] * len(time_steps)

    return delivered_events

def compute_delivered_afferent_spikes(proj, spike_events):
    """
    This function counts the number of delivered spikes for a given Projection and
    spike sequence.

    :param proj: the Projection.
    :param spike_events: the spike events per time step (the result of a spike recording)
    """
    delivered_events = 0

    for neur_rank, time_steps in spike_events.items():
        if neur_rank in proj.post.ranks:
            delivered_events += proj.dendrite(neur_rank).size * len(time_steps)

    return delivered_events

def compute_delivered_spikes_per_second(proj, spike_events, time_in_seconds, scale_factor=(10**6)):
    """
    This function implements a throughput metric for spiking neural networks.

    :param proj: the Projection
    :param spike_events: the spike events per time step (the result of a spike recording)
    :param time_in_seconds: computation time used for the operation
    :param scale_factor: usually the throughput value gets quite large, therefore its useful to rescale the value. By default, we re-scale to millions of events per second.

    **Note**:

    In the present form the function should be used only if a single projection is recorded. If multiple Projections contribute to the
    measured time (*time_in_seconds*) you need to compute the value individually for each Projection using the *compute_delivered_spikes*
    method and the times for instance retrieved from a run using --profile.
    """
    num_events = compute_delivered_spikes(proj, spike_events)

    return (num_events / time_in_seconds) / scale_factor
