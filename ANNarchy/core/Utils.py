"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.intern import Messages

######################
# Sparse matrices
######################

def sparse_random_matrix(pre, post, p, weight):
    """
    Returns a sparse (lil) matrix to connect the pre and post populations with the
    probability *p* and the value *weight*, either a constant or an ANNarchy random
    distribution object.
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Messages._warning("scipy is not installed, sparse matrices won't work")
        return None
    from random import sample
    W=lil_matrix((pre, post))
    for i in range(pre):
        k=np.random.binomial(post,p,1)[0]
        tmp = sample(range(post),k)
        W.rows[i]=list(np.sort(tmp))
        if isinstance(weight, (int, float)):
            W.data[i]=[weight]*k
        elif isinstance(weight, RandomDistribution):
            W.data[i]=weight.get_list_values(k)
        else:
            raise ValueError("sparse_random_matrix expects either a float or RandomDistribution object.")

    return W

def sparse_delays_from_weights(weight_matrix, delay):
    """
    Generates a delay matrix corresponding to the connectivity stored *weight_matrix*.
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Messages._warning("scipy is not installed, sparse matrices won't work")
        return None

    delay_matrix = lil_matrix(weight_matrix.get_shape())

    (rows,cols) = weight_matrix.nonzero()

    for r, c in zip(rows, cols):
        if isinstance(delay, (int, float)):
            delay_matrix[r,c] = delay
        elif isinstance(delay, RandomDistribution):
            delay_matrix[r,c] = delay.get_value()
        else:
            raise ValueError("sparse_random_matrix expects either a float or RandomDistribution object.")

    return delay_matrix

################################
## Performance Measurment
################################

def compute_delivered_spikes(proj, spike_events):
    """
    This function counts the number of delivered spikes for a given Projection and
    spike sequence.

    *Params*:

    * proj:             the Projection
    * spike_events:     the spike events per time step (the result of a spike recording)
    """
    nb_efferent_synapses = proj.nb_efferent_synapses()
    delivered_events = 0

    for neur_rank, time_steps in spike_events.items():
        if neur_rank in proj.pre.ranks and neur_rank in nb_efferent_synapses.keys():
            delivered_events += nb_efferent_synapses[neur_rank] * len(time_steps)

    return delivered_events

def compute_delivered_spikes_per_second(proj, spike_events, time_in_seconds, scale_factor=(10**6)):
    """
    This function implements a throughput metric for spiking neural networks.

    *Params*:

    * proj:             the Projection
    * spike_events:     the spike events per time step (the result of a spike recording)
    * time_in_seconds:  computation time used for the operation
    * scale_factor:     usually the throughput value gets quite large, therefore its useful to rescale the value. By default, we re-scale to millions of events per second.

    **Note**:

    In the present form the function should be used only if a single projection is recorded. If multiple Projections contribute to the
    measured time (*time_in_seconds*) you need to compute the value individually for each Projection using the *compute_delivered_spikes*
    method and the times for instance retrieved from a run using --profile.
    """
    num_events = compute_delivered_spikes(proj, spike_events)

    return (num_events / time_in_seconds) / scale_factor
