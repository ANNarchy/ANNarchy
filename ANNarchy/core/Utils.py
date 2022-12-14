#===============================================================================
#
#     Utils.py
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
import ANNarchy.core.Global as Global

######################
# Sparse matrices
######################

def sparse_random_matrix(pre, post, p, weight, delay=0):
    """
    Returns a sparse (lil) matrix to connect the pre and post populations with the probability p and the value weight.
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Global._warning("scipy is not installed, sparse matrices won't work")
        return None
    from random import sample
    W=lil_matrix((pre, post))
    for i in range(pre):
        k=np.random.binomial(post,p,1)[0]
        W.rows[i]=sample(range(post),k)
        W.data[i]=[weight]*k

    return W

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
        if neur_rank in proj.pre.ranks:
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
