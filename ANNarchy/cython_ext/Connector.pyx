# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np

from libc.math cimport exp, fabs, ceil

import ANNarchy
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.core.Population import Population
from ANNarchy.intern.ConfigManagement import ConfigManager

cimport ANNarchy.cython_ext.Coordinates as Coordinates

def get_dt(obj):
    """
    Retrieve the *dt* configuration.

    Note:   the net_id is not directly available within the cython module.
            Therefore, we need to use the population object as interim step.
    """

    net_id = obj.net_id if isinstance(obj, Population) else obj.population.net_id

    return ConfigManager().get('dt', net_id=net_id)

##################################################
### Connector methods, these functions are    ####
### exported towards ConnectorMethods         ####
##################################################
def all_to_all(pre, post, weights, delays, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the all-to-all pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre))
    projection.all_to_all(pre, post, weights, delays, allow_self_connections)

    return projection

def one_to_one(pre, post, weights, delays, storage_format, storage_order):
    """ Cython implementation of the one-to-one pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre))
    projection.one_to_one(pre, post, weights, delays)

    return projection

def fixed_probability(pre, post, probability, weights, delays, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the fixed_probability pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre))
    projection.fixed_probability(pre, post, probability, weights, delays, allow_self_connections)

    return projection

def fixed_number_pre(pre, post, int number, weights, delays, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the fixed_number_pre pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre))
    projection.fixed_number_pre(pre, post, number, weights, delays, allow_self_connections)

    return projection

def fixed_number_post(pre, post, int number, weights, delays, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the fixed_number_post pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre))
    projection.fixed_number_post(pre, post, number, weights, delays, allow_self_connections)

    return projection

def gaussian(pre_pop, post_pop, float amp, float sigma, delays, limit, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the gaussian pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre_pop))
    projection.gaussian(pre_pop, post_pop, amp, sigma, delays, limit, allow_self_connections)

    return projection

def dog(pre_pop, post_pop, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections, storage_format, storage_order):
    """ Cython implementation of the difference-of-gaussian (dog) pattern."""
    # instantiate pattern
    projection = LILConnectivity(dt=get_dt(pre_pop))
    projection.dog(pre_pop, post_pop, amp_pos, sigma_pos, amp_neg, sigma_neg, delays, limit, allow_self_connections)

    return projection

###################################################
########## LIL object to hold synapses ############
###################################################
cdef class LILConnectivity:

    def __cinit__(self, dt):
        self.max_delay = 0
        self.size = 0
        self.nb_synapses = 0
        self.uniform_delay = -1
        self.dt = dt
        self.requires_sorting = False
        self.last_added_idx = -1

    def __dealloc__(self):
        self.post_rank.clear()
        self.post_rank.shrink_to_fit()
        self.pre_rank.clear()
        self.pre_rank.shrink_to_fit()
        self.w.clear()
        self.w.shrink_to_fit()
        self.delay.clear()
        self.delay.shrink_to_fit()
        # not sure, if really needed ...
        self.requires_sorting = False
        self.last_added_idx = -1

    cpdef add(self, int rk, r, w, d):
        self.push_back(rk, r, w, d)

    cpdef push_back(self, int rk, vector[int] r, vector[double] w, vector[double] d):
        cdef unsigned int i
        cdef vector[int] int_delays
        cdef int max_d, unif_d

        # sanity check: added rows should be ascending sorted
        if rk < self.last_added_idx:
            if self.requires_sorting == False:
                ANNarchy.intern.Messages._warning("LILConnectivity.add()/.push_back(): dendrites should be added in an ascending order for performance reasons.")
                ANNarchy.intern.Messages._print("ANNarchy will sort the dendrites during compile() which increases the required time.")
            self.requires_sorting = True
        else:
            self.last_added_idx = rk

        # Do not add empty arrays
        if r.size() == 0:
            return

        # Store the connectivity
        self.post_rank.push_back(rk)
        self.pre_rank.push_back(r)

        # Are the weights uniform?
        if w.size() > 1 or r.size() == 1:
            self.w.push_back(w)
        else:
            self.w.push_back(vector[double](r.size(), w[0]))

        # Store delays and update the max
        for i in range(d.size()):
            int_delays.push_back(round(d[i]/self.dt))
        self.delay.push_back(int_delays)
        max_d = round(np.max(d)/self.dt)
        if max_d > self.max_delay:
            self.max_delay = max_d

        # Are the delays uniform?
        if d.size() > 1 :
            self.uniform_delay = -1
        else:
            unif_d = round(d[0]/self.dt)
            if self.uniform_delay != unif_d and self.size > 0:
                self.uniform_delay = -1
            else:
                self.uniform_delay = unif_d

        # Increase the size
        self.size += 1
        self.nb_synapses += r.size()

    cpdef int get_max_delay(self):
        return self.max_delay

    cpdef int get_uniform_delay(self):
        return self.uniform_delay

    cpdef compute_average_row_length(self):
        cdef vector[int] rl
        for i in range(self.pre_rank.size()):
            rl.push_back(self.pre_rank[i].size())

        return np.mean(rl), np.std(rl), np.amin(rl), np.amax(rl)

    cpdef compute_average_col_idx_gap(self):
        cdef vector[int] idx_gap

        for i in range(self.pre_rank.size()):
            for j in range(self.pre_rank[i].size()-1):
                idx_gap.push_back(self.pre_rank[i][j+1]-self.pre_rank[i][j])

        return np.mean(idx_gap), np.std(idx_gap)

    cpdef validate(self):
        cdef int idx, single, rk
        cdef vector[int] ranks
        cdef vector[double] weights
        cdef vector[int] delays
        cdef dict doubletons = {}
        cdef list postranks = list(self.post_rank)
        cdef list set_postranks = list(set(postranks))
        cdef list preranks, set_preranks, indices

        if len(postranks) != len(set_postranks):
            ANNarchy.intern.Messages._warning('You have added several times the same post-synaptic neuron to the LIL data in your connector method.')
            ANNarchy.intern.Messages._print('ANNarchy will try to sort the entries if possible, it may take some time...')
        else:
            return

        # Find out which post neurons are doubled.
        for single in set_postranks:
            doubletons[single] = []
            for idx, possible_double in enumerate(postranks):
                if possible_double == single:
                    doubletons[single].append(idx)
            if len(doubletons[single]) == 1 : # Only one occurence
                doubletons.pop(single)

        # Gather the info
        for rk, indices in doubletons.items():
            # Store the different infos
            ranks.clear()
            weights.clear()
            delays.clear()
            for idx in reversed(indices): # Start from below, otherwise deletion crashes
                # Gather data
                ranks.insert(ranks.end(), self.pre_rank[idx].begin(), self.pre_rank[idx].end())
                weights.insert(weights.end(), self.w[idx].begin(), self.w[idx].end())
                delays.insert(delays.end(), self.delay[idx].begin(), self.delay[idx].end())
                # Delete the old vectors
                self.post_rank.erase(self.post_rank.begin()+idx)
                self.pre_rank.erase(self.pre_rank.begin()+idx)
                self.w.erase(self.w.begin()+idx)
                self.delay.erase(self.delay.begin()+idx)

            # Check if no synapse is doubled
            preranks = list(ranks)
            set_preranks = list(set(preranks))
            if len(preranks) != len(set_preranks):
                ANNarchy.intern.Messages._error('The same synapse has been declared multiple times! Check your code.', exit=True)

            # Add the new data
            self.post_rank.push_back(rk)
            self.pre_rank.push_back(ranks)
            self.w.push_back(weights)
            self.delay.push_back(delays)

    #####################################################
    # Connector method implementations for list-of-list #
    #####################################################
    cpdef all_to_all(self, pre, post, weights, delays, allow_self_connections):
        " Implementation of the all-to-all pattern "
        cdef double weight
        cdef int r_post, size_pre, i
        cdef list tmp, post_ranks, pre_ranks
        cdef vector[int] r
        cdef vector[double] w, d

        # Retríeve ranks
        post_ranks = post.ranks.tolist()
        pre_ranks = pre.ranks.tolist()

        for r_post in post_ranks:
            # List of pre ranks
            tmp = [i for i  in pre_ranks]
            if not allow_self_connections:
                try:
                    tmp.remove(r_post)
                except: # was not in the list
                    pass
            r = tmp
            size_pre = len(tmp)
            # Weights
            if isinstance(weights, (int, float)):
                weight = weights
                w = vector[double](1, weight)
            elif isinstance(weights, RandomDistribution):
                w = weights.get_list_values(size_pre)
            # Delays
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(size_pre)

            # Create the dendrite
            self.push_back(r_post, r, w, d)

    cpdef one_to_one(self, pre, post, weights, delays):
        """ Cython implementation of the one-to-one pattern."""
        cdef int idx
        cdef list post_ranks, pre_ranks
        cdef vector[int] r
        cdef vector[double] w, d

        # Retríeve ranks
        post_ranks = post.ranks.tolist()
        pre_ranks = pre.ranks.tolist()
    
        for idx in range(len(post_ranks)):
            if idx >= pre.size:
                break
            r = vector[int](1, pre_ranks[idx])
            # Get the weights and delays
            w, d = _get_weights_delays(1, weights, delays)
            # Create the dendrite
            self.push_back(post_ranks[idx], r, w, d)

    cpdef fixed_probability(self, pre, post, probability, weights, delays, allow_self_connections):
        " Implementation of the fixed-probability pattern "
        cdef double weight
        cdef int r_post, r_pre, size_pre, max_size_pre
        cdef list post_ranks
        cdef vector[int] r
        cdef vector[double] w, d
        cdef np.ndarray random_values, tmp, pre_ranks

        # Retríeve ranks
        post_ranks = post.ranks.tolist()
        pre_ranks = pre.ranks
        max_size_pre = len(pre.ranks)

        for r_post in post_ranks:
            # List of pre ranks
            random_values = np.random.random(max_size_pre)
            tmp = pre_ranks[random_values < probability]
            if not allow_self_connections:
                tmp = tmp[tmp != r_post]

            # sort the indices to prevent irregular accesses
            r = np.sort(tmp)
            size_pre = tmp.size
            if size_pre == 0:
                continue

            # Weights
            if isinstance(weights, (int, float)):
                weight = weights
                w = vector[double](1, weight)
            elif isinstance(weights, RandomDistribution):
                w = weights.get_list_values(size_pre)
            # Delays
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(size_pre)
            # Create the dendrite
            self.push_back(r_post, r, w, d)

    cpdef fixed_number_pre(self, pre, post, int number, weights, delays, allow_self_connections):
        cdef double weight
        cdef int r_post, r_pre, size_pre
        cdef list pre_ranks, post_ranks
        cdef vector[int] r
        cdef vector[double] w, d

        # Retríeve ranks
        post_ranks = post.ranks.tolist()
        pre_ranks = pre.ranks.tolist()

        for r_post in post_ranks:
            # List of pre ranks
            r = np.random.choice(pre_ranks, size=number, replace=False)
            if len(r) == 0:
                continue
            if not allow_self_connections:
                while r_post in list(r): # the post index is in the list
                    r = np.random.choice(pre_ranks, size=number, replace=False)

            # sort the indices to prevent irregular accesses
            r = np.sort(r)

            # Weights
            if isinstance(weights, (int, float)):
                weight = weights
                w = vector[double](1, weight)
            elif isinstance(weights, RandomDistribution):
                w = weights.get_list_values(number)
            # Delays
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(number)
            # Create the dendrite
            self.push_back(r_post, r, w, d)

    cpdef fixed_number_post(self, pre, post, int number, weights, delays, allow_self_connections):
        cdef double weight
        cdef int r_post, r_pre, size_pre
        cdef list pre_ranks, post_ranks
        cdef list pre_r
        cdef dict rk_mat
        cdef vector[int] r
        cdef vector[int] tmp
        cdef vector[double] w, d

        # Retrieve ranks
        post_ranks = post.ranks.tolist()
        pre_ranks = pre.ranks.tolist()

        # Build the backward matrix
        rk_mat = {i: [] for i in post_ranks}
        for r_pre in pre_ranks:
            if number >= len(post_ranks):
                tmp = post_ranks
            else:
                tmp = np.random.choice(post_ranks, size=number, replace=False)
                if not allow_self_connections:
                    # the post index is in the list, redraw
                    while r_pre in list(tmp):   # TODO: maybe a find() would be better
                        tmp = np.random.choice(post_ranks, size=number, replace=False)
            for i in tmp:
                rk_mat[i].append(r_pre)

        # Create the dendrites
        for r_post in post_ranks:
            # List of pre ranks
            r = rk_mat[r_post]
            size_pre = len(r)
            if size_pre == 0:
                continue
            # Weights
            if isinstance(weights, (int, float)):
                weight = weights
                w = vector[double](1, weight)
            elif isinstance(weights, RandomDistribution):
                w = weights.get_list_values(size_pre)
            # Delays
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(size_pre)
            # Create the dendrite
            self.push_back(r_post, r, w, d)

    cpdef gaussian(self, pre_pop, post_pop, float amp, float sigma, delays, limit, allow_self_connections):
        cdef float distance, value
        cdef int post, pre, pre_size, post_size, c, nb_synapses, pre_dim, post_dim
        cdef tuple pre_coord, post_coord
        cdef tuple pre_geometry, post_geometry
        cdef list ranks, values

        cdef vector[int] r
        cdef vector[double] w, d

        # Population sizes
        pre_geometry = pre_pop.geometry
        post_geometry = post_pop.geometry
        if isinstance(pre_geometry, int):
            pre_size = pre_geometry
            pre_dim = 1
        else:
            pre_dim = len(pre_geometry)
            pre_size = 1
            for c in pre_geometry:
                pre_size  = pre_size * c
        if isinstance(post_geometry, int):
            post_size = post_geometry
            post_dim = 1
        else:
            post_dim = len(post_geometry)
            post_size = 1
            for c in post_geometry:
                post_size  = post_size * c

        # Create the projection data
        for post in list(range(post_size)):
            ranks = []
            values = []
            if post_dim == 1:
                post_coord = (post/float(post_size-1), )
            elif post_dim == 2:
                post_coord = Coordinates.get_normalized_2d_coord(post, post_geometry)
            elif post_dim == 3:
                post_coord = Coordinates.get_normalized_3d_coord(post, post_geometry)
            else:
                post_coord = Coordinates.get_normalized_coord(post, post_geometry)
            for pre in list(range(pre_size)):
                if not allow_self_connections and pre==post:
                    continue
                if pre_dim == 1:
                    pre_coord = (pre/float(pre_size-1), )
                    distance = Coordinates.comp_dist1D(pre_coord, post_coord)
                elif pre_dim == 2:
                    pre_coord = Coordinates.get_normalized_2d_coord(pre, pre_geometry)
                    distance = Coordinates.comp_dist2D(pre_coord, post_coord)
                elif pre_dim == 3:
                    pre_coord = Coordinates.get_normalized_3d_coord(pre, pre_geometry)
                    distance = Coordinates.comp_dist3D(pre_coord, post_coord)
                else:
                    pre_coord = Coordinates.get_normalized_coord(pre, pre_geometry)
                    distance = Coordinates.comp_distND(pre_coord, post_coord)
                value = amp * exp(-distance/(2.0*sigma**2))
                if value > limit * amp:
                    ranks.append(pre)
                    values.append(value)
            nb_synapses = len(ranks)
            r = ranks
            w = values
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(nb_synapses)
            # Create the dendrite
            self.push_back(post, r, w, d)

    cpdef dog(self, pre_pop, post_pop, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections):
        cdef float distance, value
        cdef int post, pre, pre_size, post_size, c, nb_synapses, pre_dim, post_dim
        cdef tuple pre_coord, post_coord
        cdef tuple pre_geometry, post_geometry
        cdef list ranks, values

        cdef vector[int] r
        cdef vector[double] w, d

        # Population sizes
        pre_geometry = pre_pop.geometry
        post_geometry = post_pop.geometry
        if isinstance(pre_geometry, int):
            pre_size = pre_geometry
            pre_dim = 1
        else:
            pre_dim = len(pre_geometry)
            pre_size = 1
            for c in pre_geometry:
                pre_size  = pre_size * c
        if isinstance(post_geometry, int):
            post_size = post_geometry
            post_dim = 1
        else:
            post_dim = len(post_geometry)
            post_size = 1
            for c in post_geometry:
                post_size  = post_size * c

        # Create the projection data as LIL
        for post in list(range(post_size)):
            ranks = []
            values = []
            if post_dim == 1:
                post_coord = (post/float(post_size-1), )
            elif post_dim == 2:
                post_coord = Coordinates.get_normalized_2d_coord(post, post_geometry)
            elif post_dim == 3:
                post_coord = Coordinates.get_normalized_3d_coord(post, post_geometry)
            else:
                post_coord = Coordinates.get_normalized_coord(post, post_geometry)
            for pre in list(range(pre_size)):
                if not allow_self_connections and pre==post:
                    continue
                if pre_dim == 1:
                    pre_coord = (pre/float(pre_size-1), )
                    distance = Coordinates.comp_dist1D(pre_coord, post_coord)
                elif pre_dim == 2:
                    pre_coord = Coordinates.get_normalized_2d_coord(pre, pre_geometry)
                    distance = Coordinates.comp_dist2D(pre_coord, post_coord)
                elif pre_dim == 3:
                    pre_coord = Coordinates.get_normalized_3d_coord(pre, pre_geometry)
                    distance = Coordinates.comp_dist3D(pre_coord, post_coord)
                else:
                    pre_coord = Coordinates.get_normalized_coord(pre, pre_geometry)
                    distance = Coordinates.comp_distND(pre_coord, post_coord)
                value = amp_pos * exp(-distance/(2.0*sigma_pos**2)) - amp_neg * exp(-distance/(2.0*sigma_neg**2))
                if fabs(value) > limit * fabs(amp_pos - amp_neg):
                    ranks.append(pre)
                    values.append(value)
            nb_synapses = len(ranks)
            r = ranks
            w = values
            if isinstance(delays, (float, int)):
                d = vector[double](1, delays)
            elif isinstance(delays, RandomDistribution):
                d = delays.get_list_values(nb_synapses)

            # Create the dendrite
            self.push_back(post, r, w, d)

cdef _get_weights_delays(int size, weights, delays):

    cdef vector[double] w, d
    cdef list tmp

    # Weights
    if isinstance(weights, (int, float)):
        w = vector[double](1, weights)
    elif isinstance(weights, RandomDistribution):
        tmp = weights.get_list_values(size)
        w = tmp
    # Delays
    if isinstance(delays, (float, int)):
        d = vector[double](1, delays)
    elif isinstance(delays, RandomDistribution):
        d = delays.get_list_values(size)

    return w, d
