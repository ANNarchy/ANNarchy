# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np

import random

from libc.math cimport exp, fabs, ceil

import ANNarchy
from ANNarchy.core import Global
from ANNarchy.core.Random import RandomDistribution

cimport ANNarchy.core.cython_ext.Coordinates as Coordinates

###################################################
########## LIL object to hold synapses ############
###################################################

cdef class LIL:

    def __cinit__(self):

        self.max_delay = 0
        self.size = 0
        self.nb_synapses = 0
        self.uniform_delay = -1
        self.dt = Global.config['dt']

    cpdef add(self, int rk, r, w, d):
        self.push_back(rk, r, w, d)

    cpdef push_back(self, int rk, vector[int] r, vector[double] w, vector[double] d):
        cdef unsigned int i
        cdef vector[int] int_delays
        cdef int max_d, unif_d

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
            ANNarchy.core.Global._warning('You have added several times the same post-synaptic neuron to the LIL data in your connector method.')
            ANNarchy.core.Global._print('ANNarchy will try to sort the entries if possible, it may take some time...')
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
                ANNarchy.core.Global._error('The same synapse has been declared multiple times! Check your code.', exit=True)

            # Add the new data
            self.post_rank.push_back(rk)
            self.pre_rank.push_back(ranks)
            self.w.push_back(weights)
            self.delay.push_back(delays)


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

###################################################
########## CSRC object to hold synapses ###########
###################################################
cdef class CSRC:

    def __cinit__(self, pre_size, post_size):
        # pre -> post
        self._row_ptr = vector[int](pre_size+1, 0)
        self._col_idx = vector[int]()
        self._w = vector[double]()
        self._delay = vector[int]()

        # post -> pre
        self._col_ptr = vector[int](post_size+1, 0)
        self._row_idx = vector[int]()
        self._inv_idx = vector[int]()

    cpdef add(self, int pre_rank, post_rank, w, d):
        """
        Add a single connection to the CSRC data structure.
        """
        # update vectors
        self._col_idx.insert(self._col_idx.begin()+pre_rank+1, post_rank)
        self._w.insert(self._w.begin()+pre_rank+1, w)
        self._delay.insert(self._delay.begin()+pre_rank+1, d)

        # update row borders
        for i in range(pre_rank+1, self._row_ptr.size()):
            self._row_ptr[i] += 1

    cpdef push_back(self, int pre_rank, vector[int] post_ranks, vector[double] w, vector[double] d):
        """
        Add a set of connections to the CSRC data structure.
        """
        # update vectors
        self._col_idx.insert(self._col_idx.begin()+pre_rank+1, post_ranks.begin(), post_ranks.end())
        self._w.insert(self._w.begin()+pre_rank+1, w.begin(), w.end())
        self._delay.insert(self._delay.begin()+pre_rank+1, d.begin(), d.end())

        # update row borders
        for i in range(pre_rank+1, self._row_ptr.size()):
            self._row_ptr[i] += post_ranks.size()

    cpdef inverse_connectivity(self):
        """
        Generate the backward view
        """
        pass

#################################
#### Connector methods ##########
#################################

def all_to_all(pre, post, weights, delays, allow_self_connections):
    """ Cython implementation of the all-to-all pattern."""

    cdef LIL projection
    cdef double weight
    cdef int r_post, size_pre, i
    cdef list tmp, post_ranks, pre_ranks
    cdef vector[int] r
    cdef vector[double] w, d

    # Retríeve ranks
    post_ranks = post.ranks
    pre_ranks = pre.ranks

    # Create the projection data as LIL
    projection = LIL()

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
        projection.push_back(r_post, r, w, d)

    return projection

def all_to_all_csrc(pre, post, weights, delays, allow_self_connections):
    """ Cython implementation of the all-to-all pattern, stored as CSRC and pre1st ordering. """

    cdef CSRC projection
    cdef double weight
    cdef int r_post, size_pre, i
    cdef list tmp, post_ranks, pre_ranks
    cdef vector[int] r
    cdef vector[double] w, d

    # Retríeve ranks
    post_ranks = post.ranks
    pre_ranks = pre.ranks

    # Create the projection data as CSRC
    projection = CSRC(pre.size, post.size)

    for r_pre in pre_ranks:
        # List of post ranks
        tmp = [i for i in post_ranks]
        if not allow_self_connections:
            try:
                tmp.remove(r_pre)
            except: # was not in the list
                pass
        r = tmp
        size_post = len(tmp)
        # Weights
        if isinstance(weights, (int, float)):
            weight = weights
            w = vector[double](1, weight)
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_post)
        # Delays
        if isinstance(delays, (float, int)):
            d = vector[double](1, delays)
        elif isinstance(delays, RandomDistribution):
            d = delays.get_list_values(size_post)
        # Create the dendrite
        projection.push_back(r_pre, r, w, d)

    return projection

def one_to_one(pre, post, weights, delays, shift):
    """ Cython implementation of the one-to-one pattern."""

    cdef LIL projection
    cdef double weight
    cdef int r_post, offset
    cdef list tmp, post_ranks, pre_ranks
    cdef vector[int] r
    cdef vector[double] w, d

    # Create the projection data as LIL
    projection = LIL()

    # Retríeve ranks
    post_ranks = post.ranks

    if shift:
        pre_ranks = pre.ranks
        offset = min(post_ranks) - min(pre_ranks)
    else:
        offset = 0


    if shift:
        for r_post in post_ranks:
            # List of pre ranks
            if not r_post - offset in pre_ranks:
                continue
            r = vector[int](1, r_post - offset)
            # Get the weights and delays
            w, d = _get_weights_delays(1, weights, delays)
            # Create the dendrite
            projection.push_back(r_post, r, w, d)

    else:
        for r_post in post_ranks:
            if r_post >= pre.size:
                break
            # List of pre ranks
            r = vector[int](1, r_post)
            # Get the weights and delays
            w, d = _get_weights_delays(1, weights, delays)
            # Create the dendrite
            projection.push_back(r_post, r, w, d)



    return projection



def fixed_probability(pre, post, probability, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_probability pattern."""

    cdef LIL projection
    cdef double weight
    cdef int r_post, r_pre, size_pre, max_size_pre
    cdef list post_ranks
    cdef vector[int] r
    cdef vector[double] w, d
    cdef np.ndarray random_values, tmp, pre_ranks

    # Retríeve ranks
    post_ranks = post.ranks

    pre_ranks = np.array(pre.ranks)
    max_size_pre = len(pre.ranks)

    # Create the projection data as LIL
    projection = LIL()

    for r_post in post_ranks:
        # List of pre ranks
        random_values = np.random.random(max_size_pre)
        tmp = pre_ranks[random_values < probability]
        if not allow_self_connections:
            tmp = tmp[tmp != r_post]
        r = tmp
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
        projection.push_back(r_post, r, w, d)

    return projection

def fixed_number_pre(pre, post, int number, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_number_pre pattern."""

    cdef LIL projection
    cdef double weight
    cdef int r_post, r_pre, size_pre
    cdef list pre_ranks, post_ranks
    cdef vector[int] r
    cdef vector[double] w, d

    # Retríeve ranks
    post_ranks = post.ranks
    pre_ranks = pre.ranks

    # Create the projection data as LIL
    projection = LIL()

    for r_post in post_ranks:
        # List of pre ranks
        r = random.sample(pre_ranks, number)
        if len(r) == 0:
            continue
        if not allow_self_connections:
            while r_post in list(r): # the post index is in the list
                r = random.sample(pre_ranks, number)
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
        projection.push_back(r_post, r, w, d)

    return projection

def fixed_number_post(pre, post, int number, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef LIL projection
    cdef double weight
    cdef int r_post, r_pre, size_pre
    cdef list pre_ranks, post_ranks
    cdef list pre_r, tmp
    cdef dict rk_mat
    cdef vector[int] r
    cdef vector[double] w, d

    # Retríeve ranks
    post_ranks = post.ranks
    pre_ranks = pre.ranks

    # Create the projection data as LIL
    projection = LIL()


    # Build the backward matrix
    rk_mat = {i: [] for i in post_ranks}
    for r_pre in pre_ranks:
        if number >= len(post_ranks):
            tmp = post_ranks
        else:
            tmp = random.sample(post_ranks, number)
            if not allow_self_connections:
                while r_pre in tmp: # the post index is in the list
                    tmp = random.sample(post_ranks, number)
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
        projection.push_back(r_post, r, w, d)

    return projection

def gaussian(pre_pop, post_pop, float amp, float sigma, delays, limit, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef LIL projection
    cdef float distance, value
    cdef int post, pre, pre_size, post_size, c, nb_synapses, pre_dim, post_dim
    cdef tuple pre_coord, post_coord
    cdef tuple pre_geometry, post_geometry
    cdef list ranks, values

    cdef vector[int] r
    cdef vector[double] w, d


    # Retrieve simulation time step
    dt = Global.config['dt']

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
    projection = LIL()
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
        projection.push_back(post, r, w, d)

    return projection

def dog(pre_pop, post_pop, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef LIL projection
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
    projection = LIL()
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
        projection.push_back(post, r, w, d)

    return projection
