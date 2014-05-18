# distutils: language = c++

from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from libc.math cimport exp, fabs

import ANNarchy
from ANNarchy.core.Random import RandomDistribution

cimport ANNarchy.core.cython_ext.Coordinates as Coordinates

cdef class CSR:

    def __init__(self):
        self.data = {}

    cdef add (self, int rk, vector[int] r, vector[float] w, vector[int] d):
        cdef list val
        val = []
        val.append(r)
        val.append(w)
        val.append(d)
        self.data[rk] = val

    def keys(self):
        return self.data.keys()

    cpdef get_data(self):
        return self.data

def all_to_all(int pre_size, int post_size, weights, delays, allow_self_connections):
    """ Cython implementation of the all-to-all pattern."""

    cdef CSR projection
    cdef float dt
    cdef int post, pre, size_pre
    cdef list tmp
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Create the projection data as CSR
    projection = CSR()

    # Determine the size of the the pre-field 
    # (the passed argument allow_self_connections already knows if it is the same population or not)
    if allow_self_connections:
        size_pre = pre_size
    else:
        size_pre = pre_size -1 

    for post in xrange(post_size):
        # List of pre ranks
        tmp = [i for i in xrange(pre_size)]
        if not allow_self_connections:
            tmp.remove(post)
        r = tmp
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](size_pre, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_pre)
        # Delays
        if isinstance(delays, float):
            d = vector[int](size_pre, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](size_pre, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(size_pre) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection



def fixed_probability(int pre_size, int post_size, float probability, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_probability pattern."""

    cdef CSR projection
    cdef float dt
    cdef int post, pre, size_pre
    cdef list tmp
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Create the projection data as CSR
    projection = CSR()

    for post in xrange(post_size):
        # List of pre ranks
        tmp = []
        for i in xrange(pre_size):
            if not allow_self_connections and (i==post):
                continue
            if np.random.random() < probability:
                tmp .append(i) 
        r = tmp
        size_pre = len(tmp)
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](size_pre, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_pre)
        # Delays
        if isinstance(delays, float):
            d = vector[int](size_pre, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](size_pre, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(size_pre) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection

def fixed_number_pre(int pre_size, int post_size, int number, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_number_pre pattern."""

    cdef CSR projection
    cdef float dt
    cdef int post, pre, size_pre
    cdef np.ndarray indices, tmp
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']
    
    # Create the projection data as CSR
    projection = CSR()
    
    # Array to permute
    indices = np.array(range(0, pre_size))

    for post in xrange(post_size):
        # List of pre ranks
        indices = np.random.permutation(indices)
        tmp = indices[:number]
        if not allow_self_connections:
            if (tmp == post).any(): # the post index is in the list
                tmp[tmp==post] = indices[number]
        r = list(tmp)
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](number, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(number)
        # Delays
        if isinstance(delays, float):
            d = vector[int](number, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](number, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(number) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection

def fixed_number_post(int pre_size, int post_size, int number, weights, delays, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef CSR projection
    cdef float dt
    cdef int post, pre, size_pre
    cdef np.ndarray indices, tmp
    cdef list rk_mat, pre_r
    cdef vector[int] r, d
    cdef vector[float] w

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']
    
    # Create the projection data as CSR
    projection = CSR()
    
    # Array to permute
    indices = np.array(range(0, post_size))

    # Build the backward matrix
    rk_mat = [ [] for i in xrange(post_size)]
    for pre in xrange(pre_size):
        indices = np.random.permutation(indices)
        tmp = indices[:number]
        if not allow_self_connections:
            if (tmp == pre).any(): # the post index is in the list
                tmp[tmp==pre] = indices[number] # pick the next one
        for i in xrange(number):
            rk_mat[tmp[i]].append(pre)

    # Create the dendrites
    for post in xrange(post_size):
        # List of pre ranks
        r = rk_mat[post]
        size_pre = len(rk_mat[post])
        # Weights
        if isinstance(weights, (int, float)):
            w = vector[float](size_pre, float(weights))
        elif isinstance(weights, RandomDistribution):
            w = weights.get_list_values(size_pre)
        # Delays
        if isinstance(delays, float):
            d = vector[int](size_pre, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](size_pre, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(size_pre) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection

def gaussian(tuple pre_geometry, tuple post_geometry, float amp, float sigma, delays, limit, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef CSR projection
    cdef float dt, distance, value
    cdef int post, pre, pre_size, post_size, c, nb_synapses, pre_dim, post_dim
    cdef tuple pre_coord, post_coord
    cdef list ranks, values

    cdef vector[int] r, d
    cdef vector[float] w


    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Population sizes
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
    
    # Create the projection data as CSR
    projection = CSR()
    for post in xrange(post_size):
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
        for pre in xrange(pre_size):
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
        if isinstance(delays, float):
            d = vector[int](nb_synapses, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](nb_synapses, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(nb_synapses) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection

def dog(tuple pre_geometry, tuple post_geometry, float amp_pos, float sigma_pos, float amp_neg, float sigma_neg, delays, limit, allow_self_connections):
    """ Cython implementation of the fixed_number_post pattern."""

    cdef CSR projection
    cdef float dt, distance, value
    cdef int post, pre, pre_size, post_size, c, nb_synapses, pre_dim, post_dim
    cdef tuple pre_coord, post_coord
    cdef list ranks, values

    cdef vector[int] r, d
    cdef vector[float] w


    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Population sizes
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
    
    # Create the projection data as CSR
    projection = CSR()
    for post in xrange(post_size):
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
        for pre in xrange(pre_size):
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
        if isinstance(delays, float):
            d = vector[int](nb_synapses, int(delays/dt))
        elif isinstance(delays, int):
            d = vector[int](nb_synapses, delays)
        elif isinstance(delays, RandomDistribution):
            d = [int(a/dt) for a in delays.get_list_values(nb_synapses) ]
        # Create the dendrite
        projection.add(post, r, w, d)

    return projection