# distutils: language = c++
import numpy as np
cimport numpy as np

##########################################################
#                                                        #
# computation of euklidean distance between two vectors  #
#                                                        #
##########################################################
cpdef float comp_dist1D(tuple pre, tuple post):
    """
    Compute euklidean distance between two coordinates. 
    """
    return (pre[0]-post[0])**2;

cpdef float comp_dist2D(tuple pre, tuple post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef float res = 0.0

    res += (pre[0]-post[0])**2;
    res += (pre[1]-post[1])**2;

    return res

cpdef float comp_dist3D(tuple pre, tuple post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef float res = 0.0

    res += (pre[0]-post[0])**2;
    res += (pre[1]-post[1])**2;
    res += (pre[2]-post[2])**2;
    
    return res

cpdef float comp_distND(tuple pre, tuple post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef float res = 0.0

    for i in range(len(pre)):
        res += (pre[i]-post[i])**2;
    
    return res

##########################################################
#                                                        #
# computation of coordinates from rank                   #
# (c-style indexed !!!!)                                 #
##########################################################
cpdef tuple get_1d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    return (rank, )

cpdef tuple get_2d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    cdef int x, y
    x = (rank / (geometry[1]) ) % geometry[0]
    y = rank % (geometry[1])
    
    return (x,y)

cpdef tuple get_3d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    cdef int x, y, z
    z = rank % geometry[2]
    y = (rank / geometry[2]) % geometry[1]
    x = (rank / ((geometry[1])*(geometry[2]))) % geometry[0]
    
    return (x,y,z) 

cpdef tuple get_coord(int rank, tuple geometry):
    """
    Uing Numpy
    """   
    return np.unravel_index(rank, geometry) 

##########################################################
#                                                        #
# computation of coordinates from rank                   #
# (c-style indexed !!!!)                                 #
##########################################################
cpdef int get_rank_from_1d_coord(tuple coord, tuple geometry):
    """
    row ordered
    """
    return coord[0]

cpdef int get_rank_from_2d_coord(tuple coord, tuple geometry):
    """
    row ordered
    """
    return coord[1] + geometry[1] * coord[0] 

cpdef int get_rank_from_3d_coord(tuple coord, tuple geometry):
    """
    row ordered
    """
    return coord[2] + geometry[2]*coord[1] + coord[0] * geometry[1] * geometry[2]

cpdef int get_rank_from_coord(tuple coord, tuple geometry):
    """
    Using Numpy
    """
    return np.ravel_multi_index(coord, geometry)

##########################################################
#                                                        #
# computation of normalized coordinates                  #
# (c-style indexed !!!!)                                 #
##########################################################
cpdef tuple get_normalized_1d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    cdef float norm_x
    norm_x = 0
    if geometry[0]>1:
        norm_x = rank / (geometry[0]-1)
    
    return (norm_x, )

cpdef tuple get_normalized_2d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    cdef float norm_x, norm_y
    cdef int x, y
    x = (rank / (geometry[1]) ) % geometry[0]
    y = rank % (geometry[1])

    norm_x = 0.0
    norm_y = 0.0
    
    if geometry[0]>1:
        norm_x = x / float(geometry[0]-1)
    if geometry[1]>1:
        norm_y = y / float(geometry[1]-1)
    
    return (norm_x, norm_y)

cpdef tuple get_normalized_3d_coord(int rank, tuple geometry):
    """
    row ordered
    """
    cdef float norm_x, norm_y, norm_z
    cdef int x, y, z

    z = rank % geometry[2]
    y = (rank / geometry[2]) % geometry[1]
    x = (rank / ((geometry[1])*(geometry[2]))) % geometry[0]

    norm_x = 0
    norm_y = 0
    norm_z = 0

    if geometry[0]>1:
        norm_x = x / (geometry[0]-1)
    if geometry[1]>1:
        norm_y = y / (geometry[1]-1) 
    if geometry[2]>1:
        norm_z = z / (geometry[2]-1)
    
    return (norm_x, norm_y, norm_z) 

cpdef tuple get_normalized_coord(int rank, tuple geometry):
    """
    Works for any geometry
    """
    cdef tuple coord, norm_coord
    cdef int c

    coord = np.unravel_index(rank, geometry)
    norm_coord = ()
    for c in range(len(coord)):
        norm_coord = norm_coord + (coord[c]/float(geometry[c]-1),)

    
    return norm_coord