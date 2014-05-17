# distutils: language = c++

import cython

##########################################################
#                                                        #
# computation of euklidean distance between two vectors  #
#                                                        #
##########################################################
def comp_dist1D(pre, post):
    """
    Compute euklidean distance between two coordinates. 
    """
    return (pre[0]-post[0])*(pre[0]-post[0]);

def comp_dist2D(pre, post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef double res = 0.0

    res += (pre[0]-post[0])*(pre[0]-post[0]);
    res += (pre[1]-post[1])*(pre[1]-post[1]);

    return res

def comp_dist3D(pre, post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef double res = 0.0

    res += (pre[0]-post[0])*(pre[0]-post[0]);
    res += (pre[1]-post[1])*(pre[1]-post[1]);
    res += (pre[2]-post[2])*(pre[2]-post[2]);
    
    return res

def comp_distND(pre, post):
    """
    Compute euklidean distance between two coordinates. 
    """
    cdef double res = 0.0

    for i in range(len(pre)):
        res += (pre[i]-post[i])*(pre[i]-post[i]);
    
    return res

##########################################################
#                                                        #
# computation of coordinates from rank                   #
# (c-style indexed !!!!)                                 #
##########################################################
def get_1d_coord(rank, geometry):
    """
    row ordered
    """
    return (rank, )

def get_2d_coord(rank, geometry):
    """
    row ordered
    """
    x = (rank / (geometry[1]) ) % geometry[0]
    y = rank % (geometry[1])
    
    return (x,y)

def get_3d_coord(rank, geometry):
    """
    row ordered
    """
    z = rank % geometry[2]
    y = (rank / geometry[2]) % geometry[1]
    x = (rank / ((geometry[1])*(geometry[2]))) % geometry[0]
    
    return (x,y,z) 

##########################################################
#                                                        #
# computation of coordinates from rank                   #
# (c-style indexed !!!!)                                 #
##########################################################
def get_rank_from_1d_coord(coord, geometry):
    """
    row ordered
    """
    return coord[0]

def get_rank_from_2d_coord(coord, geometry):
    """
    row ordered
    """
    return coord[1] + geometry[1] * coord[0] 

def get_rank_from_3d_coord(coord, geometry):
    """
    row ordered
    """
    return coord[2] + geometry[2]*coord[1] + coord[0] * geometry[1] * geometry[2]

##########################################################
#                                                        #
# computation of normalized coordinates                  #
# (c-style indexed !!!!)                                 #
##########################################################
def get_normalized_1d_coord(rank, geometry):
    """
    row ordered
    """
    norm_x = 0
    if geometry[0]>1:
        norm_x = rank / (geometry[0]-1)
    
    return norm_x

def get_normalized_2d_coord(rank, geometry):
    """
    row ordered
    """
    x = (rank / (geometry[1]) ) % geometry[0]
    y = rank % (geometry[1])

    norm_x = 0.0
    norm_y = 0.0
    
    if geometry[0]>1:
        norm_x = x / float(geometry[0]-1)
    if geometry[1]>1:
        norm_y = y / float(geometry[1]-1)
    
    return (norm_x, norm_y)

def get_normalized_3d_coord(rank, geometry):
    """
    row ordered
    """
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