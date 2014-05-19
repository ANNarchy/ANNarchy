# distutils: language = c++

cpdef float comp_dist1D(tuple pre, tuple post)

cpdef float comp_dist2D(tuple pre, tuple post)

cpdef float comp_dist3D(tuple pre, tuple post)

cpdef float comp_distND(tuple pre, tuple post)

cpdef tuple get_1d_coord(int rank, tuple geometry)

cpdef tuple get_2d_coord(int rank, tuple geometry)

cpdef tuple get_3d_coord(int rank, tuple geometry) 

cpdef tuple get_coord(int rank, tuple geometry) 

cpdef int get_rank_from_1d_coord(tuple coord, tuple geometry)

cpdef int get_rank_from_2d_coord(tuple coord, tuple geometry)

cpdef int get_rank_from_3d_coord(tuple coord, tuple geometry)

cpdef int get_rank_from_coord(tuple coord, tuple geometry)

cpdef tuple get_normalized_1d_coord(int rank, tuple geometry)

cpdef tuple get_normalized_2d_coord(int rank, tuple geometry)

cpdef tuple get_normalized_3d_coord(int rank, tuple geometry)

cpdef tuple get_normalized_coord(int rank, tuple geometry)