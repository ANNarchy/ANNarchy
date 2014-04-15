import numpy as np
cimport numpy as np
    
    
cdef float angle = 0.0
cdef float radius = 0.5
cdef float sigma = 2.0

cdef np.ndarray x = np.linspace(0, 19, 20)
cdef np.ndarray y = np.linspace(0, 19, 20)
    
cdef np.ndarray xx, yy 
xx, yy = np.meshgrid(x, y)


def move(pop, float angle):   

    cdef float cw = 10.0 * ( 1.0 + radius * np.cos(2 * np.pi * angle ) )
    cdef float ch = 10.0 * ( 1.0 + radius * np.sin(2 * np.pi * angle ) )
    cdef np.ndarray data = (0.5 * np.exp(-((xx-cw)**2 + (yy-ch)**2)/2.0/sigma**2)).reshape((400,))
    pop.cyInstance.baseline = data
    
