import numpy as np
cimport numpy as np
from NeuralField import simulate, get_population
    
cdef class World:
    
    cdef pop # Input population
    
    cdef float angle
    cdef float radius 
    cdef float sigma 
    cdef float period 

    cdef np.ndarray xx, yy 
    cdef float cw, ch, midw, midh
    cdef np.ndarray data 
    
    def __cinit__(self, pop, radius, sigma, period):
        self.pop = pop
        self.angle = 0.0
        self.radius = radius
        self.sigma = sigma
        self.period = period
        cdef np.ndarray x = np.linspace(0, self.pop.geometry[0]-1, self.pop.geometry[0])
        cdef np.ndarray y = np.linspace(0, self.pop.geometry[1]-1, self.pop.geometry[1])
        self.xx, self.yy = np.meshgrid(x, y)
        self.midw = self.pop.geometry[0]/2
        self.midh = self.pop.geometry[1]/2
    
    def rotate(self, int duration):
        for t in range(duration):
            self.angle += 1.0/self.period
            self.cw = self.midw * ( 1.0 + self.radius * np.cos(2.0 * np.pi * self.angle ) )
            self.ch = self.midh * ( 1.0 + self.radius * np.sin(2.0 * np.pi * self.angle ) )
            self.data = (0.5 * np.exp(-((self.xx-self.cw)**2 + (self.yy-self.ch)**2)/2.0/self.sigma**2))
            self.pop.baseline = self.data
            simulate(1)  
