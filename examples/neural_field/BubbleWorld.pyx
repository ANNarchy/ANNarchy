#cython: language_level=3
import numpy as np
cimport numpy as np
    
cdef class World:
    " Environment class allowing to clamp a rotating bubble into the baseline of a population."
    
    cdef pop # Input population
    cdef func # Function to call

    cdef float angle # Current angle
    cdef float radius # Radius of the circle 
    cdef float sigma # Width of the bubble
    cdef float period # Number of steps needed to make one revolution

    cdef np.ndarray xx, yy # indices
    cdef float cx, cy, midw, midh
    cdef np.ndarray data 
    
    def __cinit__(self, population, radius, sigma, period, func):
        " Constructor"
        self.pop = population
        self.func=func
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
        " Rotates the bubble for the given duration"
        cdef int t
        for t in xrange(duration):
            # Update the angle
            self.angle += 1.0/self.period
            # Compute the center of the bubble
            self.cx = self.midw * ( 1.0 + self.radius * np.cos(2.0 * np.pi * self.angle ) )
            self.cy = self.midh * ( 1.0 + self.radius * np.sin(2.0 * np.pi * self.angle ) )
            # Create the bubble
            self.data = (np.exp(-((self.xx-self.cx)**2 + (self.yy-self.cy)**2)/2.0/self.sigma**2))
            # Clamp the bubble into pop.baseline
            self.pop.baseline = self.data
            # Simulate for 1 step
            self.func()  
