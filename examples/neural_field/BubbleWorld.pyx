from NeuralField import *
import numpy as np
cimport numpy as np
import math

def run(InputPop, FocusPop, proj):

    
    vis = Visualization( [ { 'pop': InputPop, 'var': 'rate' }, 
                           { 'pop': FocusPop, 'var': 'rate','min': 0.0, 'max': 1.0 },
                           { 'proj': proj, 'var': 'value', 
                             'min': -0.1, 'max': 0.1, 'title': 'Receptive fields'} ]
                         
                       )

    cdef period = 5000
    cdef int w = InputPop.geometry[0]
    cdef int h = InputPop.geometry[1]
    
    cdef float angle = 0.0
    cdef float radius = 0.5
    cdef float sigma = 2.0
    
    cdef int i, idx
    cdef float cw, ch, dist, value

    cdef np.ndarray data = np.zeros(w*h)
    
    for step in xrange (period):
        angle += 1.0/float(period)
        
        cw = w / 2.0 * ( 1.0 + radius * np.cos(2 * math.pi * angle ) )
        ch = h / 2.0 * ( 1.0 + radius * np.sin(2 * math.pi * angle ) )
        
        for x in xrange(w):
            for y in xrange(h):
                dist = (x-cw)**2 + (y-ch)**2
                value = 0.5 * np.exp(-dist/2.0/sigma**2)
                idx = x+y*w
                data[idx] = value

        InputPop.baseline = data

        simulate(1)
        if step%250 == 0:
            vis.render()

    print 'done'