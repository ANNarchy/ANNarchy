# distutils: language = c++

import numpy as np
cimport numpy as np

import ANNarchy

cpdef np.ndarray smoothed_rate(dict data, float smooth):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate.

    Parameters:

    * *data* the dictionary returned by ``get_record()[Pop]['spike']``

    * *smooth* the smoothing time constant (default: 0 ms)
    """
    cdef np.ndarray res
    cdef int N, d
    cdef int n, t, timing, last_spike
    cdef float dt

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Number of neurons
    d = data['stop'] - data['start']
    N = len(data['data'])

    # Prepare the matrix
    rates = np.zeros((N, d))

    # Compute instantaneous firing rate
    for n in xrange(N):
        last_spike = data['start'] - 100
        for timing in data['data'][n]:
            if last_spike>data['start']:
                rates[n, last_spike:timing] = 1000.0/dt/float(timing - last_spike)  
            last_spike = timing 

    if smooth > 0.0:
        smoothed_rate = np.zeros((N, d))
        for t in xrange(d-1):
            smoothed_rate[:, t+1] = smoothed_rate[:, t] + (rates[:, t+1] - smoothed_rate[:, t])/smooth
        return smoothed_rate
    else:
        return rates