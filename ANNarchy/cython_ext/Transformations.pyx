# distutils: language = c++

import numpy as np
cimport numpy as np

import ANNarchy

cpdef np.ndarray raster_plot(list data):
    """
    Transforms recorded spikes into a (N, 2) array to easily display the raster plot.
    """
    cdef int N, n, t
    cdef np.ndarray res
    cdef list d
    N = len(data)
    d = []
    for n in xrange(N):
        for t in data[n]:
            d.append([t, n])
    res = np.array(d)
    return res

cpdef np.ndarray smoothed_rate(dict data, float smooth):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate.

    Parameters:

    * *data* the dictionary returned by ``Pop.get_record()['spike']``

    * *smooth* the smoothing time constant (default: 0 ms)
    """
    cdef np.ndarray res
    cdef int N, d
    cdef int n, t, timing, last_spike, idx
    cdef float dt, delta
    cdef list spikes

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Number of neurons
    d = data['stop'] - data['start']
    N = len(data['data'])

    # Prepare the matrix
    rates = np.zeros((N, d))

    # Compute instantaneous firing rate
    idx = 0
    for n, spikes in data['data'].items():
        last_spike = data['start'] - int(100.0/dt)
        for timing in spikes:
            if last_spike>data['start']:
                rates[idx, last_spike:timing] = 1000.0/dt/float(timing - last_spike)  
            last_spike = timing 
        idx += 1

    if smooth == 0.0:
        return rates

    smoothed_rate = np.zeros((N, d))
    smoothed_rate[:, 0] = rates[:, 0]
    delta = dt/smooth
    for t in xrange(d-1):
        smoothed_rate[:, t+1] = smoothed_rate[:, t] + (rates[:, t+1] - smoothed_rate[:, t])*delta
    return smoothed_rate

cpdef np.ndarray population_rate(dict data, float smooth):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate for the whole population.

    Parameters:

    * *data* the dictionary returned by ``Pop.get_record()['spike']``

    * *smooth* the smoothing time constant (default: dt)
    """
    cdef np.ndarray res
    cdef int N, d
    cdef int n, t, timing, last_spike
    cdef float dt, delta
    cdef list spikes

    # Retrieve simulation time step
    dt = ANNarchy.core.Global.config['dt']

    # Number of neurons
    d = data['stop'] - data['start'] + 1
    N = len(data['data'])

    # Prepare the matrix
    rates = np.zeros(d)

    # Compute histogram
    for n, spikes in data['data'].items():
        for t in spikes:
            rates[t - data['start']] += 1
    rates /= dt*N/1000.0
    
    # print(smooth, dt)

    if smooth <= dt:
        return rates

    smoothed_rate = np.zeros(d)
    smoothed_rate[0] = rates[0]
    delta = dt/smooth
    for t in xrange(d-1):
        smoothed_rate[t+1] = smoothed_rate[t] + (rates[t+1] - smoothed_rate[t])*delta

    return smoothed_rate

