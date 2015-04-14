import numpy as np
import ANNarchy.core.Global as Global

######################
# Plotting methods
######################
def raster_plot(data):
    """ Transforms recorded spikes to display easily a raster plot for a spiking population.

    It returns a (N, 2) Numpy array where each spike (first index) is represented by the corresponding time (first column) and the neuron index (second column).  It can be very easily plotted, for example with matplotlib::

        >>> data = get_record()
        >>> spikes = raster_plot(data[pop]['spike'])
        >>> from pylab import *
        >>> plot(spikes[:, 0], spikes[:, 1], 'o')
        >>> show()

    *Parameters*:

    * **data**: the dictionary returned by the get_record() method for the population. 
    """
    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.raster_plot(data['data'])
    #return np.array([ [t, neuron] for neuron in range(len(data['data'])) for t in data['data'][neuron] ] )

def histogram(data, binsize=Global.config['dt']):
    """
    Returns for each recorded simulation step the number of spikes occuring in the population.

    *Parameters*:

    * **data**: the dictionary returned by the get_record() method for the population. 
    * **binsize**: the duration in milliseconds where spikes are averaged (default: dt). 
    """
    if isinstance(data['start'], int): # only one recording
        duration = data['stop'] - data['start']
    else:
        duration = 0
        for t in range(len(data['start'])):
            duration += data['stop'][t] - data['start'][t]
            
    nb_neurons = len(data['data'])
    nb_bins = int(duration*Global.config['dt']/binsize)
    spikes = [0 for t in xrange(nb_bins)]
    for neuron in range(nb_neurons):
        for t in data['data'][neuron]:
            spikes[int(t/float(binsize/Global.config['dt']))] += 1
    return np.array(spikes)

def smoothed_rate(data, smooth=0.0):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate.

    *Parameters*:

    * **data**: the dictionary returned by ``get_record()[pop]['spike']``

    * **smooth**: the smoothing time constant (default: 0 ms, not smoothed)
    """
    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.smoothed_rate(data, smooth)

def population_rate(data, smooth=Global.config['dt']):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate for the whole population.

    *Parameters*:

    * **data**: the dictionary returned by ``get_record()[pop]['spike']``

    * **smooth**: the smoothing time constant (default: dt)
    """
    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.population_rate(data, smooth)



######################
# Sparse matrices
######################

def sparse_random_matrix(pre, post, p, weight, delay=0):
    """
    Returns a sparse (lil) matrix to connect the pre and post populations with the probability p and the value weight.
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Global._warning("scipy is not installed, sparse matrices won't work")
        return None
    from random import sample
    W=lil_matrix((pre, post))
    for i in xrange(pre):
        k=np.random.binomial(post,p,1)[0]
        W.rows[i]=sample(xrange(post),k)
        W.data[i]=[weight]*k

    return W



