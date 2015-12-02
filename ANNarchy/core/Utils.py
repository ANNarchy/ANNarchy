import numpy as np
import ANNarchy.core.Global as Global

######################
# Plotting methods
######################
def raster_plot(spikes):
    """
    Transforms recorded spikes to display easily a raster plot for a spiking population.

    It returns a (N, 2) Numpy array where each spike (first index) is represented by the corresponding time (first column) and the neuron index (second column).  It can be very easily plotted, for example with matplotlib::

        >>> monitor = Monitor(Pop, 'spike')
        >>> times, ranks = raster_plot( monitor.get('spike')
        >>> from pylab import *
        >>> plot(times, ranks, 'o')
        >>> show()

    *Parameters*:

    * **spikes**: the dictionary returned by the Monitor.get(...) method earlier.
    """
    ranks = []
    times = []

    if 'spike' in spikes.keys():
        data = spikes['spike']
    else:
        data = spikes

    # Compute raster
    for n in data.keys():
        for t in data[n]:
            times.append(t)
            ranks.append(n)

    return Global.dt()* np.array(times), np.array(ranks)

def histogram(data, binsize=Global.config['dt']):
    """
    **Deprecated!!**

    Returns for each recorded simulation step the number of spikes occuring in the population.

    *Parameters*:

    * **data**: the dictionary returned by the get_record() method for the population. 
    * **binsize**: the duration in milliseconds where spikes are averaged (default: dt). 
    """
    Global._warning("histogram() is deprecated, use a Monitor instead.")
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
    """ 
    **Deprecated!!**

    Takes the recorded spikes of a population and returns a smoothed firing rate.

    *Parameters*:

    * **data**: the dictionary returned by ``get_record()[pop]['spike']``

    * **smooth**: the smoothing time constant (default: 0 ms, not smoothed)
    """
    Global._warning("smoothed_rate() is deprecated, use a Monitor instead.")
    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.smoothed_rate(data, smooth)

def population_rate(data, smooth=Global.config['dt']):
    """ 
    **Deprecated!!**

    Takes the recorded spikes of a population and returns a smoothed firing rate for the whole population.

    *Parameters*:

    * **data**: the dictionary returned by ``get_record()[pop]['spike']``

    * **smooth**: the smoothing time constant (default: dt)
    """
    Global._warning("population_rate() is deprecated, use a Monitor instead.")
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



