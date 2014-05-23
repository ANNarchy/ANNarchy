import numpy as np

def raster_plot(data, compact=False):
    """ Transforms recorded spikes to display easily a raster plot for a spiking population.

    It returns a (N, 2) Numpy array where each spike (first index) is represented by the corresponding time (first column) and the neuron index (second column).  It can be very easily plotted, for example with matplotlib::

        >>> data = get_record()
        >>> spikes = raster_plot(data[pop]['spike'])
        >>> from pylab import *
        >>> plot(spikes[:, 0], spikes[:, 1], 'o')
        >>> show()

    If ``compact`` is set to ``True``, it will return a list of lists, where the first index corresponds to the neurons' ranks, and the second is a list of time steps where a spike was emitted.

    Parameters:

    * ``data``: the dictionary returned by the get_record() method for the population and 
    """
    return np.array([ [t, neuron] for neuron in range(len(data['data'])) for t in data['data'][neuron] ] )

def smoothed_rate(data, smooth=0.0):
    """ Takes the recorded spikes of a population and returns a smoothed firing rate.

    Parameters:

    * *data* the dictionary returned by ``get_record()[Pop]['spike']``

    * *smooth* the smoothing time constant (default: 0 ms, not smoothed)
    """
    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.smoothed_rate(data, smooth)