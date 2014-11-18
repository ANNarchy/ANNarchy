========================
Convolution and pooling
========================

Projections use by default a set of weights per post-synaptic neuron. Some networks, including convolutional networks, define a single operation (convolution or pooling) to be applied systematically on all pre-synaptic neurons. In such cases, it would be a waste of resources to allocate weights for each post-synaptic neuron. The extension ``weightsharing`` allows to implement such specific projections. It has to be imported explicitely at the beginning of the script::

    from ANNarchy import *
    from ANNarchy.extensions.weightsharing import *


.. warning::

    As of version 4.3.1, weight-sharing is only implemented for rate-coded networks. The only possible backend is OpenMP, CUDA will be implemented later.


Simple convolutions
===================

The simplest case of convolution is when the pre- and post-synaptic population have the same number of dimensions, for example::

    pre = Population(geometry=(100, 100), neuron = Whatever)
    post = Population(geometry=(100, 100), neuron = Whatever)


If for example the pre-synaptic population represents an image, you may want to apply a vertical edge detector to it and get the result in the post-synaptic population. Such a filter can be defined by the following Numpy array::

    vertical_filter = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0]
        ]
    )

The convolution operation is defined for all neurons in the post-synaptic population by:

.. math::

    \text{post}[i, j] = \sum_{c_i=-1}^1 \sum_{c_j=-1}^1 \text{filter}[c_i+1][c_j+1] \cdot \text{pre}[i - c_i, j - c_j] 

Such a convolution is achieved by creating a ``SharedProjection`` object and using the ``convolve()`` method to create the connection pattern::

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve()

Parallel convolutions
=====================


Pooling
=======


Bank of filters
=====================