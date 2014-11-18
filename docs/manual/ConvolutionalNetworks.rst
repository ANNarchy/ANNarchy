======================
Convolutional networks
======================

Projections use by default a set of weights per post-synaptic neuron. Some networks, including convolutional networks, define a single operation (convolution or pooling) to be applied systematically on all pre-synaptic neurons. In such cases, it would be a waste of resources to allocate weights for each post-synaptic neuron. The extension ``weightsharing`` allows to implement such specific projections. It has to be imported explicitely at the beginning of the script::

    from ANNarchy import *
    from ANNarchy.extensions.weightsharing import *


.. warning::

    As of version 4.3.1, weight-sharing is only implemented for rate-coded networks. The only possible back-end is OpenMP, CUDA will be implemented later.


Simple convolutions
===================

The simplest case of convolution is when the pre- and post-synaptic population have the same number of dimensions

