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


Contrary to normal projections, the geometry of the populations (number od dimensions and neurons in each dimension) has a great influence on the operation to be performed. In particular the number of dimensions  will define how the convolution will be applied. 

If for example the pre-synaptic population represents an 2D image, you may want to apply a vertical edge detector to it and get the result in the post-synaptic population. Such a filter can be defined by the following Numpy array::

    vertical_filter = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0]
        ]
    )

With 2 dimensions, the convolution operation is defined for all neurons in the post-synaptic population by:

.. math::

    \text{post}[i, j] = \sum_{c_i=-1}^1 \sum_{c_j=-1}^1 \text{filter}[c_i+1][c_j+1] \cdot \text{pre}[i - c_i, j - c_j] 

Such a convolution is achieved by creating a ``SharedProjection`` object and using the ``convolve()`` method to create the connection pattern::

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve(weights=vertical_filter)

Each neuron of the post-synaptic population will then receive in ``sum('exc')`` (or whatever target name is used) the convolution between the kernel and a sub-region of the pre-syanptic population. ANNarchy defines the convolution operation for populations having 1, 2, 3, or 4 dimensions.

Several options can be passed to the ``convolve()`` method:

* ``method`` describes the convolution method. By default, ``convolve()`` performs the mathematical convolution, which reduces to an element-wise multiplication between the kernel and a flipped version of the sub-array of pre-synaptic rates(minus sign in :math:`[i - c_i, j - c_j]`). In some cases, it may be useful to avoid this flip, in order to interpret the kernel as a filter or receptive field. You can set the ``method`` argument to ``'filter'`` to replace the minus sign by a plus sign in the generated code.

* ``padding`` defines the value of the pre-synaptic firing rates which will be used when the coordinates are out-of-bounds. By default zero-padding is used, but you can specify another value with this argument. You can also use the ``'border'`` value to repeat the firing rate of the neurons on the border (for example, if the filter tries to reach a neuron of coordinates (-1, -1), the firing rate of the neuron (0, 0) will be used instead).

* ``subsampling``. In convolutional networks, the convolution operation is often coupled with a reduction in the number of neurons in each dimension. In the example above, the post-synaptic population could be defined with a geometry (50, 50). For each post-synaptic neuron, the coordinates of the center of the applied kernel would be automatically shifted from two pre-synaptic neurons compared to the preious one. However, if the number of neurons in one dimension of the pre-synaptic population is not exactly a multiple of the number of post-synaptic neurons in the same dimension, ANNarchy can not guess what the correct correspondance should be. In this case, you have to specify this mapping by providing to the ``subsampling`` argument a list of pre-synaptic coordinates defining the position of the center of the kernel for each post-synaptic neuron. The list is indexed by the rank of the post-synaptic neurons (use the ``rank_from_coordinates()`` method) and must have the same size as the population. Each element should be a list of coordinates in the pre-synaptic population's geometry (with as many elements as dimensions). It is possible to provide a Numpy array instead of a list of lists.

One can access the coordinates in the pre-synaptic geometry of the center of the filter corresponding to a particular post-synaptic neuron by calling the ``center()`` method of ``SharedProjection`` with the rank or coordinates of the post neuron::


    pre = Population(geometry=(100, 100), neuron = Whatever)
    post = Population(geometry=(50, 50), neuron = Whatever)

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve(weights=vertical_filter)

    pre_coordinates = proj.center(10, 10) # returns (20, 20)


In some cases, the post-synaptic population can have less dimensions than the pre-synaptic one. An example would be when the pre-synaptic population has three dimensions (e.g. (100, 100, 3)), the last representing the R, G and components of an image. A 3D filter, with 3 components in the last dimension, would result in a (100, 100, 1) post-synaptic population (or any subsampling of it). ANNarchy accepts in this case the use of a 2D population (100, 100), but it will be checked that the number of elements in the last dimension of the flter equals the number of pre-synaptic neurons in the last dimension::

    pre = Population(geometry=(100, 100, 3), neuron = Whatever)
    post = Population(geometry=(100, 100), neuron = Whatever)

    red_filter = np.array(
        [
            [
                [2.0, -1.0, -1.0]
            ]
        ]
    )

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve(weights=red_filter, method='filter')

Non-linear convolutions
=======================

A ``SharedProjection`` uses by default a regular convolution, summing ``w * pre.r`` over the extent of the kernel. As for regular synapses, you can change this behavior when creating the projection:

* the ``psp`` argument defines what will be summed. It is ``w*pre.r`` by default but can be changed to any combination of ``w`` and ``pre.r``, such as ``w * log(1+pre.r)``::

    proj = SharedProjection(pre=pre, post=post, target='exc', psp='w*log(1+pre.r)')

* the ``operation`` argument allows to change the summation operation. You can set it to 'max' (the maximum value of ``w*pre.r`` over the extent of the filter will be returned), 'min' (minimum) or 'mean' (same as 'sum', but normalized by the number of elements in the filter). The default is 'sum'::

    proj = SharedProjection(pre=pre, post=post, target='exc', operation='max')


Layer-wise convolutions
=======================

It is possible to define kernels with less dimensions than the pre-synaptic population. A 2D filter can for example be applied on each color component independently::

    pre = Population(geometry=(100, 100, 3), neuron = Whatever)
    post = Population(geometry=(50, 50, 3), neuron = Whatever)

    vertical_filter = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0]
        ]
    )

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve(weights=vertical_filter, method='filter', keep_last_dimension=True)

The important parameter in this case is ``keep_last_dimension`` which tells the code generator that the last dimension of the input should not be used for convolution. The important constraint is that the post-synaptic population **must** have the same number of neurons in the last dimension than the pre-synaptic one (no subsampling is possible by definition). 


Bank of filters
=====================

Convolutional networks often use banks of filters to perform different operations (such as edge detection with various orientations). It is possible to specify this mode of functioning by setting the ``multiple`` parameter to ``True``::

    pre = Population(geometry=(100, 100), neuron = Whatever)
    post = Population(geometry=(50, 50, 4), neuron = Whatever)

    bank_filters = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0]
        ],
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0]
        ],
        [
            [-1.0, -1.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  1.0,  1.0]
        ],
        [
            [ 1.0,  1.0,  1.0],
            [ 0.0,  0.0,  0.0],
            [-1.0, -1.0, -1.0]
        ]
    )

    proj = SharedProjection(pre=pre, post=post, target='exc')
    proj.convolve(weights=bank_filters, method='filter', multiple=True)


Here the filter has three dimensions. The first one **must** correspond to each filter. The last dimension of the post-synaptic population **must** correspond to the total number of filters. It can not be combined with ``keep_last_dimension``.

.. note::  

    **Current limitation**:  Each filter must have the same size, it is not possible yet to convolve over multiple scales.

Pooling
=======

Another form of atypical projection for a neural network is the pooling operation. In max-pooling, each post-synaptic neuron is associated to a region of the pre-synaptic population and responds like the maximum firing rate in this region. This is already possible by defining the ``operation`` argument of the synapse type, but it would use instantiated synapses, what would be a waste of memory.

The ``SharedProjection`` class has a ``pooling`` method able to define such an operation without defining any synapse::

    pre = Population(geometry=(100, 100), neuron = Whatever)
    post = Population(geometry=(50, 50), neuron = Whatever)

    proj = SharedProjection(pre=pre, post=post, target='exc', operation='max')
    proj.pooling()

The pooling region of a post-synaptic region is automatically determined by comparing the dimensions of the two populations: here each post-synaptic neuron will cover an area of 2*2 pixels. 

If the number of dimensions do not match, you have to specify the ``extent`` argument to ``pooling()``. For example, you can pool completely over one dimension of the pre-synaptic population::

    pre = Population(geometry=(100, 100, 10), neuron = Whatever)
    post = Population(geometry=(50, 50), neuron = Whatever)

    proj = SharedProjection(pre=pre, post=post, target='exc', operation='max')
    proj.pooling(extent=(2, 2, 10))

.. warning::

    The default ``operation`` for a SharedProjection is ``'sum'``. You should not forget to set it to ``'max'`` if you want max-pooling (or ``'min'`` or ``'mean'``). 


Sharing weights with another projection
=======================================

A different possibility to share weights is between two projections. If your network is composed of populations of the same size, and the projection patterns are identical, it could save some memory to "share" the weights of one projection with another, so they are created only once.

To this end, you can use the ``copy()`` method of SharedProjection and pass it an existing projection::

    pop1 = Population(geometry=(30, 30), neuron=Whatever)
    pop2 = Population(geometry=(20, 20), neuron=Whatever)
    pop3 = Population(geometry=(20, 20), neuron=Whatever)


    proj1 = Projection(pop1, pop2, 'exc').connect_gaussian(amp = 1.0, sigma=0.3, delays=2.0)
    proj2 = SharedProjection(pop1, pop3, 'exc').copy(proj1)

This only works when the pre- and post-populations of each projection have the same geometry, but they can be different, of course. If the original projection is learnable, the copied projection will see the changes. However, it is not possible for the shared projection to learn on its own. SharedProjection only accepts ``psp`` and ``operation`` as parameters, which can be different from the original projection.

It is only possible to copy regular projections, not other shared projections. The transmission delays wil lbe identical between the two projections.