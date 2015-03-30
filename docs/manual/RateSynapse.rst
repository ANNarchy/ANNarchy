*******************************
Rate-coded synapses
*******************************

As for neurons, you can define the synaptic behavior using a ``Synapse`` object. Although the description is local to a synapse, the same ODE will be applied to all synapses of a given Projection from one population to another. The same vocabulary as for neurons is accessible (constants, functions, conditional statements), except that the synapse must distinguish pre-synaptic and post-synaptic parameters/variables. 

Like ``r`` for a rate-coded neuron, one variable is special for a rate-coded synapse:

* ``w`` represents the synaptic efficiency (or the weight of the connection). If an ODE is defined for this variable, this will implement a learning rule. If none is provided, the synapse is non-plastic.

The ODEs for synaptic variables follow the same syntax as for neurons. As for neurons, the following variables are already defined:

* ``t``: time in milliseconds elapsed since the creation of the network.

* ``dt``: the discretization step is 1.0ms by default. 

  

Synaptic plasticity
--------------------------

Learning is possible by modifying the  variable ``w`` of a single synapse during the simulation. 

For example, the Oja learning rule (see the example :doc:`../example/BarLearning`):

.. math::

    \tau \frac{d w(t)}{dt} &= r_\text{pre} * r_\text{post} - \alpha * r_\text{post}^2 * w(t) 

could be implemented this way:

.. code-block:: python 

    Oja = Synapse(
        parameters="""
            tau = 5000
            alpha = 8.0
        """,
        equations="""
            tau * dw / dt = pre.r * post.r - alpha * post.r^2 * w
        """
    )
    
Note that it is equivalent to define the increment directly if you want to apply the explicit Euler method:

.. code-block:: python 

    equations="""
        w += dt / tau * ( pre.r * post.r - alpha * post.r^2 * w)
    """

The same vocabulary as for rate-coded neurons applies. Custom functions can also be defined:

.. code-block:: python 

    Oja = Synapse(
        parameters="""
            tau = 5000
            alpha = 8.0
        """,
        equations="""
            tau * dw / dt = product(pre.r,  post.r) - alpha * post.r^2 * w
        """,
        functions="""
            product(x,y) = x * y
        """,
    )


Neuron-specific variables
-----------------------------------

A synapse needs to access neural variables both at the pre- and post-synaptic levels.  For the pre-synaptic neuron, biologically realistic synapses should only need its firing rate, but in some cases it may be useful to access other variables as well.

In order to use neural variables in a synaptic variable, you have to prefix them with ``pre.`` or ``post.``. For example: 

.. code-block:: python

    pre.r, post.baseline, post.mp...
    
ANNarchy will check before the compilation that the pre- or post-synaptic neuron types indeed define such variables.

.. note::

    If the projection uses delays, all pre-synaptic variables used in the synapse model will be delayed.


Global operations
-----------------

Some learning rules require global information about the pre- or post-synaptic population, which is not local to the synapse, such as the mean or maximal activity in the pre-synaptic population. This information can be accessed at the synapse-level. The special functions:

* ``min(v)`` for the minimum: :math:`\min_i v_i`,
* ``max(v)`` for the maximum: :math:`\max_i v_i`,
* ``mean(v)`` for the mean: :math:`\frac{1}{N} \sum_i v_i`,
* ``norm1(v)`` for the L1-norm: :math:`\frac{1}{N} \sum_i |v_i|`,
* ``norm2(v)`` for the L2-norm: :math:`\frac{1}{N} \sum_i v_i^2`
  
are available for any pre- or post-synaptic variable.

For example, some covariance-based learning rules depend on the mean firing in the pre- and post-synaptic populations: 

.. math::

    \tau \frac{d w(t)}{dt} &= (r_\text{pre} - \hat{r}_\text{pre} )  * (r_\text{post} - \hat{r}_\text{post} )

Using the global operations, such a learning rule is trivial to implement:

.. code-block:: python 

    Covariance = Synapse(
        parameters="""
            tau = 5000.0
        """,
        equations="""
            tau * dw/dt = (pre.r - mean(pre.r) ) * (post.r - mean(post.r) )
        """
    )

.. warning::

    * Such global operations can become expensive to compute if the populations are too big.
    * The global operations are performed over the whole population, not only the synapses which actually reach the post-synaptic neuron.
    * They can only be applied to a single variable, not a combination or function of them.


Defining the post-synaptic potential (psp)
-------------------------------------------

The argument ``psp`` of a ``Synapse`` object represents the post-synaptic potential evoked by the pre-synaptic neuron. This value is actually summed by the post-synaptic neuron over all other synapses of the same projection in ``sum(target)``. If not defined, it will simply represent the product between the pre-synaptic firing rate (``pre.r``) and the weight value (``w``).

The post-synaptic potential of a single synapse is by default:

.. code-block:: python

    psp = w * pre.r
    
where ``pre.r`` is the pre-synaptic firing rate, but you may want to override this behaviour in certain cases. 

For example, you may want to model a non-linear synapse with a logarithmic term:

    .. math::
    
        r_{i} = \sum_j log \left( \frac {( r_{j} * w_{ij} ) + 1 } { ( r_{j} * w_{ij} ) - 1 } \right)

In this case, you can just modify the ``psp`` argument of the synapse:

.. code-block:: python 

    NonLinearSynapse = Synapse( 
        psp = """
            log( (pre.r * w + 1 ) / (pre.r * w - 1) )
        """
    )

No further modification has to be done in the post-synaptic neuron, this value will be summed over all pre-synaptic neurons automatically when using ``sum(target)``.



Defining the post-synaptic operation
----------------------------------------

By default, a post-synaptic neuron calling ``sum(target)`` will compute the sum over all incoming synapses of their defined ``psp``:

.. math::

    \text{sum(exc)} = \sum_{i \in \text{exc}} \text{psp}(i) = \sum_{i \in \text{exc}} w_i * \text{pre}.r_i 

It is possible to define a different operation performed on the connected synapses, using the ``operation`` argument of the synapse:

.. code-block:: python 

    MaxPooling = Synapse(
        psp = "w * pre.r",
        operation = "max"
    )

In this case, ``sum(target)`` will represent the maximum value of ``w * pre.r`` over all incoming synapses, not their sum. It can be useful when defining pooling operations in a convolutional network, for example.

The available operations are:

* "sum" (default): sum of all incoming psps.
* "max": maximum of all incoming psps.
* "min": minimum of all incoming psps.
* "mean": mean of all incoming psps.

.. warning::

    These operations are only possible for rate-coded synapses.


    
