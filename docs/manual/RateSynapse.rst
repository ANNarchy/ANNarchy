*******************************
Rate-coded synapses
*******************************

As for neurons, you can define the synaptic behavior using a ``RateSynapse`` object. Although the description is local to a synapse, the same ODE will be applied to all synapses of a given Projection from one population to another. The same vocabulary as for neurons is accessible (constants, functions, conditional statements), except that the synapse must distinguish presynaptic and postsynaptic parameters/variables. 

Like ``r`` for a rate-coded neuron, one variable is critical for a rate-coded synapse:

* ``w`` represents the synaptic efficiency (or the weight of the connection). If an ODE is defined for this variable, this will implement a learning rule. If none is provided, the synapse is non-plastic.

The ODEs for synaptic variables follow the same syntax as for neurons. As for neurons, the following variables are already defined:

* ``t``: time in milliseconds elapsed since the creation of the network.

* ``dt``: the discretization step is 1.0ms by default. 

  

Synaptic plasticity
--------------------------

Learning is possible by modifying the  variable ``w`` of a single synapse. 

For example, the Oja learning rule (see the example :doc:`../example/BarLearning`):

.. math::

    \tau \frac{d w(t)}{dt} &= r_\text{pre} * r_\text{post} - \alpha * r_\text{post}^2 * w(t) 

could be implemented this way:

.. code-block:: python 

    Oja = RateSynapse(
        parameters="""
            tau = 5000
            alpha = 8.0
        """,
        equations="""
            tau * dw / dt = pre.r * post.r - alpha * post.r^2 * w
        """
    )
    
Note that in most cases, it would be equivalent to define the increment directly:

.. code-block:: python 

    equations="""
        w += dt / tau * ( pre.r * post.r - alpha * post.r^2 * w)
    """

The same vocabulary as for rate-coded neurons applies. Custom functions can also be defined:

.. code-block:: python 

    Oja = RateSynapse(
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

A synapse needs to access neural variables both at the pre- and post-synaptic levels.  For the presynaptic neuron, biologically realistic synapses should only need its firing rate, but in some cases it may be useful to access other variables as well.

In order to use neural variables in a synaptic variable, you have to prefix them with ``pre.`` or ``post.``. For example: 

.. code-block:: python

    pre.r, post.baseline, post.mp...
    
ANNarchy will check before the compilation that the pre- or post-synaptic neuron types indeed define such variables.

.. warning::

    As of version 4.1.3, only ``pre.r`` takes delays into account. Trying to access other presynaptic variables will return their value at the current simulation step.


Global operations
-----------------

Some learning rules require global information about the pre- or post-synaptic population, which is not local to the synapse, such as the mean or maximal activity in the presynaptic population. This information can be accessed at the synapse-level. The special functions:

* ``min`` for minimum,
* ``max`` for maximum and
* ``mean`` for mean
  
are available for any pre- or post-synaptic variable.

For example, some covariance-based learning rules depend on the mean firing in the pre- and post-synaptic populations: 

.. math::

    \tau \frac{d w(t)}{dt} &= (r_\text{pre} - \hat{r}_\text{pre} )  * (r_\text{post} - \hat{r}_\text{post} )

Using the global operations, such a learning rule is trivial to implement:

.. code-block:: python 

    Covariance = RateSynapse(
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

Defining the postsynaptic potential (psp)
-----------------------------------------

The argument ``psp`` of a ``RateSynapse`` object represents the postsynaptic potential evoked by the presynaptic neuron. This value is actually summed by the postsynaptic neuron over all other synapses of the same projection in ``sum(target)``. If not defined, it will simply represent the product between the pre-synaptic firing rate (``pre.r``) and the weight value (``w``).

The postsynaptic potential of a single synapse is by default:

.. code-block:: python

    psp = w * pre.r
    
where ``pre.r`` is the presynaptic firing rate, but you may want to override this behaviour in certain cases. 

For example, you may want to model a non-linear synapse with a logarithmic term:

    .. math::
    
        r_{i} = \sum_j log \left( \frac {( r_{j} * w_{ij} ) + 1 } { ( r_{j} * w_{ij} ) - 1 } \right)

In this case, you can just modify the ``psp`` argument of the synapse:

.. code-block:: python 

    NonLinearSynapse = RateSynapse( 
        psp = """
            log( (pre.r * w + 1 ) / (pre.r * w - 1) )
        """
    )

No further modification has to be done in the postsynaptic neuron, this value will be summed over all presynaptic neurons automatically when using ``sum(target)``.






    
