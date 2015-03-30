*******************************
Rate-coded neurons
*******************************

Defining parameters and variables
---------------------------------

Let's consider first a simple rate-coded neuron of the leaky-integrator type, which simply integrates the weighted sum of its excitatory inputs:

.. math::

    \tau \frac{d \text{mp}(t)}{dt} &= ( B - \text{mp}(t) ) + \sum_{i}^{\text{exc}} \text{r}_{i} * w_{i} \\ 
           
    \text{r}(t) & = ( \text{mp}(t) )^+
    
where :math:`mp(t)` represents the membrane potential of the neuron, :math:`\tau` the time constant of the neuron, :math:`B` its baseline firing rate, :math:`\text{r}(t)` its instantaneous firing rate, :math:`i` an index over all excitatory synapses of this neuron, :math:`w_i` the efficiency of the synapse with the pre-synaptic neuron of firing rate :math:`\text{r}_{i}`. 

It can be implemented in the ANNarchy framework with:

.. code-block:: python

    LeakyIntegratorNeuron = Neuron(
        parameters="""   
            tau = 10.0
            baseline = -0.2
        """,
        equations = """
            tau * dmp/dt + mp = baseline + sum(exc)
            r = pos(mp)
        """
    )
    
The only required variable is ``r``, which represents the instantaneous firing rate and will be used to propagate activity in the network. All other parameters and variables are freely decided by the user. More detailed examples can be found in the section :doc:`../Example`.

Custom functions
-----------------

Custom functions can also be passed when creating the Neuron type:

.. code-block:: python

    LeakyIntegratorNeuron = Neuron(
        parameters="""   
            tau = 10.0
            baseline = -0.2
        """,
        equations = """
            tau * dmp/dt + mp = baseline + sum(exc)
            r = sigmoid(mp)
        """,
        functions == """
            sigmoid(x) = 1.0 / (1.0 + exp(-x))
        """
    )

Predefined attributes
----------------------

The ODE can depend on other parameters of the neuron (e.g. ``r`` depends on ``mp``), but not on unknown names. ANNarchy already defines the following variables and parameters for a neuron:
    
    * variable *t*: time in milliseconds elapsed since the creation of the network.
    
    * parameter *dt*: the discretization step, default is 1 ms. 
    
Weighted sum of inputs
-----------------------

The ``sum()`` method of a neuron gives a direct access to the weighted sum of all inputs to the post-synaptic neuron separately by the target. These synapses are organized in a data structure called ``Dendrite``. 

It is possible to modify how weighted sum is computed when creating a :doc:`RateSynapse`.

.. warning:: 

    The connection type, e.g. *exc* or *inh*, need to match with the names used as a ``target`` parameter when creating a ``Projection``. If such a projection does not exist when the network is compiled, the weighted sum will be set to 0.0.



Global operations
-----------------

One has the possibility to use global operations on the population inside the neuron definition, such as the maximal activity in the population. One only needs to use one of the following operations:

* ``min(v)`` for the minimum: :math:`\min_i v_i`,
* ``max(v)`` for the maximum: :math:`\max_i v_i`,
* ``mean(v)`` for the mean: :math:`\frac{1}{N} \sum_i v_i`,
* ``norm1(v)`` for the L1-norm: :math:`\frac{1}{N} \sum_i |v_i|`,
* ``norm2(v)`` for the L2-norm: :math:`\frac{1}{N} \sum_i v_i^2`

Example where neurons react to their inputs only where they exceed the mean over the population::

     WTANeuron = Neuron(
        parameters="""   
            tau = 10.0
        """,
        equations = """
            input = sum(exc)
            tau * dr/dt + r = pos(input - mean(input))
        """
    )   

.. note::

    The global operations are computed using values at the previous time step (like weighted sums), not in the step currently evaluated. There is therefore implicitely a delay of ``dt``, but it cannot be changed. 
