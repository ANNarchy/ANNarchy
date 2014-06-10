*******************************
Rate-coded neurons
*******************************

Defining parameters and variables
---------------------------------

Let's consider first a simple rate-coded neuron of the leaky-integrator type, which simply integrates the weighted sum of its excitatory inputs:

.. math::

    \tau \frac{d \text{mp}(t)}{dt} &= ( B - \text{mp}(t) ) + \sum_{i}^{\text{exc}} \text{r}_{i} * w_{i} \\ 
           
    \text{r}(t) & = ( \text{mp}(t) )^+
    
where :math:`mp(t)` represents the membrane potential of the neuron, :math:`\tau` the time constant of the neuron, :math:`B` its baseline firing rate, :math:`\text{r}(t)` its instantaneous firing rate, :math:`i` an index over all excitatory synapses of this neuron, :math:`w_i` the efficiency of the synapse with the presynaptic neuron of firing rate :math:`\text{r}_{i}`. 

It can be implemented in the ANNarchy framework with:

.. code-block:: python

    LeakyIntegratorNeuron = RateNeuron(
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

    LeakyIntegratorNeuron = RateNeuron(
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

The ODE can depend on other parameters of the neuron (e.g. ``r`` depends on ``mp``), but not on unknown names. ANNarchy already ensures the existence of the following variables and parameters for a neuron:
    
    * variable *t*: time in milliseconds elapsed since the creation of the network.
    
    * parameter *dt*: the discretization step, default is 1.0ms. 
    
Weighted sum of inputs
-----------------------

The ``sum()`` method of a neuron gives a direct access to the weighted sum of all inputs to the postsynaptic neuron separately by the target. These synapses are organized in a data structure called ``Dendrite``. 

It is possible to modify how weighted sum is computed when creating a :doc:`RateSynapse`.

.. warning:: 

    The connection type, e.g. *exc* or *inh*, need to match with the names used as a ``target`` parameter when creating a ``Projection``. If such a projection does not exist when the network is compiled, the weighted sum will be set to 0.0.


