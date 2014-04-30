*******************************
Defining synapses
*******************************

As for neurons, you can define some synaptic behaviour using parameters and variables within the **RateSynapse** respectively **SpikeSynapse** class. Although the description is local to a synapse, the same ODE will be applied to all synapses of a given Projection from one population to another. The same vocabulary as for neurons is accessible (constants, functions, conditional statements), except that the synapse must distinguish presynaptic and postsynaptic parameters/variables. 

RateSynapse
===================================

Like ``rate`` for a neuron, one variable is critical for a rate-coded synapse:

* ``value`` represents the synaptic efficiency (or the weight of the connection). If an ODE is defined for this variable, this will implement a learning rule. If none is provided, the synapse is non-plastic.

The ODEs for synaptic variables follow the same syntax as for neurons. The following attributes are defined:

* *dt*: the discretization step is 1.0ms by default. 

* *t*: current step of the simulation (incremented after each step of the simulation).

* *value* : represents the efficiency of a synapse. Overriding its equation implements synaptic plasticity.

* ``psp`` represents the postsynaptic potential evoked by the presynaptic neuron. This value is actually summed by the postsynaptic neuron with all other synapses of the same projection in ``sum(type)``. If not defined, it will simply represent the product between the pre-synaptic firing rate (``pre.rate``) and the weight value (``value``).

Neural variables therefore have to be prefixed with ``pre.`` or ``post.``: 

.. code-block:: python

    pre.rate, post.baseline, post.mp...
    
ANNarchy will check before the compilation that the pre- or post-synaptic neuron types indeed define such variables.

**Defining the postsynaptic potential (psp)**

The postsynaptic potential of a single synapse is by default:

.. code-block:: python

    psp = value * pre.rate
    
where ``pre.rate`` is the presynaptic firing rate, but you may want to override this behaviour in certain cases. 

For example, you may want to model a non linear synapse with a logarithmic term:

    .. math::
    
        r_{i} = \sum_j log \left( \frac {( r_{j} * w_{ij} ) + 1 } { ( r_{j} * w_{ij} ) - 1 } \right)

In this case, you can just modify the postsynaptic potential (*psp*) variable of the synapse:

.. code-block:: python 

    NonLinearSynapse = Synapse( 
        equation = """
            psp = log( (pre.rate * value + 1 ) / (pre.rate * value - 1) )
        """
    )

**Defining the learning rule**

Learning mechanisms are implemented in ANNarchy modifying the *value* variable of a Synapse. Within this modification any other variables of the synapse, the pre- and postsynaptic populations are available. 

The Oja learning rule:

.. math::

    \tau \frac{d w(t)}{dt} &= r_{pre} * r_{post} - \alpha * r_{post}^2 * w(t) 

could be implemented this way:

.. code-block:: python 

    equations="""
        tau * dvalue / dt = pre.rate * post.rate - alpha * post.rate^2 * value
    """
    
Note that in most cases, it would be equivalent to define the increment directly:

.. code-block:: python 

    equations="""
        value += dt / tau * ( pre.rate * post.rate - alpha * post.rate^2 * value)
    """

The synaptic weight (``value``) is already predefined so we need only to introduce the other parameters in a parameters block:

.. code-block:: python 

    parameters="""
        tau = 5000,
        alpha = 8.0,
    """

Please note that the simulation step ``dt`` is  globally defined in ANNarchy (default = 1 ms). The full description of a synapse learning according to the Oja learning rule would then be:

.. code-block:: python 

    Oja = RateSynapse(
        parameters="""
            tau = 5000,
            alpha = 8.0,
        """
        equations="""
            tau * dvalue / dt = pre.rate * post.rate - alpha * post.rate^2 * value
        """
    )

SpikeSynapse
===================================
   
**Increase of conductance**

In the simplest case, a presynaptic spike increases a ``target`` conductance value in the postsynaptic neuron. The rule defining how this conductance is modified can be placed in the pre-synaptic event section of a synapse:

.. code-block:: python

    pre_spike="""
        g_target += value
    """
    
Note that this is the default behaviour, only exceptions to this rule have to be implemented.

.. hint:: **current limitation**

    For the current implementation, it is obligatory to use the keyword ``g_target``. This value relates to the corresponding value in postsynaptic neuron: The ``target`` will be replaced with the projection's target (for example ``exc`` or ``inh``). So if you use this synapse in a projection with target = 'exc', the value of g_exc in postsynaptic neuron will be automatically replaced. In a further release it will be analogous to Brian.

**Defining the learning rule**

To define the learning rule you can describe the pre- and postsynaptic events separately in the synapse description (what happens when a pre- resp. post-synaptic spike is perceived at the corresponding synapse). The following example describes a basic implementation of STDP (Spike-Timing Dependent Plasticity), with the same formalism as in Brian:

.. code-block:: python

    SimpleLearn=SpikeSynapse(
        parameters = """
            tau_pre = 5 : postsynaptic
            tau_post = 5 : postsynaptic
            cApre = 1 : postsynaptic
            cApost = -1 : postsynaptic
        """,
        equations = """
            tau_pre * dApre/dt = -Apre
            tau_post * dApost/dt = -Apost
        """,
        pre_spike = """
            Apre += cApre
            g_target += value
            value += Apost
        """,                  
        post_spike = """
            Apost += cApost
            value += Apre
        """      
    ) 
    
The parameters are declared postsynaptic because they are the same for all synapses in the projection. The variables ``Apre`` and ``Apost`` are exponentially decreasing traces of pre- and post-synaptic spikes, as shown by the leaky integration in ``equations``. When a presynaptic spike is emitted, ``Apre`` is incremented, the conductance level of the postsynaptic neuron ``g_target`` too, and the synaptic efficiency is decreased proportionally to ``Apost`` (this means that if a post-synaptic spike was emitted shortly before, LTD will strongly be apllied, while if it was longer ago, no major change will be observed). When a post-synaptic spike is observed, ``Apost`` increases and the synaptic efficiency is increased proportionally to ``Apre``. 



    
