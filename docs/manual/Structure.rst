*******************************
Neural network
*******************************

Definition of a neural network
==============================

A neural network in ANNarchy is a collection of interconnected **Populations**. Each population comprises a set of similar artificial **Neurons**, whose mean-firing rate or spiking behavior is governed by one or many ordinary differential equations (ODE). These ODEs are dependent on the activity of other neurons through **Synapses**. The connection pattern between two populations is called a **Projection**.

The efficiency of the connections received by a neuron is stored in different arrays called **Dendrites**, depending on the type that was assigned to them: excitatory, inhibitory, modulatory... This typed organization of afferent connections also allows to easily apply them different learning rules (Hebbian, three-factor, Oja, BCM, STDP...).

.. image:: ../_static/neuralnetwork.png
    :width: 50%
    :align: center
    :alt: Structure of a neural network


To define a neural network and simulate its behaviour, you need to define the following information:

* The number of populations, their geometry (number of neurons, optionally the spatial structure - 1D/2D/3D).
        
* For each population, the type of neuron composing it, with all the necessary ODEs.
        
* For each projection between two populations, the connection pattern (all-to-all, one-to-one, distance-dependent...), the initial synaptic efficiencies, the delays in synaptic transmission.
        
* Optionally for a projection, the ODEs describing the evolution of synaptic efficiencies during learning.
        
* The interaction of the network with its environment (I/O relationships, rewarded tasks, fitting procedure...)     
    
ANNarchy provides a convenient way to define this information in a single Python script. In this manual, we will focus on an simple rate-coded network composed of two interconnected populations ``pop1`` and ``pop2``, but more complex architectures are of course possible (see the examples in section :doc:`../Example`).

Basic structure of a script
===========================

In a script file (e.g. ``MyNetwork.py``), you first need to import the ANNarchy package:

.. code-block:: python
    
    from ANNarchy import *

All the necessary objects and class definitions are then imported. The next step is to define the neurons and synapses needed by your network. To keep things simple, we will define a simple neuron model, whose firing rate is determined by the leaky-integration of excitatory inputs:

.. code-block:: python
    
    LeakyIntegratorNeuron = Neuron(
        parameters = """
            tau = 10.0
            baseline = -0.2
        """,
        equations = """
            tau * dmp/dt  + mp = baseline + sum(exc)
            r = pos(mp)
        """
    )

``mp`` is an internal variable integrating with the time constant ``tau`` the weighted sum of excitatory inputs ``sum(exc)`` to this neuron plus its ``baseline`` activity. ``r`` is the instantaneous firing rate of the neuron, defined as the positive part of ``mp``. More details on the difference between parameters and variables, as well as details on the mathematical parser are to be found in the sections :doc:`Parser` and :doc:`RateNeuron`. 

The synapse type between the two populations will implement a simple Oja learning rule, which is a Hebbian learning rule with an additional regularization term:

.. code-block:: python

    Oja = Synapse(
        parameters="""
            tau = 5000.0
            alpha = 8.0
        """, 
        equations = """
            tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w
        """
    )

``w`` represents the synaptic efficiency (or weight value). Its evolution over time depends on a time constant ``tau``, the regularization parameter ``alpha``, the pre-synaptic firing rate ``pre.r`` and the post-synaptic firing rate ``post.r``. See :doc:`RateSynapse` for more details.

Once these objects are defined, the populations can be created (section :doc:`Populations`). We create here two populations ``pop1`` and ``pop2`` containing 100 neurons each and using the ``LeakyIntegratorNeuron`` neural model:

.. code-block:: python

    pop1 = Population(name='pop1', geometry=100, neuron=LeakyIntegratorNeuron)
    pop2 = Population(name='pop2', geometry=100, neuron=LeakyIntegratorNeuron)

We additionally define an excitatory projection between the neurons of ``pop1`` and ``pop2``, with a target ``exc`` and a all_to_all connection pattern (section :doc:`Projections`). The synaptic weights are initialized randomly between 0.0 and 1.0:

.. code-block:: python

    proj = Projection(pre=pop1, post=pop2, target='exc', synapse=Oja)
    proj.connect_all_to_all(weights = Uniform(0.0, 1.0))

Now that the structure of the network is defined, it can be analysed to generate optimized C++ code in the ``annarchy/`` subfolder and create the objects:

.. code-block:: python

    compile()

The network is now ready to be simulated for the desired amount of time:

.. code-block:: python
    
    simulate(1000.0) # simulate for 1 second
