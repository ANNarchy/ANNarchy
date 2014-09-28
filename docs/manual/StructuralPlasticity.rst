***********************************
Structural plasticity
***********************************

ANNarchy supports the dynamic addition/suppression of synapses during the simulation (i.e. after compilation). Two methods of the ``Dendrite`` class are available for this:

* ``add_synapse()``

* ``remove_synapse()``   

Because structural plasticity adds some complexity to the generated code, it has to be enabled before compilation by setting the ``structural_plasticity`` flag to ``True`` in the call to ``setup()``:

.. code-block:: python

    setup(structural_plasticity=True)

If the flag is not set, the methods will do nothing.


Creating synapses
==================

Let's suppose that we want to add regularly new synapses between strongly active but not yet connected neurons with a low probability. One could for example define a neuron type with an additional variable averaging the firing rate over a long period of time.

.. code-block:: python

    LeakyIntegratorNeuron = Neuron(
        parameters="""   
            tau = 10.0
            baseline = -0.2
            tau_mean = 100000.0
        """,
        equations = """
            tau * dmp/dt + mp = baseline + sum(exc)
            r = pos(mp)
            tau_mean * dmean_r/dt =  (r - mean_r) : init = 0.0
        """
    )

Two populations are created and connected using a sparse connectivity:

.. code-block:: python

    pop1 = Population(1000, LeakyIntegratorNeuron)
    pop2 = Population(1000, LeakyIntegratorNeuron)
    proj = Projection(pop1, pop2, 'exc', Oja).connect_fixed_probability(weights = 1.0, probability=0.1)

After a long piece of simulation:

.. code-block:: python

    simulate(100000.0) # Simulate 100s

one could randomly add new synapses between strongly active neurons:

.. code-block:: python

    import random
    # For all postsynaptic neurons
    for post in xrange(pop2.size):
        # For all presynaptic neurons
        for pre in xrange(pop1.size):
            # If the neurons are not connected yet
            if not pre in proj[post].ranks:
                # If they are both sufficientely active
                if random.random() < pop1[pre].mean_r * pop2[post].mean_r :
                    # Add a synapse with weight 1.0 and the default delay
                    proj[post].add_synapse(pre, 1.0)    
            
Removing synapses 
==================

Removing useless synapses (pruning) is also possible. Let's consider a synapse type whose "age" is incremented as long as both pre- and post-synaptic neurons are inactive at the same time:

.. code-block:: python

    Oja = Synapse(
        parameters="""
            tau = 5000
            alpha = 8.0
        """,
        equations="""
            tau * dw / dt = pre.r * post.r - alpha * post.r^2 * w
            age = if pre.r * post.r > 0.0 : 
                    0
                  else :
                    age + 1 : init = 0, int
        """
    )

One could periodically track the too "old" synapses and remove them:

.. code-block:: python

    # Threshold on the age:
    T = 100000
    # For all post-synaptic neurons receiving synapses
    for post in proj.post_ranks:
        # For all existing synapses
        for pre in proj[post].ranks:
            # If the synapse is too old
            if proj[post].age[pre] > T :
                # Remove it
                proj[post].remove_synapse(pre)
            
.. warning::

    Structural plasticity is rather slow because:

    * The ``for`` loops are in Python, not C++. Implementing structural plasticity in Cython should already help.
    * The internal structure of ANNarchy allows for an efficient allocation/desallocation of synapses within a margin of 5% compared  to the initial number of synapses. Above this threshold, it can lead to massive transfer of data, slowing the simulation down.
      
    It is of course the user's responsability to balance synapse creation/destruction, otherwise projections could become either empty or fully connected on the long-term.