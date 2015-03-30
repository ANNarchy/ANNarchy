***********************************
Structural plasticity
***********************************

ANNarchy supports the dynamic addition/suppression of synapses during the simulation (i.e. after compilation).   

Because structural plasticity adds some complexity to the generated code, it has to be enabled before compilation by setting the ``structural_plasticity`` flag to ``True`` in the call to ``setup()``:

.. code-block:: python

    setup(structural_plasticity=True)

If the flag is not set, the following methods will do nothing.

There are two possibilities to dynamically create or delete synapses:

* Externally, using methods at the dendrite level from Python.

* Internally, by defining conditions for creating/pruning in the synapse description.


Dendrite level
================

Two methods of the ``Dendrite`` class are available for creating/deleting synapses:

* ``create_synapse()``

* ``prune_synapse()`` 


Creating synapses
------------------

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

After an initial period of simulation, one could add new synapses between strongly active pair of neurons:

.. code-block:: python

    # For all post-synaptic neurons
    for post in xrange(pop2.size):
        # For all pre-synaptic neurons
        for pre in xrange(pop1.size):
            # If the neurons are not connected yet
            if not pre in proj[post].ranks:
                # If they are both sufficientely active
                if pop1[pre].mean_r * pop2[post].mean_r > 0.7:
                    # Add a synapse with weight 1.0 and the default delay
                    proj[post].create_synapse(pre, 1.0)   

``create_synapse`` only allows to specify the value of the weight and the delay. Other syanptic variables will take the value they would have had before compile(). If another value is desired, it should be explicitely set afterwards. 
            
Removing synapses 
-----------------

Removing useless synapses (pruning) is also possible. Let's consider a synapse type whose "age" is incremented as long as both pre- and post-synaptic neurons are inactive at the same time:

.. code-block:: python

    AgingSynapse = Synapse(
        equations="""
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
            if proj[post][pre].age > T :
                # Remove it
                proj[post].prune_synapse(pre)
            
.. warning::

    This form of structural plasticity is rather slow because:

    * The ``for`` loops are in Python, not C++. Implementing this structural plasticity in Cython should already help.

    * The memory allocated for the synapses of a projection may have to be displaced at another location. This can lead to massive transfer of data, slowing the simulation down.
      
    It is of course the user's responsability to balance synapse creation/destruction, otherwise projections could become either empty or fully connected on the long-term.


Synapse level
==============

Conditions for creating or deleting synapses can also be specified in the synapse description, through the ``creating`` or ``pruning`` arguments. Thise arguments accept string descriptions of the boolean conditions at which a synapse should be created/deleted, using the same notation as other arguments.

Creating synapses
------------------

The creation of a synapse must be described by a boolean expression:

.. code-block:: python 

    CreatingSynapse = Synapse(
        parameters = " ... ",
        equations = " ... ",
        creating = "pre.mean_r * post.mean_r > 0.7 : proba = 0.5, w = 1.0"
    )

The condition can make use of any pre- or post-synaptic variable, but NOT synaptic variables, as they obviously do not exist yet. Global parameters (defined with the ``post-synaptic`` flag) can nevertheless be used. 

Several flags can be passed to the expression: 

* ``proba`` specifies the probability according to which a synapse will be created, if the condition is met. The default is 1.0 (i.e. a synapse will be created whenever the condition is fulfilled).

* ``w`` specifies the value for the weight which will be created (default: 0.0).

* ``d`` specifies the delay (default: the same as all other synapses if the delay is constant in the projection, ``dt`` otherwise). 

.. warning::

    Note that the new value for the delay can not exceed the maximal delay in the projection, nor be different from the others if they were all equal.


Other synaptic variables will take the default value after creation.

Synapse creation is not automatically enabled at the start of the simulation: the Projectiom method ``start_creating()`` must be called:

.. code-block:: python
    
    proj.start_creating(period=100.0)

This method accepts a ``period`` parameter specifying how often the conditions for creating synapses will be checked (in ms). By default they would be checked at each time step (``dt``), what would be too costly.

Similarly, the ``stop_creating()`` method can be called to stop the creation conditions from being checked.


Deleting synapses
------------------

Synaptic pruning also rely on a boolean expression: 


.. code-block:: python 

    PruningSynapse = Synapse(
        parameters = " T = 100000 : int, post-synaptic ",
        equations = """
            age = if pre.r * post.r > 0.0 : 
                    0
                  else :
                    age + 1 : init = 0, int""",
        pruning = "age > T : proba = 0.5"
    )

* A synapse type can combine ``creating`` and ``pruning`` arguments.

* The ``pruning`` argument can rely on synaptic variables (here ``age``), as the synapse already exist.

* Only the ``proba`` flag can be passed to specify the probability at which the synapse will be deleted if the condition is met.

* Pruning has to be started/stopped with the ``start_pruning()`` and ``stop_pruning()`` methods. ``start_pruning()`` accepts a ``period`` argument.