**********************************
Module ANNarchy
**********************************

This the main module of ANNarchy, which contains both Python Code and wrapping of the C++ library. 

Configuration and compilation
=================================================

Contrary to other simulators, ANNarchy is entirely based on code generation. It provides a set of first level functions to ensre the network is correctly created. It is **important** to call these functions in the right order. 

ANNarchy.setup()
-------------------------------------------------

Before using any other function or classes of ANNarchy4 the user must call the ``setup()`` method in order to define global parameters.

.. autofunction:: ANNarchy.setup

ANNarchy.compile()
-------------------------------------------------

The goal of this function is to generate all needed classes, compile all C++ sources and Cython wrappers needed to represent the network.

.. autofunction:: ANNarchy.compile
    
.. warning:: the `cpp_stand_alone` argument is an experimental feature, mainly used for internal debugging.

Simulation
================================================

Different methods are available to run the simulation:


.. autofunction:: ANNarchy.simulate

.. autofunction:: ANNarchy.simulate_until

.. autofunction:: ANNarchy.step

.. autoclass:: ANNarchy.every

.. autofunction:: ANNarchy.enable_callbacks

.. autofunction:: ANNarchy.disable_callbacks


Reset the network
=====================

If you want to run multiple experiments with the same network, or if your experiment setup requires a pre learning phase, you can reset selectively neural or synaptic variables to their initial values. 

.. autofunction:: ANNarchy.reset

Access to populations
=====================

.. autofunction:: ANNarchy.get_population

Learning
=====================

.. autofunction:: ANNarchy.enable_learning

.. autofunction:: ANNarchy.disable_learning


Saving/Loading
===============

To save or load the network state you can use the following methods:

.. autofunction:: ANNarchy.save

.. autofunction:: ANNarchy.load
    
Please note that these functions are only usable after the call to ``ANNarchy.compile()``.


Access to simulation times
===========================

.. autofunction:: ANNarchy.get_time
.. autofunction:: ANNarchy.set_time
.. autofunction:: ANNarchy.get_current_step
.. autofunction:: ANNarchy.set_current_step
.. autofunction:: ANNarchy.dt

