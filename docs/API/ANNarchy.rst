**********************************
Module ANNarchy
**********************************

This the main module of ANNarchy, which contains both Python Code and wrapping of the C++ library. 

Configuration and compilation
=================================================

Contrary to other simulators, ANNarchy is entirely based on code generation. It provides a set of first level functions to ensre the network is correctly created. Its **important** to call these functions in this order. 

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

Next to read or write of the variables / parameters existing a network ANNarchy provides some additional functions available after ``ANNarchy.compile``.

ANNarchy.simulate()
------------------------------------------------
    
After calling ``ANNarchy.compile()`` and a successful compilation process, you may run the simulation with:

.. code-block:: python
    
    ANNarchy.simulate(100.0) # simulate 100 milliseconds
    
and access to all objects within your network. Complete examples could be found in the section :doc:`../Example`.

ANNarchy.step()
------------------------------------------------
    
This function can also be called to perform a single simulation step:

.. code-block:: python
    
    ANNarchy.step() # simulate 1 time steps
    

ANNarchy.reset()
-------------------------------------------------

If you want to run multiple experiments with the same network, or if your experiment setup requires a pre learning phase, you can reset selectively neural or synaptic variables to their initial values. 

.. autofunction:: ANNarchy.reset

Recordings
===========

ANNarchy.start_record()
-------------------------------------------------

.. autofunction:: ANNarchy.start_record

ANNarchy.stop_record()
-------------------------------------------------

.. autofunction:: ANNarchy.stop_record

ANNarchy.get_record()
-------------------------------------------------

.. autofunction:: ANNarchy.get_record

ANNarchy.pause_record()
-------------------------------------------------

.. autofunction:: ANNarchy.pause_record

ANNarchy.resume_record()
-------------------------------------------------

.. autofunction:: ANNarchy.resume_record

Saving/Loading
===============

ANNarchy.save()
-------------------------------------------------

To save the network state you may use the following method:

.. autofunction:: ANNarchy.save
    
Please note that these function is only usable after ANNarchy.compile().

ANNarchy.load()
-------------------------------------------------

To save the network state you may use the following method:

.. autofunction:: ANNarchy.load
    
Please note that these function is only usable after ANNarchy.compile().


