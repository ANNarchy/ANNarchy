**********************************
Module ANNarchy
**********************************

This the main module of ANNarchy, which contains both Python Code and wrapping of the C++ library. 

First level functions
=================================================

ANNarchy provides a set of first level functions. Its **important** to call these functions in the order they appear in this section. 

ANNarchy.setup()
-------------------------------------------------

Before using any other function or classes of ANNarchy4 the user must call the `setup()` in order to parameterize global parameters.

    .. autofunction:: ANNarchy.setup

ANNarchy.compile()
-------------------------------------------------

The goal of this function is to generate all needed classes and then compile all C++ sources and cython wrappes needed to represent the network. Through this process two libraries will created: ``ANNarchyCore`` contains the c++ core functions and ``ANNarchyCython`` contains the cython extensions.

    .. autofunction:: ANNarchy.compile
    
.. warning:: the `cpp_stand_alone` argument is an experimental feature, mainly used for internal debugging.

Network interaction via global functions
================================================

Next to read or write of the variables / parameters existing a network ANNarchy provides some additional functions available after ``ANNarchy.compile``.

ANNarchy.simulate()
------------------------------------------------
    
After calling ``ANNarchy.compile()`` and a successful compilation process, you may run the simulation with:

.. code-block:: python
    
    ANNarchy.simulate( 100 ) # simulate 100 time steps
    
and access to all objects within your network. Complete examples could be found in the section :doc:`../Example`.

ANNarchy.step()
------------------------------------------------
    
This function can also be called to perform a single simulation step:

.. code-block:: python
    
    ANNarchy.step() # simulate 1 time steps
    

ANNarchy.reset()
-------------------------------------------------

For instance you may run multiple experiments with one network or your experiment setup requires a pre learning phase. In this case you run the simulation several steps and then reset either populations or connections to their initial values. 

    .. autofunction:: ANNarchy.reset

ANNarchy.record()
-------------------------------------------------

    .. autofunction:: ANNarchy.record


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


Available classes
====================================

The ANNarchy simulation environment provides a set of several classes. In the following sections, you'll have access to the documentation of every class, method or attribute accessible in Python.

.. toctree::
    :maxdepth: 2

    Network.rst
    Neuron.rst
    Synapse.rst
    Population.rst
    Projections.rst
    Dendrite.rst
    RandomDistribution.rst
