**********************************
Saving / Loading
**********************************


Saving / loading the state of the network
==========================================

To save or load the network state you can use the following methods:

.. autofunction:: ANNarchy.save

.. autofunction:: ANNarchy.load
    
Please note that these functions are only usable after the call to ``ANNarchy.compile()``.


Saving / loading the parameters of the network
===============================================

.. autofunction:: ANNarchy.save_parameters

.. autofunction:: ANNarchy.load_parameters