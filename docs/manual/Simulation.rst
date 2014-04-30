***********************************
Simulating the network
***********************************

Running the simulation
===================================

The python interpreter allows to usage modes: interactive and scripting mode, both supported by ANNarchy. In the interpreter it prompts for the next command with the primary prompt, usually three greater-than signs (>>>); for continuation lines it prompts with the secondary prompt, by default three dots (...). The interpreter prints a welcome message stating its version number and a copyright notice before printing the first prompt::

        user@machine:~$ python
        Python 2.7.3 (default, Apr 10 2013, 06:20:15)
        [GCC 4.6.3] on linux 2
        Type "help", "copyright", "credits" or "license" for more information.
        >>>
        
Otherwise you provide a ANNarchy script as argument on python and use the scripting mode::

    user@machine:~$ python NeuralField.py

In most cases you may use the scripting mode, only for debug purposes the interactive mode is more beneficial.

Interactive mode
-----------------------

First you need to import your network:

.. code-block:: python

    >>> from NeuralField import *
    >>> compile()

If you are in interactive mode, you may want to keep the interpreter accessible while the simulation is running, in order to change some parameters live. You can use the method ``simulate()`` of **ANNarchy**:

.. code-block:: python

    >>> simulate(10000) # Simulate 10s
    
The simulation is now running in the background for 10000 steps and then return to prompt. At any point you can break the simulation with the key combination: ``Ctrl + C``. You can then change some parameters, plot some results and so on, and maybe start the simulation again.

Parallel computing with OpenMP
-------------------------------

The default paradigm for an ANNarchy simulation is through openMP, which distributes automatically the computations over the available CPU cores.

By default, OpenMP uses all the available cores for your simulation, even if it is not optimal: small networks in particular tend to run faster with a small amount of cores (for the provided example with Neural Fields, it is for example 2). You can control the number of cores by passing the ``OMP_NUM_THREADS`` environment variable to the command line before the call to Python::

    user@machine:~$ OMP_NUM_THREADS=2 python NeuralField.py
    
or by passing the ``-j`` flag::

    user@machine:~$ python NeuralField.py -j2
    
It is the responsability of the user to find out which number of cores is optimal for his network, by comparing simulation times. When this optimal number is found, it can be hard-coded in the script by setting the ``num_threads`` argument to ``ANNarchy.setup()``:

.. code-block:: python

    from ANNarchy import *
    setup(num_threads=2)


Parallel computing with CUDA
-------------------------------

TODO: not implemented yet, planned in version 4.2.


Recording variables during the simulation
==============================================

TODO
