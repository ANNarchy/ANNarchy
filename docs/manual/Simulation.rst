***********************************
Simulation
***********************************

Compiling the network
=====================

Once all the relevant information has been defined, one needs to actually compile the network, by calling the ``ANNarchy.compile()`` method:

.. code-block:: python

    compile()
    
The optimized C++ code will be generated in the ``annarchy/`` subfolder relative to your script, compiled, the underlying objects created and made available to the Python interface.

Simulating the network
======================

After the network is correctly compiled, the simulation can be run for the specified duration (in milliseconds) through the ``ANNarchy.simulate()`` method:

.. code-block:: python

    simulate(1000.0) # Simulate for 1 second

The provided duration should be a multiple of ``dt``. If not, the number of simulation steps performed will be approximated.

In some cases, you may want to perform only one step of the simulation, instead of specifing the duration. The ``ANNarchy.step()`` can then be used.

.. code-block:: python

    step() # Simulate for 1 step

Running the simulation
===================================

The resulting script can be directly executed in the console::

    $ python MyNetwork.py

or in the interactive mode::

    $ python
    Python 2.7.5 (default, Feb 19 2014, 13:47:28) 
    [GCC 4.8.2 20131212 (Red Hat 4.8.2-7)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from MyNetwork import *
    ANNarchy 4.1 (4.1.3) on linux2 (posix). 
    >>>


Cleaning the compilation directory
-----------------------------------

When calling ``compile()`` for the first time, a subfolder ``annarchy/`` will be created in the current directory, where the generated code will be compiled. The first compilation may last a couple of seconds, but further modifications to the script are much faster. If no modification to the network has been made, it will not be recompiled, saving this overhead.

ANNarchy tracks the changes in the script and re-generates the corresponding code. In some cases (a new version of ANNarchy has been installed, bugs), it may be necessary to perform a fresh compilation of the network (for example you get a segmentation fault). You can either delete the ``annarchy/`` subfolder and restart the script::

    $ rm -rf annarchy/
    $ python MyNetwork.py

or pass the ``--clean`` flag to Python::

    $ python MyNetwork.py --clean 


Parallel computing with OpenMP
-------------------------------

The default paradigm for an ANNarchy simulation is through openMP, which distributes automatically the computations over the available CPU cores.

By default, OpenMP would use all the available cores for your simulation, even if it is not optimal: small networks in particular tend to run faster with a small amount of cores (for the provided example with Neural Fields, it is for example 2). 
For this reason, the ``OMP_NUM_THREADS`` environment variable has no effect in ANNarchy. You can control the number of cores by passing  the ``-j`` flag to the Python command::

    user@machine:~$ python NeuralField.py -j2
    
It is the responsability of the user to find out which number of cores is optimal for his network, by comparing simulation times. When this optimal number is found, it can be hard-coded in the script by setting the ``num_threads`` argument to ``ANNarchy.setup()``:

.. code-block:: python

    from ANNarchy import *
    setup(num_threads=2)


Parallel computing with CUDA
-------------------------------

TODO: not implemented yet, planned in version 4.2.


