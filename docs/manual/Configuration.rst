##############################
Configuration
##############################

Once the script is ready, it can be directly executed in the console::

    $ python MyNetwork.py

or in the interactive mode::

    $ python
    Python 2.7.5 (default, Feb 19 2014, 13:47:28) 
    [GCC 4.8.2 20131212 (Red Hat 4.8.2-7)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from MyNetwork import *
    ANNarchy 4.5 (4.5.4) on linux2 (posix). 
    >>>

You can also open an iPython notebook::

    $ ipython notebook

and import ANNarchy from here.

Setting the discretization step
--------------------------------

An important value for the simulation is the discretization step ``dt``. Its default value is 1 ms, which is usually fine for rate-coded networks, but may be too high for spiking networks, as the equations are stiffer. Taken too high, it can lead to high numerical errors. Too low, and the simulation will take an unnecessary amount of time.

To set the discretization step, just pass the desired value to ``setup()`` at the beginning of the script, or at any rate before the call to ``compile()``::

    setup(dt=0.1)

Changing its value after calling ``compile()`` will not have any effect. 

Setting the seed of the random number generators
-------------------------------------------------

By default, the random number generators are seeded with ``time(NULL)``, so each simulation will be different. If you want to have deterministic simulations, you simply need to provide a fixed seed to ``setup()``::

    setup(seed=62756)

Note that this also sets the seed of Numpy, so you can also reproduce random initialization values prduced by ``numpy.random``. 

Cleaning the compilation directory
-----------------------------------

When calling ``compile()`` for the first time, a subfolder ``annarchy/`` will be created in the current directory, where the generated code will be compiled. The first compilation may last a couple of seconds, but further modifications to the script are much faster. If no modification to the network has been made except for parameter values, it will not be recompiled, sparing us this overhead.

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

First of all, please note, that the CUDA paradigm is implemented for simulation of *rate-coded* neural networks. To run your network on GPUs you need to state to ANNarchy that you want to use CUDA as paradigm:

.. code-block:: python

    from ANNarchy import *
    setup(paradigm="cuda")

Currently two versions of the CUDA paradigm are provided:
    
    * 2.0 and later ( Fermi cards )
    * 3.5 and later ( Keplar cards )

You can check the version of your card on the official website: https://developer.nvidia.com/cuda-gpus

.. hint::

    As the current implementation is a development version, some of the features provided by ANNarchy are not supported yet:
    
        * spiking networks
        * weight sharing
        * non-uniform synaptic delays
        * structural plasticity