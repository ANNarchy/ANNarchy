***********************************
Networks
***********************************

A typical ANNarchy script represents a single network of populations and projections. Most of the work in computational neuroscience consists in running the same network again and again, varying some free parameters each time, until the fit to the data is publishable.  The ``reset()`` allows to return the network to its state before compilation, but this is particuarly tedious to implement.

In order to run different networks using the same script, the ``Network`` object can be used to make copies of existing objects (populations, projections and monitors) and simulate them either sequentially or in parallel.

Let's suppose the following dummy network is defined::



Multiple networks
===================


Parallel simulations
=====================