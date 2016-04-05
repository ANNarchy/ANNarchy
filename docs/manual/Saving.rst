***********************************
Saving and loading a network
***********************************

Complete state of the network
------------------------------

The state of all variables, including the synaptic weights, can be saved in a text file, compressed binary file or Matlab file using the ``save()`` method:

.. code-block:: python

    save('data.txt')
    save('data.txt.gz')
    save('data.mat')

Filenames ending with '.mat' correspond to Matlab files (it requires the installation of Scipy), filenames ending with '.gz' are compressed using gzip (normally standard to all Python distributions, but may require installation), other extensions are normal text files using cPickle (standard). 

``save()`` also accepts the ``populations`` and ``projections`` boolean flags. If ``True`` (the default), the neural resp. synaptic variables will be saved. For example, if you only care about synaptic plasticity but not the neural variables, you can set ``populations`` to ``False``, and only synaptic variables will be saved. 

.. code-block:: python

    save('data.txt', populations=False)

Except for the Matlab format, you can also load the state of variables stored in these files **once the network is compiled**:

.. code-block:: python

    load('data.txt')

.. warning::

    The structure of the network must of course be the same as when the file was saved: number of populations, neurons and projections. The neuron and synapse types must define the same variables. If a variable was saved but does not exist anymore, it will be skipped. If the variable did not exist, its current value will be kept, what can lead to crashes.

``load()`` also accepts the ``populations`` and ``projections`` boolean flags (for example if you want to load only the synaptic weights but not restore the neural variables).

Populations and projections individually
----------------------------------------

``Population`` and ``Projection`` objects also have ``save()`` and ``load()``, allowing to save only the interesting information:

.. code-block:: python

    pop1.save('pop1.txt')
    proj.save('proj.txt')

    pop1.load('pop1.txt')
    proj.load('proj.txt')

The same file formats are allowed (Matlad data can not be loaded).