***********************************
Saving and loading a network
***********************************

For the complete APIs, see :doc:`../API/IO` in the library reference.

Global parameters
------------------------------

The global parameters of the network (flagged with ``population`` or ``projection`` in the Neuron/Synapse definitions) can be saved to and loaded from a JSON file using the functions ``save_parameters()`` and ``load_parameters()``::

    save_parameters('network.json')
    load_parameters('network.json')

The saved JSON file for a network of two populations of Izhikevich neurons connected with STDP will look like:

.. code-block:: json

    {
        "populations": {
            "pop0": {
                "a": 0.1,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
                "tau_ampa": 5.0,
                "tau_nmda": 150.0,
                "v_rev": 0.0,
                "v_thresh": 30.0
            },
            "pop1": {
                "a": 0.02,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
                "tau_ampa": 5.0,
                "tau_nmda": 150.0,
                "v_rev": 0.0,
                "v_thresh": 30.0
            }
        },
        "projections": {
            "proj0": {
                "tau_plus": 20.0,
                "tau_minus": 60.0,
                "A_plus": 0.0002,
                "A_minus": 6.6e-05,
                "w_max": 0.03
            }
        },
        "network": {},
    }

By default, populations and projections have names like ``pop0`` and ``proj1``. For readability, we advise setting explicit (and **unique**) names in their constructor::

    pop = Population(100, Izhikevich, name="PFC_exc")
    proj = Projection(pop, pop2, 'exc', STDP, name="PFC_exc_to_inh")

Only global parameters can be saved (no array is allowed in the JSON file). By default, only global parameters will be loaded, except if the ``global_only`` argument to ``load_parameters()`` is set to False. In that case, even local parameters can be set by the JSON file, but they will all use the same values. 

If you want to initialize other things than population/projection global parameters, you can define arbitrary values in the ``"network"`` dictionary:

.. code-block:: json

    {
        "network": {
            "pop1_r_min": 0.1,
            "pop1_r_max": 1.3,
        },
    }

``load_parameters()`` will return the corresponding dictionary::

    params = load_parameters('network.json')

You can then use them to initialize programmatically non-global parameters or variables::

    pop1.r = Uniform(params['pop1_r_min'], params['pop1_r_max'])  



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

``load()`` also accepts the ``populations`` and ``projections`` boolean flags (for example if you want to load only the synaptic weights but not to restore the neural variables).

Populations and projections individually
----------------------------------------

``Population`` and ``Projection`` objects also have ``save()`` and ``load()`` methods, allowing to save the corresponding information individually:

.. code-block:: python

    pop1.save('pop1.npz')
    proj.save('proj.npz')

    pop1.load('pop1.npz')
    proj.load('proj.npz')

The allowed file formats are:

* ``.npz``: compressed Numpy binary format (``np.savez_compressed``), preferred.
* ``*.gz``: gunzipped binary text file.
* ``*.mat``: Matlab 7.2.
* ``*``: binary text file.

As before, ``.mat`` can only be used for saving, not loading.

