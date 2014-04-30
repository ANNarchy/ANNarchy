**************************
Neural Fields
**************************

In ``examples/neural_field`` is a simple model using `Neural Fields <http://www.scholarpedia.org/article/Neural_fields>`_. It consists of two 2D populations ``input`` and ``focus``, with one-to-one connections between ``input`` and ``focus``, and difference-of-Gaussians (dog) lateral connections within ``focus``.

You can simply try the network by typing::

    python NeuralField.py
    
    
.. image:: ../_static/neuralfield.png
    :align: center
    :width: 70%
    
Model overview
--------------------
    
Each population consists of N*N neurons, with N=20. The ``input`` population is solely used to represent inputs for ``focus``. The firing rate of each neuron is defined by a simple equation:

.. math::
    
    \text{input}_i(t) = (\text{baseline}_i(t) + \eta(t))^+
    
where :math:`\text{input}_i(t)` is the instantaneous firing rate, :math:`\text{baseline}_i(t)` its baseline activity, :math:`\eta(t)` an additive noise uniformly taken in :math:`[-0.5, 0.5]` and :math:`()^+` the positive function. 

The ``focus`` population implements a discretized neural field, with neurons following the ODE:

.. math::

    \tau \frac{d \text{mp}_i(t)}{dt} + \text{mp}_i(t) = \text{input}_i(t) + \sum_{j=1}^{N} w_{i, j} \cdot \text{focus}_j(t) + \eta(t)
    
    \text{focus}_i(t) = f(\text{mp}_i(t))
    
where :math:`\text{focus}_i(t)` is the neuron's firing rate, :math:`\text{mp}_i(t)` its membrane potential, :math:`\tau` a time constant and :math:`w_{i, j}` the weight value (synaptic efficiency) of the synapse between the neurons j and i. :math:`f()` is a semi-linear function, ensuring the firing rate is bounded between 0 and 1.

Each neuron in ``focus`` takes inputs from the neuron of ``input`` which has the same index (or rank), leading to a ``one2one`` connection pattern.

The lateral connections within ``focus`` follow a difference-of-Gaussians (``dog``) connection pattern, with the connection weights :math:`w_{i,j}` depending on the normalized euclidian distance between the neurons in the N*N population:

.. math:: 

    w_{i, j} = A^+ \cdot \exp(-\frac{1}{2}\frac{d(i, j)^2}{\sigma_+^2}) -  A^- \cdot \exp(-\frac{1}{2}\frac{d(i, j)^2}{\sigma_-^2})

If i and j have coordinates :math:`(x_i, y_i)` and :math:`(x_j, y_j)` in the N*N space, the distance between them is computed as:

.. math::

    d(i, j)^2 = (\frac{x_i - x_j}{N})^2 + (\frac{y_i - y_j}{N})^2
    
Inputs are given to the network by changing the baseline of ``input`` neurons. This example clamps one or several gaussian profiles (called "bubbles") with an additive noise, moving along a circular path at a certain speed (launch the example to understand this sentence...).

Starting the script
-------------------

The beginning of the script solely consists of importing the ANNarchy library and setting up the discretization step ``dt``:

.. code-block:: python

    from ANNarchy import *
    
    setup(dt=1.0)
    
Note that ``dt=1.0`` is already the default and the call to ``setup`` could have been skipped.

Defining the neurons
--------------------------

There are two different equations for the neurons, so we need to define two **Neuron** objects: ``InputNeuron`` and ``NeuralFieldNeuron`` (for example). 


**InputNeuron**

``InputNeuron`` is straightforward, as ``baseline`` is an external input and the equation for the firing rate is regular:

.. code-block:: python

    InputNeuron = RateNeuron(   
        parameters="""
            baseline = 0.0
        """,
        equations="""
            noise = Uniform(-0.5, 0.5)
            rate = pos(baseline + noise)
        """ 
    )
    
``InputNeuron`` is here an instance of ``RateNeuron``, whose only parameter is ``baseline`` (initialized to 0.0, but it does not matter here). ``noise`` is a random number generator, taken from a uniform distribution between -0.5 and 0.5, whose value is randomly chosen at each computational step for each neuron. ``rate``, the only required variable, is simply the positive part of the sum of ``baseline`` and ``noise``. ``pos()`` is a built-in function of ANNarchy



**NeuralFieldNeuron**

The second neuron we need is a bit more complex, as it is governed by an ODE and considers inputs from other neurons. It also has a non-linear activation function, which is linear when the membrane potential is between 0.0 and 1.0, and constant otherwise. 

.. code-block:: python

    NeuralFieldNeuron = RateNeuron(
        parameters=""" 
            tau = 10.0 : population
        """,
        equations="""
            noise = Uniform(-0.5, 0.5)
            tau * dmp / dt + mp = sum(exc) + sum(inh) + noise
            rate = if mp < 1.0 : pos(mp) else: 1.0 
        """
    )
    
``tau`` is a population-wise parameter, whose value will be the same for all neuron of the population. ``noise`` is a random number generator. ``mp`` is the membrane potential, whose dynamics are governed by a first-order linear ODE, integrating the sums of excitatory and inhibitory inputs with noise. As explained in the section `Defining a Neuron <../manual/Neuron.html>`_, ``sum(exc)`` retrieves the weighted sum of presynaptic firing rates for the synapses having the connection type *exc*, here the one2one connections between ``input`` and ``focus``. ``sum(inh)`` does the same for *inh* type connections, here the lateral connections within ``focus``.

``rate`` is defined by a piecewise linear function of ``mp``, making sure that it is bounded between 0.0 and 1.0. The function is defined by the conjunction of a conditional statement (if-then-else) and the ``pos()`` positive function.

Creating the populations
--------------------------------

The two populations  have a geometry of (20, 20), therefore 400 neurons each. They are created simply by instantiating the ``Population`` class:

.. code-block:: python

    InputPop = Population(name = 'Input', geometry = (20, 20), neuron = InputNeuron)
    FocusPop = Population(name = 'Focus', geometry = (20, 20), neuron = NeuralFieldNeuron)
    
Each population should be assigned a unique name (here 'Input' and 'Focus') in order to be be able to retrieve them if the references ``InputPop`` and ``FocusPop`` are lost. They are given a 2D geometry and associated to the corresponding ``RateNeuron`` instance. 

Creating the projections
------------------------------

The first projection is a one-to-one projection from Input to Focus with the type 'exc'. This connection pattern pattern is possible because the two populations have the same geometry. The weights are initialized to 1.0, and this value will not change with time (no learning), so it is not necessary to define a synapse type:

.. code-block:: python

    input_focus = Projection( 
        pre = InputPop, 
        post = FocusPop, 
        target = 'exc'
    ).connect_one_to_one( weights=1.0 )
    
The refereces to the pre- and post-synaptic population (or their names), as well as the target type, are passed to the constructor of ``Projection``. The connector method ``connect_one_to_one()`` is immediately applied to the Projection, defining how many synapses will be created. The weights are initialized uniformly to 1.0. 

The second projection is a difference of gaussians (DoG) for the lateral connections within 'focus'. The connector method is already provided by ANNarchy, so there is nothing more to do than to call it with the right parameters:

.. code-block:: python

    focus_focus = Projection(
        pre = FocusPop, 
        post = FocusPop, 
        target = 'inh'     
    ).connect_dog(    
        amp_pos=0.2, 
        sigma_pos=0.1, 
        amp_neg=0.1, 
        sigma_neg=0.7                    
    )


Compiling the network and simulating
--------------------------------------

Once the populations and projections are created, the network is ready to be compiled and simulated. Compilation is simply done by calling ``ANNarchy.compile()``:

.. code-block:: python 

    compile()
    
This generates optimized C++ code from the neurons' definition and network structure, compiles it with gcc and instantiates all objects, particularly the synapses. If some errors were made in the neuron definition, they will be signalled at this point.

.. hint::

    The call to ``compile()`` is mandatory in any script. After it is called, populations and projections can not be added anymore.
    
Once the compilation is successful, the network can be simulated by calling ``ANNarchy.simulate()``:

.. code-block:: python 

    simulate(1000.0) # simulate for 1 second
    
As no input has been fed into the network, calling ``simulate()`` now won't lead to anything interesting. The next step is to clamp inputs into the input population's baseline.

Defining the environment
-------------------------

**Pure Python approach**

In this example, we consider as input a moving bubble of activity rotating along a circle in the input space in 5 seconds. A naive way of setting such inputs would be to access population attributes (namely ``InputPop.baseline``) in a tight loop in Python:

.. code-block:: python

    angle = 0.0
    x, y = np.meshgrid(np.linspace(0, 19, 20), np.linspace(0, 19, 20))
    
    # Main loop
    while True:
        # Update the angle
        angle += 1.0/5000.0
        # Compute the center of the bubble
        cx = 10.0 * ( 1.0 + 0.5 * np.cos(2.0 * np.pi * angle ) )
        cy = 10.0 * ( 1.0 + 0.5 * np.sin(2.0 * np.pi * angle ) )
        # Clamp the bubble into pop.baseline
        InputPop.baseline = (np.exp(-((x-cx)**2 + (y-cy)**2)/8.0))
        # Simulate for 1 ms
        step()  
            
``angle`` represents the angle made by the bubble with respect to the center of the input population. ``x`` and ``y`` are Numpy arrays representing the X- and Y- coordinates of neurons in the input population. At each iteration of the simulation (i.e. every millisecond of simulation, the bubble is slightly rotated (``angle`` is incremented) so as to make a complete revolution in 5 seconds (5000 steps). ``cx`` and ``cy`` represent the coordinates of the center of the bubble in neural coordinates according to the new value of the angle.

A Gaussian profile (in the form of a Numpy array) is then clamped into the baseline of ``InputPop`` using the distance between each neuron of the population (``x`` and ``y``) and the center of the bubble. Last, a single simulation step is performed using ``step()``, before the whole process starts again until the user quits. ``step()`` is equivalent to ``simulate(1)``, although a little bit faster as it does not check the type of argument (int or float).

Although this approach works, you would observe that it is very slow: the computation of the bubble and its feeding into ``InputPop`` takes much more time than the call to ``step()``. The interest of using a parallel simulator disappears. This is due to the fact that Python is knowingly bad at performing tight loops because of its interpreted nature. If the ``while`` loop were compiled from C code, the computation would be much more efficient. This is what Cython brings you.
            
            
**Cython approach**

The Cython approach requires to write Cython-specific code in a ``.pyx`` file, generate the corresponding C code with Python access methods, compile it and later import it into your Python code.

Happily:

* the Cython syntax is very close to Python. In the most basic approach, it is simply Python code with a couple of type declarations. Instead of:

.. code-block:: python

    bar = 1
    foo = np.ones((10, 10))
    
you would write in Cython:

.. code-block:: cython

    cdef int bar = 1
    cdef np.ndarray foo = np.ones((10, 10))
    
By specifing the type of a variable (which can not be changed later contrary to Python), you help Cython generate optimized C code, what can lead in some cases to speedups up to 100x. The rest of the syntax (indentation, for loops, if...) is the same as in Python. You can also import any Python module in your Cython code. Some modules (importantly Numpy) even provide a Cython interface where the equivalent Cython code can be directly imported (so it becomes very fast to use).

* the whole compilation procedure is very easy. One particularly simple approach is to use the ``pyximport`` module shipped with Cython. Let us suppose you wrote a ``dummy()`` method in a Cython file named ``TestModule.pyx``. All you need to use this method in your python code is to write:

.. code:: python

    import pyximport; pyximport.install()
    from TestModule import dummy
    dummy()
    
``pyximport`` takes care of the compilation process (but emits quite a lot of warnings), and allows to import ``TestModule`` as if it were a regular Python module. Please refer to the `Cython documentation <http://docs.cython.org/>`_ to know more. 


Running the simulation
----------------------------


    
Visualizing the network
----------------------------

