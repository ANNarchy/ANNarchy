************************************
Bar learning problem
************************************

The implementation of the bar learning problem is located in the ``examples/bar_learn`` folder. The bar learning problem describes the process of learning receptive fields on an artificial input pattern. Images consisting of independent bars are used. Those images are generated as following: an 8*8 image can filled randomly by eight horizontal or vertical bars, with a probability of 1/8 for each. 

These input images are fed into a neural population, whose neurons should learn to extract the independent components of the input distribution, namely single horizontal or vertical bars.

If you have ``pyqtgraph`` installed, you can simply try the network by typing:

.. code-block:: python

    python BarLearn.py
    
Model overview
------------------------

The model consists of two populations ``Input`` and ``Feature``. The size of ``Input`` should be chosen  to fit the input image size (here 8*8). The number of neurons in the ``Feature`` population should be higher than the total number of independent bars  (16, we choose here 32 neurons). The ``Feature`` population gets excitory connections from ``Input`` through an all-to-all connection pattern. The same pattern is used for the inhibitory connections within ``Feature``.

Defining the neurons
------------------------


* *Input* population: 

    The input pattern will be clamped into this population by the main loop for every trial, so we need just an empty neuron at this point:

.. code-block:: python

    InputNeuron = RateNeuron(   
        parameters="""
            r = 0.0
        """
    )
    
The trick here is to declare ``r`` as a parameter, not a variable: its value will not be computed by the simulator, but only set by external input. The ``Input`` population can then be created:
    
.. code-block:: python

    input_pop = Population(geometry=(8, 8), neuron=InputNeuron)
        
* *Feature* population: 

The neuron type composing this population sums up all the excitory inputs gain from ``Input`` and the lateral inhibition within ``Feature``.

.. math::
    
    \tau \frac {dr_{j}^{\text{Feature}}}{dt} + r_{j}^{Feature} = \sum_{i} w_{ij} \cdot r_{i}^{\text{Input}}  - \sum_{k, k \ne j} w_{kj} * r_{k}^{Feature} 

could be implemented as the following:

.. code-block:: python

    LeakyNeuron = RateNeuron(
        parameters=""" 
            tau = 10.0 : population
        """,
        equations="""
            tau * dr/dt + r = sum(exc) - sum(inh) : min=0.0
        """
    )

The firing rate is restricted to positive values with the ``min=0.0`` flag. The population is created in the following way:

.. code-block:: python
        
    feature_pop = Population(geometry=(8, 4), neuron=LeakyNeuron)
    
We give it a (8, 4) geometry for visualization only, it does not influence computations at all.

Defining the synapses
------------------------

Both feedforward (``Input`` :math:`\rightarrow` ``Feature``) and lateral (``Feature`` :math:`\rightarrow` ``Feature``) projections are learned using the Oja learning rule (a regularized Hebbian learning rule ensuring the sum of all weights coming to a neuron is constant). Only parameters will differ between the projections.

.. math::
                
    \tau \frac{dw_{ij}}{dt} &= r_{i} * r_{j} - \alpha * r_{j}^{2} * w_{ij}
        
where :math:`\alpha` is a parameter defining the strength of the regularization, :math:`r_i` is the pre-synaptic firing rate and :math:`r_j` the post-synaptic one. The implementation of this synapse type is straightforward:
        

.. code-block:: python

    Oja = RateSynapse(
        parameters=""" 
            tau = 2000.0 : post-synaptic
            alpha = 8.0 : post-synaptic
            min_w = 0.0 : post-synaptic
        """,
        equations="""
            tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=min_w
        """
    )  
 

Create the projections
------------------------

For this network we need to create two projections, one excitory between the populations ``Input`` and ``Feature`` and one inhibitory within the ``Feature`` population itself:
    
.. code-block:: python

    Input_Feature = Projection(
        pre=Input, 
        post=Feature, 
        target='exc', 
        synapse = Oja    
    ).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
                         
    Feature_Feature = Projection(
        pre=Feature, 
        post=Feature, 
        target='inh', 
        synapse = Oja
    ).connect_all_to_all( weights = Uniform(0.0, 1.0) )

The two projections are all-to-all and use the ``Oja`` synapse type. They only differ by the parameter ``alpha`` (lower in ``Feature_Feature``) and the fact that the weights of ``Input_Feature`` are allowed to be negative (so we set the minimum value to -10.0):

.. code-block:: python   

    Input_Feature.min_w = -10.0
    Feature_Feature.alpha = 0.3

Setting inputs
------------------

Once the network is defined, one has to specify how inputs are fed into the ``Input`` population. A simple solution is to define a method that sets the firing rate of ``Input`` according to the specified probabilities every time it is called:

.. code-block:: python

    def set_input():
        # Reset the firing rate for all neurons
        Input.r = 0.0
        # Clamp horizontal bars
        for h in range(Input.geometry[0]):
            if np.random.random() < 1.0/ float(Input.geometry[0]):
                Input[h, :].r = 1.0
        # Clamp vertical bars
        for w in range(Input.geometry[1]):
            if np.random.random() < 1.0/ float(Input.geometry[1]):
                Input[:, w].r = 1.0
                
This method starts by resetting the firing rate of ``input`` to 0.0:

.. code-block:: python

    Input.r = 0.0

One can use here a single value or a Numpy array (e.g. ``np.zeros(Input.geometry))``), it does not matter.

For all possible horizontal bars, a decision is then made whether the bar should appear or not, in which case the firing rate of the correspondng neurons is set to 1.0:

.. code-block:: python

    for h in range(Input.geometry[0]):
        if np.random.random() < 1.0/ float(Input.geometry[0]):
            Input[h, :].r = 1.0
            
``Input[h, :]`` is a PopulationView, i.e. a group of neurons defined by the sub-indices (here the row of index ``h``). Their attributes, such as ``r``, can be accessed as if it were a regular population. The same is done for vertical bars.
                
                

Running the Simulation
------------------------

Once the method for setting inputs is defined, the simulation can be started. A basic approach would be to define an infinite loop where the inputs are first set, and the network is simulated for 50 milliseconds afterwards:

.. code-block:: python

    compile()
    
    while True:
        set_input()
        simulate(50)
        
In the file ``BarLearning.py``, a visualization class using pyqtgraph is used, but the user is free to use whatever method he prefers to visualize the result of learning.
