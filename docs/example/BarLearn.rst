************************************
Bar learning problem
************************************

The implementation of the bar learning problem is located in the ``examples/bar_learn`` folder. The bar learning problem describes the process of learning receptive fields on an artificial input pattern. Therefore images consisting of independent bars are used. Those images are generated as following: an eight by eight image is filled by eight horizontal bars and respectively eight vertical bars. Each bar can occur independently with the probability of 1/8. As learning result the single neurons of this neuronal network should learn to respond to these components.

You can simply try the network by typing:

.. code-block:: python

    python BarLearn.py
    
Model overview
------------------------

The model consists of two populations ``Input`` and ``Feature``. The size of ``Input`` could be chosen as you wish (in our example 8*8). The neuron amount in the ``Feature`` population should be higher then the amount of independent bars, which could appear (in our sample 32 neurons). The ``Feature`` population got excitory connections to ``Input`` by an all-to-all connection pattern. The same pattern is used for the inhibitory connections.

Defining the neurons
------------------------

Sequently the implementation and the corresponding equations are shown. After each definition, the Population instantiation code for the three populations Input, L1 and L2 is given.

    * *Input*: 

        The input pattern will be set by the main loop for every trial, so we need just an empty neuron at this point:
    
            .. code-block:: python
            
                InputNeuron = RateNeuron(
                    parameters=""" 
                        tau = 10.0 : population
                        baseline = 0.0 
                    """,
                    equations="""
                        tau * drate/dt + rate = baseline : min=0.0
                    """
                )

        which is used in the ``Input`` population:
        
            .. code-block:: python
            
                input_pop = Population(geometry=(8, 8), neuron=InputNeuron)
        
    * *Feature*: 

        The neuron add up all the excitory inputs gain from Input and substract the own lateral inhibition (as only positives weights 
        are allowed on this connections). As in the previous equation we also restrict the firing to positive values.

        .. math::
            
            \tau \frac {dr_{j}^{Feature}}{dt} &= \sum r_{i}^{Input} * w_{ij} - \sum_{k, k \ne j} w_{kj} * r_{k}^{Feature} - r_{j}^{Feature}

        could be implemented as the following:

            .. code-block:: python

                LeakyNeuron = RateNeuron(
                    parameters=""" 
                        tau = 10.0 : population
                    """,
                    equations="""
                        tau * drate/dt + rate = sum(exc) - sum(inh)
                    """
                )

        Additionally we want to restrict the fire rate, allowing only positive values. For this we modify the above code like the following:

            .. code-block:: python
        
                LeakyNeuron = RateNeuron(
                    parameters=""" 
                        tau = 10.0 : population
                    """,
                    equations="""
                        tau * drate/dt + rate = sum(exc) - sum(inh) : min=0.0
                    """
                )

        The population is created in the following way:
        
            feature_pop = Population(geometry=(8, 4), neuron=LeakyNeuron)

Defining the synapses
------------------------

As in the previous section, we describe the learning rules as equation and there implementation. After each implementation the instantiation of the 
projection is shown.

    * *Oja*: 

        Implementation of the oja learning, applied on the excitory connections between Input and Fature.

            .. math::
                
                \tau \frac{dw_{ij}^{L1}}{dt} &= r_{j} * r_{i} - \alpha * r_{j}^{2} * w_{ij}
        
        could be realized as:
        
            .. code-block:: python
            
                Oja = RateSynapse(
                    parameters=""" 
                        tau = 2000.0 : postsynaptic
                        alpha = 8.0 : postsynaptic
                    """,
                    equations="""
                        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value
                    """
                )  

    * *Anit-Hebb*: 

        defines the change of inhibitory weights within the Feature population. Additionally we want to restrict the weight, allowing only positive 
        values.

            .. math::
                
                \tau \frac{dw_{ij}^{L1}}{dt} &= r_{j} * r_{i} - \alpha * r_{j} * w_{ij}

        could be implemented as the following:

            .. code-block:: python
    
                AntiHebb = RateSynapse(
                    parameters=""" 
                        tau = 2000.0 : postsynaptic
                        alpha = 0.3 : postsynaptic
                    """,
                    equations="""
                        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min = 0.0
                    """
                )  

Create projections
------------------------

    For this network we need to create two projections, one excitory between the populations Input and Feature and one inhibitory within the 
    population itself:
    
        .. code-block:: python

            input_feature = Projection(
                pre=input_pop, 
                post=feature_pop, 
                target='exc', 
                synapse = Oja    
            ).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
                                 
            feature_feature = Projection(
                pre=feature_pop, 
                post=feature_pop, 
                target='inh', 
                synapse = AntiHebb
            ).connect_all_to_all( weights = Uniform(0.0, 1.0) )

Running the Simulation
------------------------

As explained in the section `Starting the Simulation <manual/Simulation.html#starting-the-simulation>`_, one has the choice between running the simulation in interactive or scripting mode.

In scripting mode (``python BarLearning.py``), it is good practice to put all simulation-related commands under the ``if __name__ == "__main__":`` part of the script (executed only in scripting mode, not interactive):

    .. code-block:: python

        # neuron, population definition as above mentioned
        # ...

        # Analyse and compile everything, initialize the parameters/variables...
        compile()    

        if __name__ == "__main__":

            for trial in range(50000):
                bars = np.zeros((8,8))
                
                # appears a vertical bar?
                for i in xrange(8):
                    if np.random.rand(1) < 1.0/8.0:
                       bars[:,i] = 1.0

                # appears a horizontal bar?
                for i in xrange(8):
                    if np.random.rand(1) < 1.0/8.0:
                       bars[i,:] = 1.0

                InputPop.rate = bars.reshape(8*8)
                
                simulate(100)
        
The ``compile()`` creates all needed sources, compile them and build up the network objects and starts the simulation describect in the ``main function`` block. We present over 50.000 trials a new created input image (``bars``), which is set to the network model through:

    .. code-block:: python
    
            InputPop.rate = bars.reshape(8*8)

After this we simulate 100 timesteps with:

    .. code-block:: python

        InputPop.rate = bars.reshape(8*8)
