*******************************
Neural network in ANNarchy
*******************************

An ANNarchy network is a collection of interconnected **Populations**. Each population comprises a set of similar artificial **Neurons**, whose mean-firing rate or spike timing behavior is governed by one or many ordinary differential equations (ODE). These ODEs are dependent on the firing rate of other neurons through **Synapses**. The connection pattern between two populations is called a **Projection**.

The efficiency of the connections received by a neuron is stored in different arrays called **Dendrites**, depending on the type that was assigned to them: realistic neurons do not integrate equally all their inputs, but differentially process them depending on their neurotransmitter type (AMPA, NMDA, GABA, dopamine...), position on the dendritic tree (proximal/distal) or even region of origin (cortical columns do not treat thalamic inputs the same way as long-distance cortico-cortical connections).

This typed organization of afferent connections also allows to easily apply them different learning rules (Hebbian, Anti-Hebbian, dopamine-modulated, Oja, BCM, Covariance-based...).

.. image:: ../_static/neuralnetwork.png
    :width: 70%
    :align: center
    :alt: Structure of a neural network
    
To define a neural network and simulate its behaviour, you need to define the following information:

    * The number of populations, their geometry (number of neurons, optionally the spatial structure - 1D/2D/3D).
    
    * For each population, the type of neuron composing it, with all the necessary ODEs.
    
    * For each projection between two populations, the connection pattern (all-to-all, one-to-one, distance-dependent...), the initial synaptic efficiencies, the delays in synaptic transmission.
    
    * Optionally for a projection, the ODEs describing the evolution of synaptic efficiencies during learning.
    
    * The interaction of the network with its environment (I/O relationships, rewarded tasks, fitting procedure...)
    
ANNarchy provides a convenient way to define this information in a single Python script. 
