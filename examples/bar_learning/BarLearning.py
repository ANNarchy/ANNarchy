from ANNarchy4 import *

# Defining the neurons
InputNeuron = Neuron( 
    tau = 10.0, 
    baseline = Variable(init=0.0),
    rate = Variable(init=0.0, eq="tau * drate/dt + rate = baseline", min=0.0)
)

LeakyNeuron = Neuron( 
    tau = 10.0, 
    rate = Variable(init=0.0, eq="tau * drate/dt + rate = sum(exc) - sum(inh)", min=0.0)
)

# Defining the synapses
Oja = Synapse(
    tau = 2000,
    alpha = 8.0,
    value = Variable(
        init=0.0, 
        eq="tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value"
    )
)  

AntiHebb = Synapse(
    tau = 2000,
    alpha = 0.3,
    value = Variable(
        init=0.0, 
        eq="tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value",
        min=0.0
    )
)

# Creating the populations
nb_neurons = 16  
input_pop = Population(name="input", geometry=(nb_neurons, nb_neurons), neuron=InputNeuron)
feature_pop = Population(name="feature", geometry=(15, 4), neuron=LeakyNeuron)

# Creating the projections
input_feature = Projection(
    pre=input_pop, 
    post=feature_pop, 
    target='exc', 
    synapse = Oja,
    connector=Connector('All2All', 
                        weights=RandomDistribution('uniform', [-0.5, 0.5] ) 
                       )
)
                    
feature_feature = Projection(
    pre=feature_pop, 
    post=feature_pop, 
    target='inh', 
    synapse = AntiHebb,
    connector=Connector('All2All', 
                        weights=RandomDistribution('uniform', [0.0, 1.0] ) 
                       )
) 
# Compiling the network
compile()

# Definition of the environment
def set_input():
    # Choose which bars will be used as inputs
    values = np.zeros((nb_neurons, nb_neurons))
    for w in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[:, w] = 1.
    for h in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[h, :] = 1.
    # Set the input
    input_pop.baseline = values.reshape(nb_neurons**2)


# Run the simulation        
for trial in range(3000):
    set_input()
    simulate(50) 

# Visualizing the result of learning    
plot1 = {'pop': input_pop, 'var': 'rate'}
plot2 = {'pop': feature_pop, 'var': 'rate'}
plot3 = {'proj': input_feature, 'var': 'value', 
         'max': 0.1, 'title': 'Receptive fields'}
vis = Visualization( [plot1, plot2, plot3 ] )
vis.render()  

raw_input()
