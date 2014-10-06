#   Bar Learning example
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach

from ANNarchy import *

setup(paradigm="cuda_35")

# Defining the neuron
InputNeuron = Neuron(   
    parameters="""
        r = 0.0
    """
)

LeakyNeuron = Neuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        tau * dr/dt + r = sum(exc) - sum(inh) : min=0.0
    """
)

# Defining the synapse
Oja = Synapse(
    parameters=""" 
        tau = 2000.0 : postsynaptic
        alpha = 8.0 : postsynaptic
        min_w = 0.0 : postsynaptic
    """,
    equations="""
        tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=min_w
    """
)  


# Creating the populations
nb_neurons = 96
Input = Population(geometry=(nb_neurons, nb_neurons), neuron=InputNeuron)
Feature = Population(geometry=(nb_neurons, 4), neuron=LeakyNeuron)

# Creating the projections
Input_Feature = Projection(
    pre=Input, 
    post=Feature, 
    target='exc', 
    synapse = Oja    
).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
Input_Feature.min_w = -10.0
                     
Feature_Feature = Projection(
    pre=Feature, 
    post=Feature, 
    target='inh', 
    synapse = Oja
).connect_all_to_all( weights = Uniform(0.0, 1.0) )
Feature_Feature.alpha = 0.3


# Definition of the environment
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
    
if __name__=='__main__':
    import sys

    if len(sys.argv) > 1:
        try:
            nt = int(sys.argv[1])
        except:
            nt = 32
    else:
        nt = 32
    config = {
        'device': 1,
        Input: { 'num_threads': nt, 'stream': 0 },
        Feature: { 'num_threads': nt, 'stream': 1 },
        Input_Feature: { 'num_threads': 160, 'stream': 0 },
        Feature_Feature: { 'num_threads': 64, 'stream': 1 }
    }
    set_cuda_config(config)

    compile()
    from datetime import datetime

    # Simulate for 50 ms with a new input
    tstart = datetime.now()

    start_record( { Feature: 'r' } )
    for i in range(5):
        set_input()
        simulate(10) 
    data = get_record()
    print 'Done in', datetime.now() - tstart
