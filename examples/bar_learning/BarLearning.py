#   Bar Learning example
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
from ANNarchy import *

# Input neuron: r is set externally
InputNeuron = Neuron(parameters="r = 0.0")

# Leaky neuron
LeakyNeuron = Neuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        tau * dr/dt + r = sum(exc) - sum(inh) : min=0.0
    """
)

# Oja synapse
Oja = Synapse(
    parameters=""" 
        tau = 2000.0 : projection
        alpha = 8.0 : projection
        min_w = 0.0 : projection
    """,
    equations="""
        tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=min_w
    """
)  

# Creating the populations
Input = Population(geometry=(8, 8), neuron=InputNeuron)
Feature = Population(geometry=(8, 4), neuron=LeakyNeuron)

# Creating the projections
ff = Projection(
    pre=Input, 
    post=Feature, 
    target='exc', 
    synapse = Oja    
)
ff.connect_all_to_all(weights = Uniform(-0.5, 0.5))
ff.min_w = -10.0
                     
lat = Projection(
    pre=Feature, 
    post=Feature, 
    target='inh', 
    synapse = Oja
)
lat.connect_all_to_all(weights = Uniform(0.0, 1.0))
lat.alpha = 0.3


# Definition of the environment
def trial():
    # Reset the firing rate for all neurons
    Input.r = 0.0
    # Clamp horizontal bars randomly
    for h in range(Input.geometry[0]):
        if np.random.random() < 1.0/ float(Input.geometry[0]):
            Input[h, :].r = 1.0
    # Clamp vertical bars randomly
    for w in range(Input.geometry[1]):
        if np.random.random() < 1.0/ float(Input.geometry[1]):
            Input[:, w].r = 1.0
    # Simulate for 50ms
    simulate(50.)
    # Return firing rates and receptive fields
    return Input.r, Feature.r, ff.receptive_fields()
    
       
if __name__=='__main__':

    compile()

    # Create and launch the GUI
    from Viz import Viewer
    view = Viewer(func=trial)
    view.run()
