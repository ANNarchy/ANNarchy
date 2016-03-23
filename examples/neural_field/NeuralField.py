#   Simple example of a neural field
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
#
from ANNarchy import *

#setup(paradigm="cuda")

# Define the neuron classes
InputNeuron = Neuron(   
    parameters="""
        baseline = 0.0
    """,
    equations="""
        noise = Uniform(-0.5, 0.5)
        r = pos(baseline + noise)
    """ 
)

NeuralFieldNeuron = Neuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        noise = Uniform(-0.5, 0.5)
        tau * dr/dt + r = sum(exc) + sum(inh) + noise : min=0.0, max=1.0
    """
)

# Create the populations
N = 20
InputPop = Population(geometry = (N, N), neuron = InputNeuron)
FocusPop = Population(geometry = (N, N), neuron = NeuralFieldNeuron)

# Create the projections
ff = Projection(pre=InputPop, post=FocusPop, target='exc')
ff.connect_one_to_one(weights=1.0, delays = 20.0)

lat = Projection(pre=FocusPop, post=FocusPop, target='inh')
lat.connect_dog(amp_pos=0.2, sigma_pos=0.1, amp_neg=0.1, sigma_neg=0.7)

# Analyse and compile everything
compile()   

# Import the environment for the simulation (Cython)
import pyximport; pyximport.install()
from BubbleWorld import World
world = World(pop=InputPop, radius=0.5, sigma=2.0, period=5000.0, func=step)

# Launch the GUI and run
from Viz import loop_bubbles
loop_bubbles(populations = [InputPop, FocusPop], func=world.rotate, update=200)


     


