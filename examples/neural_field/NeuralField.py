#   Simple example of a neural field
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
from ANNarchy import *

# Optionally run the simulation on the graphical card
#setup(paradigm="cuda")

# Input neuron just adds noise to the baseline
InputNeuron = Neuron(
    parameters="""
        baseline = 0.0
    """,
    equations="""
        r = pos(baseline + Uniform(-0.5, 0.5))
    """
)

# Neural field neuron
NeuralFieldNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
    """,
    equations="""
        tau * dr/dt + r = sum(exc) + sum(inh) + Uniform(-0.5, 0.5) : min=0.0, max=1.0
    """
)

# Create the populations
N = 20
inp = Population(geometry = (N, N), neuron = InputNeuron)
focus = Population(geometry = (N, N), neuron = NeuralFieldNeuron)

# Create the projections
ff = Projection(pre=inp, post=focus, target='exc')
ff.connect_one_to_one(weights=1.0, delays = 20.0)

lat = Projection(pre=focus, post=focus, target='inh')
lat.connect_dog(amp_pos=0.2, sigma_pos=0.1, amp_neg=0.1, sigma_neg=0.7)

# Analyse and compile everything
compile()

# Import the environment for the simulation (Cython)
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from BubbleWorld import World
world = World(population=inp, radius=0.5, sigma=2.0, period=5000.0, func=step)

# Launch the GUI and run the simulation
from Viz import loop_bubbles
if __name__ == '__main__':
    loop_bubbles(populations = [inp, focus], func=world.rotate, update_rate=200)
