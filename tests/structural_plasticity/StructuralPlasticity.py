#
#	A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1)

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        r = 0.2 : init = 0.1 
    """
)

OutputNeuron = RateNeuron(
    equations="""
        r = sum(exc)
    """
)

NEURON = 30
CONN = 10

input_pop = Population(geometry=(NEURON), neuron=InputNeuron)
output_pop = Population(geometry=(1), neuron=OutputNeuron)

proj = Projection(input_pop, output_pop, 'exc').connect_fixed_number_pre(CONN, 1.0)

compile() # needed to save connectivity matrix

print proj.dendrite(0).rank

rank = raw_input('Add connection to: ')

proj.dendrite(0).add_synapse(int(rank), 1.0)

print proj.dendrite(0).rank

rank = raw_input('Remove connection to: ')

proj.dendrite(0).remove_synapse(int(rank))

print proj.dendrite(0).rank
